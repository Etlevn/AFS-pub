from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import threading
from typing import Any, Callable

from dotenv import load_dotenv
from tqdm.auto import tqdm

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


SUPPORTED_AGENT_MODES = {"livetrading", "nowcasting"}


class TeeStream:
    """Mirror writes to terminal and an opened log file."""

    def __init__(self, terminal_stream, log_file, suppress_progress_artifacts: bool = False):
        self._terminal_stream = terminal_stream
        self._log_file = log_file
        self._suppress_progress_artifacts = suppress_progress_artifacts
        self._line_buffer = ""
        self._last_emitted_blank_line = False
        self._ansi_csi_pattern = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
        self._tqdm_bar_pattern = re.compile(
            r"\s*\d{1,3}%\|[^|\n]*\|\s*\d+/\d+\s*\[[^\]\n]*(?:it/s|s/it|\?it/s)\]"
        )

    def write(self, data: str) -> int:
        self._terminal_stream.write(data)
        filtered = self._prepare_log_payload(data)
        if filtered:
            self._write_log_payload(filtered)
        return len(data)

    def flush(self) -> None:
        self._terminal_stream.flush()
        if self._suppress_progress_artifacts and self._line_buffer:
            self._emit_log_line(self._line_buffer, ends_with_newline=False)
            self._line_buffer = ""
        self._log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._terminal_stream, "isatty", lambda: False)())

    def __getattr__(self, name: str):
        return getattr(self._terminal_stream, name)

    def _prepare_log_payload(self, data: str) -> str:
        """Sanitize log payload while keeping terminal output unchanged."""
        if not self._suppress_progress_artifacts:
            return data
        if not data:
            return ""

        sanitized = self._ansi_csi_pattern.sub("", data)
        if "\r" in sanitized:
            # tqdm reuses the same terminal row; keep only the final segment.
            sanitized = sanitized.rsplit("\r", 1)[-1]
        sanitized = self._tqdm_bar_pattern.sub("", sanitized)

        if not sanitized:
            return ""

        return sanitized

    def _write_log_payload(self, payload: str) -> None:
        if not self._suppress_progress_artifacts:
            self._log_file.write(payload)
            return

        self._line_buffer += payload
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._emit_log_line(line, ends_with_newline=True)

    def _emit_log_line(self, line: str, ends_with_newline: bool) -> None:
        is_blank_line = line.strip() == ""
        if is_blank_line and self._last_emitted_blank_line:
            return

        if ends_with_newline:
            self._log_file.write(f"{line}\n")
        elif line:
            self._log_file.write(line)
        self._last_emitted_blank_line = is_blank_line


def build_runtime_config(
    runtime_options: dict[str, Any], default_options: dict[str, Any]
) -> dict[str, Any]:
    """Build TradingAgents config from resolved runtime options."""
    runtime_config = DEFAULT_CONFIG.copy()
    runtime_config["llm_provider"] = runtime_options["provider"]
    # Single-agent workflow primarily uses one model; keep both keys for compatibility.
    runtime_config["quick_think_llm"] = runtime_options["model"]
    runtime_config["deep_think_llm"] = runtime_options["model"]
    runtime_config["agent_mode"] = runtime_options["agent_mode"]
    runtime_config["trace_level"] = runtime_options["trace_level"]
    runtime_config["max_debate_rounds"] = runtime_options["max_debate_rounds"]
    runtime_config["data_vendors"] = default_options.get(
        "data_vendors",
        {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
        },
    )
    runtime_config["model"] = runtime_options["model"]
    return runtime_config


def parse_tickers(raw_tickers: str) -> list[str]:
    tickers = [ticker.strip().upper() for ticker in raw_tickers.split(",")]
    return [ticker for ticker in tickers if ticker]


def normalize_config_tickers(raw_tickers: Any) -> str:
    """Support both comma-separated string and list for tickers."""
    if isinstance(raw_tickers, list):
        return ",".join(str(ticker).strip() for ticker in raw_tickers)
    if isinstance(raw_tickers, str):
        return raw_tickers
    raise ValueError("`tickers` in config.py must be a list or comma-separated string.")


def normalize_agent_mode(raw_mode: Any) -> str:
    """Normalize agent mode to one of the supported v2 workflows."""
    mode = str(raw_mode or "livetrading").strip().lower()
    if mode not in SUPPORTED_AGENT_MODES:
        raise ValueError(
            "`agent_mode` must be one of: livetrading, nowcasting."
        )
    return mode


def normalize_trace_level(raw_level: Any) -> str:
    """Normalize trace level to one of the supported values."""
    level = str(raw_level or "basic").strip().lower()
    if level not in {"basic", "full", "audit"}:
        raise ValueError("`trace_level` must be one of: basic, full, audit.")
    return level


def load_main_defaults(config_path: Path) -> dict[str, Any]:
    """Load runtime defaults from config.py."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. Create it or pass --config."
        )
    if config_path.suffix != ".py":
        raise ValueError(f"Config file must be a .py file: {config_path}")

    module_name = f"_runtime_config_{config_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load config module from: {config_path}")
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    raw = getattr(config_module, "MAIN_CONFIG", None)
    if raw is None:
        raw = getattr(config_module, "CONFIG", None)
    if raw is None or not isinstance(raw, dict):
        raise ValueError(
            f"{config_path} must define a dict named MAIN_CONFIG (or CONFIG)."
        )
    required_keys = [
        "tickers",
        "max_workers",
        "provider",
        "model",
    ]
    missing = [key for key in required_keys if key not in raw]
    if missing:
        raise ValueError(f"Missing required keys in {config_path}: {', '.join(missing)}")
    return raw


def resolve_runtime_options(
    args: argparse.Namespace, defaults: dict[str, Any]
) -> dict[str, Any]:
    """Apply precedence: CLI args > config.py defaults."""
    return {
        "tickers": args.tickers
        if args.tickers is not None
        else normalize_config_tickers(defaults["tickers"]),
        "date": args.date
        if args.date is not None
        else defaults.get("date", datetime.now().strftime("%Y-%m-%d")),
        "max_workers": args.max_workers
        if args.max_workers is not None
        else int(defaults["max_workers"]),
        "agent_mode": normalize_agent_mode(
            args.agent_mode
            if args.agent_mode is not None
            else defaults.get("agent_mode", "livetrading")
        ),
        "trace_level": normalize_trace_level(
            args.trace_level
            if args.trace_level is not None
            else defaults.get("trace_level", "basic")
        ),
        "provider": args.provider if args.provider is not None else defaults["provider"],
        "model": (
            args.model
            if args.model is not None
            else defaults.get("model")
            or defaults.get("deep_model")
            or defaults.get("quick_model")
        ),
        "max_debate_rounds": args.max_debate_rounds
        if args.max_debate_rounds is not None
        else int(defaults.get("max_debate_rounds", 1)),
        "output_dir": args.output_dir
        if args.output_dir is not None
        else "output",
        "debug": args.debug
        if args.debug is not None
        else bool(defaults.get("debug", False)),
        "log": args.log if args.log is not None else bool(defaults.get("log", False)),
        "selected_analysts": [],
    }


def analyze_one(
    ticker: str,
    trade_date: str,
    runtime_config: dict[str, Any],
    debug: bool,
    flow_index: int,
    flow_total: int,
    selected_analysts: list[str],
    run_start_monotonic: float | None = None,
    thread_code_allocator: Callable[[str], str] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run one ticker analysis and return normalized result payload."""
    thread_name = threading.current_thread().name
    thread_code = (
        thread_code_allocator(thread_name)
        if callable(thread_code_allocator)
        else None
    )
    try:
        graph = TradingAgentsGraph(
            selected_analysts=selected_analysts, debug=debug, config=runtime_config
        )
        trace_context = {
            "flow_index": flow_index,
            "flow_total": flow_total,
            "thread_name": thread_name,
            "ticker": ticker,
        }
        if thread_code:
            trace_context["thread_code"] = thread_code
        if isinstance(run_start_monotonic, (int, float)):
            trace_context["run_start_monotonic"] = float(run_start_monotonic)
        if progress_callback is not None:
            trace_context["progress_callback"] = progress_callback
        if callable(log_callback):
            trace_context["log_callback"] = log_callback
        final_state, decision = graph.propagate(
            ticker,
            trade_date,
            trace_context=trace_context,
        )
        return {
            "ticker": ticker,
            "trade_date": trade_date,
            "status": "ok",
            "rating": str(decision).strip().upper(),
            "final_trade_decision": final_state.get("final_trade_decision"),
            "forecast_decision": final_state.get("forecast_decision"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ticker": ticker,
            "trade_date": trade_date,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_batch(
    tickers: list[str],
    trade_date: str,
    runtime_config: dict[str, Any],
    max_workers: int,
    debug: bool,
    selected_analysts: list[str],
    agent_mode: str,
    run_start_monotonic: float | None = None,
    log_capture: bool = False,
    log_sink: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run tickers concurrently and preserve completion results."""
    results: list[dict[str, Any]] = []
    total_flows = len(tickers)
    if agent_mode not in SUPPORTED_AGENT_MODES:
        raise ValueError(
            f"Unsupported agent_mode '{agent_mode}'. Use one of {sorted(SUPPORTED_AGENT_MODES)}."
        )
    progress_total = total_flows
    progress_lock = threading.Lock()
    thread_code_lock = threading.Lock()
    thread_code_map: dict[str, str] = {}
    next_thread_code = 1

    print("")
    # Keep the progress bar on the real terminal only, so it never gets captured
    # by TeeStream and therefore never appears in log files.
    progress_bar = tqdm(
        total=progress_total,
        leave=True,
        file=sys.__stdout__,
        dynamic_ncols=True,
        mininterval=0.2,
        disable=not bool(getattr(sys.__stdout__, "isatty", lambda: False)()),
    )

    def emit_log(line: str) -> None:
        with progress_lock:
            # Keep the bar at the bottom by routing terminal logs through tqdm.write.
            tqdm.write(line, file=sys.__stdout__)
            # Persist the same line to log file without passing through terminal stream.
            if log_capture and callable(log_sink):
                log_sink(line)

    active_log_callback = emit_log if debug else None

    def allocate_thread_code(thread_name: str) -> str:
        nonlocal next_thread_code
        with thread_code_lock:
            if thread_name in thread_code_map:
                return thread_code_map[thread_name]
            code = f"{next_thread_code:03d}"
            thread_code_map[thread_name] = code
            next_thread_code += 1
            if next_thread_code > 999:
                next_thread_code = 1
            return code

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                analyze_one,
                ticker,
                trade_date,
                runtime_config,
                debug,
                index,
                total_flows,
                selected_analysts,
                run_start_monotonic,
                allocate_thread_code,
                None,
                active_log_callback,
            ): ticker
            for index, ticker in enumerate(tickers, start=1)
        }
        for future in as_completed(future_map):
            results.append(future.result())
            with progress_lock:
                progress_bar.update(1)
    progress_bar.close()
    return results


def write_main_report(
    output_dir: Path, trade_date: str, results: list[dict[str, Any]], run_id: str
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}.json"
    payload = []
    for item in sorted(results, key=lambda entry: entry["ticker"]):
        if item.get("status") != "ok":
            continue
        forecast_decision = item.get("forecast_decision")
        final_trade_decision = item.get("final_trade_decision")
        if isinstance(forecast_decision, dict):
            payload.append(forecast_decision)
        elif isinstance(final_trade_decision, dict):
            payload.append(final_trade_decision)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run non-interactive TradingAgents batch analysis."
    )
    parser.add_argument(
        "--config",
        default="config.py",
        help="Path to Python config file with default runtime parameters.",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="Comma-separated ticker symbols.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Trade date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum parallel workers.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider override (e.g. openai, ollama, deepseek).",
    )
    parser.add_argument(
        "--agent-mode",
        default=None,
        choices=sorted(SUPPORTED_AGENT_MODES),
        help="Execution workflow: livetrading or nowcasting.",
    )
    parser.add_argument(
        "--trace-level",
        default=None,
        choices=["basic", "full", "audit"],
        help="Debug trace verbosity: basic (latest message), full (all messages), audit (full + metadata).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model override for single-agent workflows.",
    )
    parser.add_argument(
        "--max-debate-rounds",
        type=int,
        default=None,
        help="Maximum investment debate rounds.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write JSON output.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable graph debug stream output.",
    )
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable terminal output logging to log/<timestamp>.log.",
    )
    return parser.parse_args()


def _run_main_flow(
    run_start_monotonic: float,
    runtime_options: dict[str, Any],
    defaults: dict[str, Any],
    run_id: str,
    log_sink: Callable[[str], None] | None = None,
) -> None:
    """Execute one full non-interactive batch run."""

    tickers = parse_tickers(runtime_options["tickers"])
    if not tickers:
        raise ValueError("No tickers provided. Set `tickers` in config or pass --tickers.")

    print("-" * 60)
    print(f"date:\t\t{runtime_options['date']}")
    print(f"tickers:\t{len(tickers)} ({','.join(tickers)})")
    print(f"workers:\t{max(1, runtime_options['max_workers'])}")
    print(f"mode:\t\t{runtime_options['agent_mode']}")
    print(f"trace level:\t{runtime_options['trace_level']}")
    print(f"provider:\t{runtime_options['provider']}")
    print(f"model:\t\t{runtime_options['model']}")
    print("analysts:\t(single-agent)")
    print(f"debug:\t\t{runtime_options['debug']}")
    print(f"log:\t\t{runtime_options['log']}")
    print("-" * 60)

    runtime_config = build_runtime_config(runtime_options, defaults)
    results = run_batch(
        tickers=tickers,
        trade_date=runtime_options["date"],
        runtime_config=runtime_config,
        max_workers=max(1, runtime_options["max_workers"]),
        debug=runtime_options["debug"],
        selected_analysts=runtime_options["selected_analysts"],
        agent_mode=runtime_options["agent_mode"],
        run_start_monotonic=run_start_monotonic,
        log_capture=bool(runtime_options.get("log")),
        log_sink=log_sink,
    )
    output_path = write_main_report(
        Path(runtime_options["output_dir"]), runtime_options["date"], results, run_id
    )

    success = sum(1 for item in results if item["status"] == "ok")
    failed = len(results) - success
    print(f"\n[DONE] total={len(results)} | ok={success} | err={failed} | report={output_path}")


def main() -> None:
    run_start_monotonic = time.perf_counter()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    load_dotenv()
    args = parse_args()
    defaults = load_main_defaults(Path(args.config))
    runtime_options = resolve_runtime_options(args, defaults)

    if runtime_options["log"]:
        log_dir = Path("log")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            tee_stdout = TeeStream(
                sys.stdout, log_file, suppress_progress_artifacts=True
            )
            tee_stderr = TeeStream(
                sys.stderr, log_file, suppress_progress_artifacts=True
            )

            def write_log_only(line: str) -> None:
                payload = line if line.endswith("\n") else f"{line}\n"
                log_file.write(payload)
                log_file.flush()

            with redirect_stdout(tee_stdout), redirect_stderr(tee_stderr):
                print(f"[LOG] output capture started -> {log_path}")
                _run_main_flow(
                    run_start_monotonic,
                    runtime_options,
                    defaults,
                    run_id,
                    log_sink=write_log_only,
                )
                print(f"[LOG] output capture finished -> {log_path}")
    else:
        _run_main_flow(run_start_monotonic, runtime_options, defaults, run_id)


if __name__ == "__main__":
    main()
