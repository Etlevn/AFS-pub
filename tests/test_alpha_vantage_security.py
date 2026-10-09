"""Verify credentials cannot escape through Alpha Vantage tool failures."""
import os
import traceback
import unittest
from unittest.mock import Mock, patch

import requests

from tradingagents.dataflows import alpha_vantage_common as av


class AlphaVantageSecurityTests(unittest.TestCase):
    def setUp(self):
        self.key = "test-credential-do-not-publish"
        self.env = patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": self.key})
        self.env.start()
        self.addCleanup(self.env.stop)

    def assert_safe_failure(self, failure, expected):
        with patch.object(av.requests, "get", side_effect=failure):
            try:
                av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
            except requests.RequestException as exc:
                formatted = "".join(traceback.format_exception(exc))
                self.assertNotIn(self.key, formatted)
                self.assertNotIn("apikey=", formatted)
                self.assertIn(expected, str(exc))
            else:
                self.fail("Expected a sanitized request failure")

    def test_http_error_hides_url_and_key(self):
        response = requests.Response()
        response.status_code = 403
        failure = requests.HTTPError(
            f"403 for https://example.invalid/query?apikey={self.key}",
            response=response,
        )
        self.assert_safe_failure(failure, "HTTP 403")

    def test_transport_error_hides_url_and_key(self):
        self.assert_safe_failure(
            requests.ConnectionError(f"Connection failed: apikey={self.key}"),
            "ConnectionError",
        )

    def test_vendor_echo_is_redacted(self):
        response = Mock(text=f'{{"Information":"api key {self.key} is invalid"}}')
        with patch.object(av.requests, "get", return_value=response) as get:
            with self.assertRaises(av.AlphaVantageRateLimitError) as result:
                av._make_api_request("TIME_SERIES_DAILY", {"symbol": "AAPL"})
            self.assertNotIn(self.key, str(result.exception))
            self.assertIn("[REDACTED]", str(result.exception))
            self.assertEqual(get.call_args.kwargs["timeout"], 30)

    def test_success_preserves_csv_and_parameters(self):
        csv = "timestamp,close\n2026-01-01,100\n"
        response = Mock(text=csv)
        params = {"symbol": "AAPL"}
        with patch.object(av.requests, "get", return_value=response) as get:
            self.assertEqual(av._make_api_request("TIME_SERIES_DAILY", params), csv)
            self.assertEqual(params, {"symbol": "AAPL"})
            self.assertEqual(get.call_args.kwargs["params"]["apikey"], self.key)


if __name__ == "__main__":
    unittest.main()
