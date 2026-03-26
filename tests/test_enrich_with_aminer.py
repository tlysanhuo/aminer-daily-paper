from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.enrich_with_aminer import _post_json


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeOpener:
    def __init__(self, events):
        self._events = list(events)
        self.calls = 0

    def open(self, request, timeout=0):
        self.calls += 1
        event = self._events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


class EnrichWithAminerTests(unittest.TestCase):
    def test_post_json_retries_ssl_eof_and_succeeds(self) -> None:
        opener = _FakeOpener(
            [
                urllib.error.URLError("EOF occurred in violation of protocol"),
                _FakeResponse({"data": [{"id": "paper-1"}]}),
            ]
        )
        with patch("scripts.enrich_with_aminer._build_https_opener", return_value=opener):
            with patch("scripts.enrich_with_aminer.time.sleep"):
                payload = _post_json("https://example.com", {"ids": ["paper-1"]}, "token")

        self.assertEqual(payload["data"][0]["id"], "paper-1")
        self.assertEqual(opener.calls, 2)

    def test_post_json_raises_after_retryable_transport_failures(self) -> None:
        opener = _FakeOpener(
            [
                urllib.error.URLError("EOF occurred in violation of protocol"),
                ssl.SSLError("handshake failure"),
                urllib.error.URLError("EOF occurred in violation of protocol"),
            ]
        )
        with patch("scripts.enrich_with_aminer._build_https_opener", return_value=opener):
            with patch("scripts.enrich_with_aminer.time.sleep"):
                with self.assertRaises(RuntimeError) as exc_info:
                    _post_json("https://example.com", {"ids": ["paper-1"]}, "token")

        self.assertIn("aminer_request_unreachable", str(exc_info.exception))
        self.assertEqual(opener.calls, 3)


if __name__ == "__main__":
    unittest.main()
