from __future__ import annotations

import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.aminer_person_search import _post_json


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeOpener:
    def __init__(self, side_effects: list[object]) -> None:
        self.side_effects = list(side_effects)
        self.calls = 0

    def open(self, request, timeout=20):  # noqa: ANN001
        self.calls += 1
        if not self.side_effects:
            raise AssertionError("no side effects configured")
        effect = self.side_effects.pop(0)
        if isinstance(effect, BaseException):
            raise effect
        return effect


class AminerPersonSearchTests(unittest.TestCase):
    def test_post_json_retries_urlerror_then_succeeds(self) -> None:
        opener = _FakeOpener(
            [
                urllib.error.URLError("EOF occurred in violation of protocol"),
                _FakeResponse({"success": True, "data": [{"id": "person-1"}]}),
            ]
        )
        with patch("scripts.aminer_person_search._build_https_opener", return_value=opener), patch(
            "scripts.aminer_person_search.time.sleep",
            return_value=None,
        ):
            payload = _post_json(
                "https://example.com/person/search",
                {"name": "张帆进"},
                headers={"Content-Type": "application/json"},
                timeout=5,
                retry_attempts=2,
            )
        self.assertEqual(opener.calls, 2)
        self.assertTrue(payload["success"])

    def test_post_json_returns_compact_unreachable_error_after_retries(self) -> None:
        opener = _FakeOpener(
            [
                urllib.error.URLError("EOF occurred in violation of protocol"),
                urllib.error.URLError("EOF occurred in violation of protocol"),
            ]
        )
        with patch("scripts.aminer_person_search._build_https_opener", return_value=opener), patch(
            "scripts.aminer_person_search.time.sleep",
            return_value=None,
        ):
            with self.assertRaises(RuntimeError) as exc_info:
                _post_json(
                    "https://example.com/person/search",
                    {"name": "张帆进"},
                    headers={"Content-Type": "application/json"},
                    timeout=5,
                    retry_attempts=2,
                )
        self.assertIn("aminer_request_unreachable", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
