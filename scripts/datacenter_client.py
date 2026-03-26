from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


DEFAULT_SEGMENTATION_URL = ""
DEFAULT_TIMEOUT_SECONDS = 15


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def resolve_segmentation_url(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    datacenter_config = config.get("datacenter") if isinstance(config.get("datacenter"), dict) else {}
    explicit = _clean_text(datacenter_config.get("segmentation_url") or os.getenv("DATACENTER_SEGMENTATION_URL"))
    return explicit or DEFAULT_SEGMENTATION_URL


def resolve_segmentation_timeout(config: dict[str, Any] | None = None) -> int:
    config = config or {}
    datacenter_config = config.get("datacenter") if isinstance(config.get("datacenter"), dict) else {}
    try:
        return max(int(datacenter_config.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS), 1)
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT_SECONDS


def call_segmentation_pro(
    *,
    query: str,
    user_id: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    segmentation_url = resolve_segmentation_url(config)
    if not _clean_text(segmentation_url):
        raise RuntimeError("missing_segmentation_url")
    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    token = _clean_text(aminer_config.get("token") or os.getenv("AMINER_TOKEN"))
    payload = json.dumps({"query": _clean_text(query), "userId": _clean_text(user_id)}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(segmentation_url, data=payload, headers=headers, method="POST")
    timeout_seconds = resolve_segmentation_timeout(config)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout_seconds) as response:  # nosec B310
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"segmentation_http_{exc.code}:{detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"segmentation_unreachable:{exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("segmentation_invalid_json") from exc
    if not parsed.get("success"):
        raise RuntimeError(_clean_text(parsed.get("message")) or "segmentation_unsuccessful")
    data = parsed.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("segmentation_missing_data")
    return data
