from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any

STATUS_PRIORITY = {"success": 0, "partial_success": 1, "degraded": 2}


def clean_text(value: Any) -> str:
    """Normalize whitespace and strip text. Consolidated from multiple duplicate definitions."""
    return " ".join(str(value or "").split()).strip()


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Deduplicate strings while preserving order, case-insensitive."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = clean_text(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def strip_tags(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(text or ""))).strip()


def normalize_arxiv_id(raw_id: str) -> str:
    raw = raw_id.strip().rstrip("/")
    raw = raw.rsplit("/", 1)[-1]
    return re.sub(r"v\d+$", "", raw)


def read_topics(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"topics file not found: {path}")
    topics = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not topics:
        raise ValueError(f"topics file is empty: {path}")
    return topics


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.isoformat() if isinstance(value, datetime) else str(value),
        )
        + "\n",
        encoding="utf-8",
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


def summarize_statuses(statuses: list[str]) -> str:
    unique = set(statuses)
    if unique == {"success"}:
        return "success"
    if "partial_success" in unique or ("success" in unique and len(unique) > 1):
        return "partial_success"
    return "degraded"


def combine_stage_statuses(*statuses: str) -> str:
    normalized = [str(status or "success").strip() or "success" for status in statuses]
    valid = [status if status in STATUS_PRIORITY else "success" for status in normalized]
    return max(valid or ["success"], key=lambda status: STATUS_PRIORITY[status])


def payload_degraded_reasons(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    reason = str(payload.get("degraded_reason", "")).strip()
    if reason:
        reasons.append(reason)
    extra = payload.get("degraded_reasons")
    if isinstance(extra, list):
        for item in extra:
            text = str(item).strip()
            if text and text not in reasons:
                reasons.append(text)
    return reasons


def merge_payload_status(payload: dict[str, Any], upstream_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(payload)
    merged["status"] = combine_stage_statuses(upstream_payload.get("status", "success"), payload.get("status", "success"))
    reasons = payload_degraded_reasons(upstream_payload)
    for reason in payload_degraded_reasons(payload):
        if reason not in reasons:
            reasons.append(reason)
    if reasons:
        merged["degraded_reasons"] = reasons
        if len(reasons) == 1:
            merged["degraded_reason"] = reasons[0]
    else:
        merged.pop("degraded_reason", None)
        merged.pop("degraded_reasons", None)
    return merged


def first_nonempty(*values: str) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""
