#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import payload_degraded_reasons, read_json, utc_now_iso, write_json
from scripts.feishu_cards import build_paper_card, select_paper_url
from scripts.llm_client import normalize_structured_summary


REQUIRED_FIELDS = (
    "title",
    "keywords",
    "summary",
    "famous_authors",
    "authors",
    "aminer_paper_url",
    "aminer_author_profiles",
    "author_entries",
)


def _is_valid_absolute_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def validate_paper(paper: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in paper]
    if missing:
        raise ValueError(f"paper missing required fields: {', '.join(missing)}")
    for field, allow_empty in (
        ("keywords", True),
        ("famous_authors", True),
        ("authors", False),
        ("aminer_author_profiles", True),
        ("author_entries", False),
    ):
        value = paper.get(field)
        if not isinstance(value, list):
            raise ValueError(f"paper field must be a list: {field}")
        if field not in {"aminer_author_profiles"} and not allow_empty and not value:
            raise ValueError(f"paper field must be non-empty: {field}")
        for item in value:
            if field == "aminer_author_profiles":
                if not isinstance(item, dict):
                    raise ValueError(f"paper field must contain objects: {field}")
                continue
            if field == "author_entries":
                if not isinstance(item, dict):
                    raise ValueError(f"paper field must contain objects: {field}")
                continue
            if not isinstance(item, str):
                raise ValueError(f"paper field must contain strings: {field}")
    for field in ("title", "summary"):
        if not str(paper.get(field, "")).strip():
            raise ValueError(f"paper field must be non-empty: {field}")
    if "structured_summary" in paper:
        try:
            paper["structured_summary"] = normalize_structured_summary(paper.get("structured_summary"))
        except Exception as exc:
            raise ValueError("paper field must contain a valid structured_summary") from exc
    paper_url = select_paper_url(paper)
    if paper_url and not _is_valid_absolute_url(paper_url):
        raise ValueError("paper URL must be a valid absolute URL")


def collect_degraded_reasons(payload: dict[str, Any]) -> list[str]:
    reasons = payload_degraded_reasons(payload)
    papers = payload.get("papers", [])
    for status_key, reason_key, label in (
        ("aminer_status", "aminer_reason", "AMiner"),
        ("summary_status", "summary_reason", "Summary"),
    ):
        reason = ""
        for paper in papers:
            if str(paper.get(status_key, "success")) == "success":
                continue
            candidate = str(paper.get(reason_key, "")).strip() or "unknown_reason"
            reason = candidate
            break
        if reason:
            labeled_reason = f"{label}={reason}"
            if labeled_reason not in reasons:
                reasons.append(labeled_reason)
    return reasons


def render_message(index: int, paper: dict[str, Any], degraded_reasons: list[str]) -> dict[str, Any]:
    validate_paper(paper)
    card = build_paper_card(index, paper, degraded_reasons)
    return {
        "index": index,
        "arxiv_id": paper.get("arxiv_id", ""),
        "title": paper["title"],
        "card_json": json.dumps(card, ensure_ascii=False, separators=(",", ":")),
    }


def render_feishu_messages(payload: dict[str, Any]) -> dict[str, Any]:
    papers = payload.get("papers", [])
    degraded_reasons = collect_degraded_reasons(payload)
    messages = [
        render_message(index, paper, degraded_reasons if index == 1 else [])
        for index, paper in enumerate(papers, start=1)
    ]
    return {
        "status": payload.get("status", "success"),
        "generated_at": utc_now_iso(),
        "paper_count": len(messages),
        "degraded_reasons": degraded_reasons,
        "profile_topics": list(payload.get("profile_topics") or []),
        "profile_name": str(payload.get("profile_name", "") or ""),
        "profile_source": str(payload.get("profile_source", "") or ""),
        "messages": messages,
        "final_response": "NO_REPLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render Feishu interactive card messages.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = render_feishu_messages(read_json(args.input))
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
