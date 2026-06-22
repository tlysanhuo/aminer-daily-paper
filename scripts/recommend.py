#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import write_json
from scripts.run_pipeline import run_pipeline


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _split_text_values(values: list[str]) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        for piece in re.split(r"[,，;/；、\n]+", str(value or "")):
            item = _clean_text(piece)
            key = item.casefold()
            if not item or key in seen:
                continue
            seen.add(key)
            items.append(item)
    return items


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _default_config_path(base_dir: Path) -> Path:
    local_config = base_dir / "config.yaml"
    if local_config.exists():
        return local_config
    return base_dir / "config.example.yaml"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run AMiner personalized paper recommendation without Feishu or OpenClaw.",
    )
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--aminer-user-id", default="")
    parser.add_argument("--topics", nargs="*", default=[])
    parser.add_argument("--topic", action="append", default=[])
    parser.add_argument("--scholar-name", default="")
    parser.add_argument("--scholar-org", default="")
    parser.add_argument("--paper-title", action="append", dest="paper_titles", default=[])
    parser.add_argument("--papers-file", default="")
    parser.add_argument("--free-text", default="")

    parser.add_argument("--language-sort", choices=("zh", "en"), default="")
    parser.add_argument("--start-year", type=int, default=0)
    parser.add_argument("--end-year", type=int, default=0)
    return parser


def _has_profile_input(args: argparse.Namespace, topics: list[str]) -> bool:
    paper_titles = [_clean_text(item) for item in list(args.paper_titles or []) if _clean_text(item)]
    return bool(
        _clean_text(args.aminer_user_id)
        or topics
        or _clean_text(args.scholar_name)
        or _clean_text(args.scholar_org)
        or paper_titles
        or _clean_text(args.papers_file)
        or _clean_text(args.free_text)
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    base_dir = args.base_dir.resolve()
    config_path = (args.config.resolve() if args.config else _default_config_path(base_dir).resolve())
    output_dir = (args.output_dir.resolve() if args.output_dir else (base_dir / "outputs_cli").resolve())
    output_markdown = args.output_markdown.resolve() if args.output_markdown else output_dir / "recommendation.md"
    output_json = args.output_json.resolve() if args.output_json else output_dir / "recommendation_result.json"

    topics = _split_text_values([*list(args.topics or []), *list(args.topic or [])])
    paper_titles = [_clean_text(item) for item in list(args.paper_titles or []) if _clean_text(item)]
    if not _has_profile_input(args, topics):
        parser.error(
            "provide at least one profile signal: --topics, --free-text, "
            "--scholar-name, --aminer-user-id, --paper-title, or --papers-file"
        )

    result = run_pipeline(
        base_dir=base_dir,
        output_dir=output_dir,
        config=_load_yaml(config_path),
        aminer_user_id=_clean_text(args.aminer_user_id),
        topics=topics,
        scholar_name=_clean_text(args.scholar_name),
        scholar_org=_clean_text(args.scholar_org),
        paper_titles=paper_titles,
        papers_file=_clean_text(args.papers_file),
        free_text=_clean_text(args.free_text),
        language_sort=_clean_text(args.language_sort),
        start_year=int(args.start_year or 0),
        end_year=int(args.end_year or 0),
        target="",
        account_id="local",
        skip_dispatch=True,
    )

    if result.get("final_response") != "TEXT":
        raise RuntimeError(f"unexpected final_response: {result.get('final_response')}")
    reply_text = str(result.get("reply_text") or "").strip()
    if not reply_text:
        raise RuntimeError("pipeline returned empty recommendation text")

    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(reply_text + "\n", encoding="utf-8")
    write_json(output_json, result)

    print(
        json.dumps(
            {
                "status": result.get("status", "success"),
                "mode": result.get("mode", ""),
                "markdown": str(output_markdown),
                "json": str(output_json),
                "artifacts": {
                    "profile": result.get("profile_path", ""),
                    "candidates": result.get("candidates_path", ""),
                    "ranked": result.get("ranked_path", ""),
                    "summarized": result.get("summarized_path", ""),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
