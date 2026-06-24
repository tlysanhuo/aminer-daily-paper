#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import write_json
from scripts.run_pipeline import run_pipeline

MAX_TOPICS = 8
MAX_TOPIC_LENGTH = 80
MAX_PAPER_TITLES = 8
MAX_PAPER_TITLE_LENGTH = 300
MAX_SCHOLAR_NAME_LENGTH = 80
MAX_SCHOLAR_ORG_LENGTH = 160
MAX_FREE_TEXT_LENGTH = 600
MIN_YEAR = 1900
MAX_YEAR = 2100
ALLOWED_PAPERS_FILE_SUFFIXES = {".json"}


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


def _build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run AMiner personalized paper recommendation without Feishu or OpenClaw.",
    )
    parser.add_argument("--base-dir", type=Path, default=Path.cwd())
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


def _validate_text_length(parser: argparse.ArgumentParser, value: str, *, field: str, max_length: int) -> str:
    cleaned = _clean_text(value)
    if len(cleaned) > max_length:
        parser.error(f"{field} must be at most {max_length} characters")
    return cleaned


def _validate_list(
    parser: argparse.ArgumentParser,
    values: list[str],
    *,
    field: str,
    max_count: int,
    max_length: int,
) -> list[str]:
    if len(values) > max_count:
        parser.error(f"{field} accepts at most {max_count} items")
    validated: list[str] = []
    for index, value in enumerate(values, start=1):
        cleaned = _clean_text(value)
        if len(cleaned) > max_length:
            parser.error(f"{field}[{index}] must be at most {max_length} characters")
        if cleaned:
            validated.append(cleaned)
    return validated


def _validate_aminer_user_id(parser: argparse.ArgumentParser, value: str) -> str:
    cleaned = _clean_text(value)
    if cleaned and not re.fullmatch(r"[0-9a-fA-F]{24}", cleaned):
        parser.error("--aminer-user-id must be a 24-character hexadecimal string")
    return cleaned


def _validate_year(parser: argparse.ArgumentParser, value: int, *, field: str) -> int:
    year = int(value or 0)
    if year and not (MIN_YEAR <= year <= MAX_YEAR):
        parser.error(f"{field} must be between {MIN_YEAR} and {MAX_YEAR}")
    return year


def _resolve_papers_file(parser: argparse.ArgumentParser, base_dir: Path, path_text: str) -> str:
    cleaned = _clean_text(path_text)
    if not cleaned:
        return ""

    candidate = Path(cleaned).expanduser()
    resolved_base_dir = base_dir.resolve()
    resolved_candidate = (resolved_base_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_base_dir)
    except ValueError:
        parser.error("--papers-file must stay inside --base-dir")
    if resolved_candidate.suffix.lower() not in ALLOWED_PAPERS_FILE_SUFFIXES:
        parser.error("--papers-file only supports .json files")
    if not resolved_candidate.exists():
        parser.error(f"--papers-file does not exist: {resolved_candidate}")
    return str(resolved_candidate)


def main(argv: Sequence[str] | None = None, *, prog: str | None = None) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)

    base_dir = args.base_dir.resolve()
    config_path = (args.config.resolve() if args.config else _default_config_path(base_dir).resolve())
    output_dir = (args.output_dir.resolve() if args.output_dir else (base_dir / "outputs_cli").resolve())
    output_markdown = args.output_markdown.resolve() if args.output_markdown else output_dir / "recommendation.md"
    output_json = args.output_json.resolve() if args.output_json else output_dir / "recommendation_result.json"

    aminer_user_id = _validate_aminer_user_id(parser, args.aminer_user_id)
    topics = _validate_list(
        parser,
        _split_text_values([*list(args.topics or []), *list(args.topic or [])]),
        field="topics",
        max_count=MAX_TOPICS,
        max_length=MAX_TOPIC_LENGTH,
    )
    scholar_name = _validate_text_length(parser, args.scholar_name, field="scholar_name", max_length=MAX_SCHOLAR_NAME_LENGTH)
    scholar_org = _validate_text_length(parser, args.scholar_org, field="scholar_org", max_length=MAX_SCHOLAR_ORG_LENGTH)
    paper_titles = _validate_list(
        parser,
        [_clean_text(item) for item in list(args.paper_titles or []) if _clean_text(item)],
        field="paper_titles",
        max_count=MAX_PAPER_TITLES,
        max_length=MAX_PAPER_TITLE_LENGTH,
    )
    papers_file = _resolve_papers_file(parser, base_dir, args.papers_file)
    free_text = _validate_text_length(parser, args.free_text, field="free_text", max_length=MAX_FREE_TEXT_LENGTH)
    start_year = _validate_year(parser, args.start_year, field="--start-year")
    end_year = _validate_year(parser, args.end_year, field="--end-year")
    if start_year and end_year and start_year > end_year:
        parser.error("--start-year cannot be greater than --end-year")

    if not _has_profile_input(args, topics):
        parser.error(
            "provide at least one profile signal: --topics, --free-text, "
            "--scholar-name, --aminer-user-id, --paper-title, or --papers-file"
        )

    result = run_pipeline(
        base_dir=base_dir,
        output_dir=output_dir,
        config=_load_yaml(config_path),
        aminer_user_id=aminer_user_id,
        topics=topics,
        scholar_name=scholar_name,
        scholar_org=scholar_org,
        paper_titles=paper_titles,
        papers_file=papers_file,
        free_text=free_text,
        language_sort=_clean_text(args.language_sort),
        start_year=start_year,
        end_year=end_year,
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
