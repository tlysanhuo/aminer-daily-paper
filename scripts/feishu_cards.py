from __future__ import annotations

import re
from typing import Any

from scripts.aminer_schema import render_author_markdown
from scripts.llm_client import SUMMARY_ALL_KEYS, SUMMARY_SECTION_LABELS


def markdown_block(title: str, content: str) -> dict[str, Any]:
    return {
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": f"**{title}**\n{content.strip() or '暂无'}",
        },
    }


def plain_markdown_block(content: str) -> dict[str, Any]:
    return {
        "tag": "div",
        "text": {
            "tag": "lark_md",
            "content": content.strip() or "暂无",
        },
    }


def _normalize_summary_sentence(text: str) -> str:
    compact = re.sub(r"\s+", " ", str(text).strip())
    if not compact:
        return ""
    return compact.rstrip("。.;；,，")


def _extract_paper_year(paper: dict[str, Any]) -> str:
    year = str(paper.get("year") or "").strip()
    if re.fullmatch(r"\d{4}", year):
        return year
    for field in ("published", "published_date"):
        value = str(paper.get(field) or "").strip()
        matched = re.search(r"\b(19|20)\d{2}\b", value)
        if matched:
            return matched.group(0)
    return ""


def _split_famous_author_entries(items: list[Any]) -> list[str]:
    authors: list[str] = []
    for item in items:
        compact = str(item).strip()
        if not compact:
            continue
        parts = re.split(
            r"(?:\s+(?=(?:[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){1,5}|[\u4e00-\u9fff·]{2,20})[:：]))",
            compact,
        )
        for part in parts:
            normalized = part.strip()
            if normalized:
                authors.append(normalized)
    return authors


def _normalize_person_name(text: str) -> str:
    return "".join(char for char in str(text or "").casefold() if char.isalnum())


def _extract_famous_author_name(text: str) -> str:
    compact = str(text or "").strip()
    if not compact:
        return ""
    for pattern in (
        r"^\s*([A-Za-z][A-Za-z .'\-]{1,80}|[\u4e00-\u9fff·]{2,30})\s*[:：]",
        r"^\s*([A-Za-z][A-Za-z .'\-]{1,80}|[\u4e00-\u9fff·]{2,30})\s*[，,]",
    ):
        matched = re.match(pattern, compact)
        if matched:
            return matched.group(1).strip()
    first_token = re.split(r"\s*[，,:：]\s*|\s+来自", compact, maxsplit=1)[0].strip()
    return first_token


def _link_famous_author_entry(entry: str, profiles: list[dict[str, Any]]) -> str:
    compact = str(entry or "").strip()
    if not compact:
        return ""
    candidate_name = _extract_famous_author_name(compact)
    if not candidate_name:
        return compact
    normalized_candidate = _normalize_person_name(candidate_name)
    if not normalized_candidate:
        return compact
    for profile in profiles:
        profile_url = str(profile.get("profile_url") or "").strip()
        if not profile_url:
            continue
        names = [
            str(profile.get("name") or "").strip(),
            str(profile.get("name_zh") or "").strip(),
            str(profile.get("query_name") or "").strip(),
        ]
        if not any(_normalize_person_name(name) == normalized_candidate for name in names if name):
            continue
        return re.sub(
            re.escape(candidate_name),
            f"[{candidate_name}]({profile_url})",
            compact,
            count=1,
        )
    return compact


def select_paper_url(paper: dict[str, Any]) -> str:
    return (
        str(paper.get("aminer_paper_url", "")).strip()
        or str(paper.get("abs_url", "")).strip()
        or str(paper.get("pdf_url", "")).strip()
    )


def _summary_blocks_from_structured_summary(paper: dict[str, Any]) -> list[dict[str, Any]]:
    structured_summary = paper.get("structured_summary")
    if not isinstance(structured_summary, dict):
        return []
    parts: list[str] = []
    for key in SUMMARY_ALL_KEYS:
        text = _normalize_summary_sentence(structured_summary.get(key, ""))
        if not text:
            continue
        parts.append(text)
    if not parts:
        return []
    return [markdown_block("小结", "；".join(parts) + "。")]


def _summary_blocks_from_text(summary_text: str) -> list[dict[str, Any]]:
    parts: list[str] = []
    for line in summary_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = False
        for key in SUMMARY_ALL_KEYS:
            label = SUMMARY_SECTION_LABELS[key]
            prefix = f"{label}："
            alt_prefix = f"{label}:"
            if stripped.startswith(prefix):
                content = _normalize_summary_sentence(stripped[len(prefix) :])
                if content:
                    parts.append(content)
                matched = True
                break
            if stripped.startswith(alt_prefix):
                content = _normalize_summary_sentence(stripped[len(alt_prefix) :])
                if content:
                    parts.append(content)
                matched = True
                break
        if not matched:
            content = _normalize_summary_sentence(stripped)
            if content:
                parts.append(content)
    if not parts:
        return []
    return [markdown_block("小结", "；".join(parts) + "。")]


def render_summary_blocks(paper: dict[str, Any]) -> list[dict[str, Any]]:
    structured_blocks = _summary_blocks_from_structured_summary(paper)
    if structured_blocks:
        return structured_blocks
    summary_text = str(paper.get("summary", "")).strip()
    parsed_blocks = _summary_blocks_from_text(summary_text)
    if parsed_blocks:
        return parsed_blocks
    normalized = _normalize_summary_sentence(summary_text)
    return [markdown_block("小结", normalized + "。")] if normalized else []


def render_famous_author_blocks(paper: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = [item for item in list(paper.get("aminer_author_profiles") or []) if isinstance(item, dict)]
    authors = [_link_famous_author_entry(item, profiles) for item in _split_famous_author_entries(list(paper.get("famous_authors") or []))]
    authors = [item for item in authors if item]
    if not authors:
        return []
    blocks = [markdown_block("大牛作者", authors[0])]
    blocks.extend(plain_markdown_block(author) for author in authors[1:])
    return blocks


def build_paper_card(index: int, paper: dict[str, Any], degraded_reasons: list[str]) -> dict[str, Any]:
    keywords = " / ".join(paper.get("keywords") or []) or "暂无"
    year = _extract_paper_year(paper) or "暂无"
    authors = render_author_markdown(
        paper.get("author_entries") or [],
        paper.get("authors") or [],
        paper.get("aminer_author_profiles") or [],
    )
    aminer_url = str(paper.get("aminer_paper_url", "")).strip()
    paper_url = select_paper_url(paper)
    if aminer_url:
        paper_link = f"[查看论文]({aminer_url})"
    elif paper_url:
        paper_link = f"暂无\n[arXiv 查看论文]({paper_url})"
    else:
        paper_link = "暂无"
    elements: list[dict[str, Any]] = []
    elements.extend(
        [
            markdown_block("关键词", keywords),
            markdown_block("年份", year),
        ]
    )
    elements.extend(render_summary_blocks(paper))
    elements.extend(render_famous_author_blocks(paper))
    elements.extend(
        [
            markdown_block("作者列表", authors),
            markdown_block("AMiner 论文链接", paper_link),
        ]
    )
    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": {"title": {"tag": "plain_text", "content": f"{index}. {paper['title']}"}},
        "elements": elements,
    }
