from __future__ import annotations

import re
from typing import Any

from scripts.constants import AMINER_AUTHOR_URL_TEMPLATE

FAMOUS_AUTHOR_HINDEX_THRESHOLD = 30
FAMOUS_AUTHOR_CITATIONS_THRESHOLD = 5000


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _normalize_name(text: str) -> str:
    return "".join(char for char in str(text or "").casefold() if char.isalnum())


def _names_match(left: str, right: str) -> bool:
    normalized_left = _normalize_name(left)
    normalized_right = _normalize_name(right)
    if not normalized_left or not normalized_right:
        return False
    return normalized_left == normalized_right


def _collect_text_items(*values: Any) -> list[str]:
    items: list[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            parts = [fragment.strip() for fragment in re.split(r"[,;/，；、|]", value) if fragment.strip()]
            items.extend(parts or ([value.strip()] if value.strip() else []))
            continue
        if isinstance(value, list):
            for item in value:
                text = _clean_text(item.get("name") if isinstance(item, dict) else item)
                if text:
                    items.append(text)
            continue
        if isinstance(value, dict):
            text = _clean_text(value.get("name") or value.get("value") or value.get("label"))
            if text:
                items.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _collect_honors(person: dict[str, Any]) -> list[str]:
    honors = _collect_text_items(person.get("honor"), person.get("award"))
    return honors[:3]


def normalize_author_profile(person: dict[str, Any]) -> dict[str, Any]:
    author_id = _clean_text(person.get("_id"))
    name = _clean_text(person.get("name"))
    name_zh = _clean_text(person.get("name_zh"))
    affiliation = _clean_text(person.get("aff")) or _clean_text(person.get("org"))
    position = _clean_text(person.get("pos")) or _clean_text(person.get("title"))
    return {
        "name": name,
        "name_zh": name_zh,
        "author_id": author_id,
        "profile_url": AMINER_AUTHOR_URL_TEMPLATE.format(author_id=author_id) if author_id else "",
        "affiliation": affiliation,
        "affiliation_zh": "",
        "position": position,
        "hindex": _as_int(person.get("h_index") or person.get("hindex")),
        "citations": _as_int(person.get("n_citation") or person.get("citations")),
        "interests": _collect_text_items(person.get("interests"), person.get("tags_zh"), person.get("tags")),
        "honors": _collect_honors(person),
        "honor_raw": person.get("honor") or [],
        "bio": "",
        "is_disambiguated": bool(author_id),
        "query_name": _clean_text(person.get("_query_name")),
    }


def normalize_author_profiles_for_names(
    source_authors: list[str],
    raw_profiles_by_name: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for author_name in source_authors:
        compact = _clean_text(author_name)
        if not compact:
            continue
        candidates = raw_profiles_by_name.get(compact, [])
        if not candidates:
            continue
        best = normalize_author_profile(candidates[0])
        author_id = _clean_text(best.get("author_id"))
        if author_id and author_id in seen_ids:
            continue
        if author_id:
            seen_ids.add(author_id)
        profiles.append(best)
    return profiles


def build_author_entries(source_authors: list[str], profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    remaining_profiles = list(profiles)
    for author_name in source_authors:
        compact = _clean_text(author_name)
        match_index = next(
            (
                index
                for index, profile in enumerate(remaining_profiles)
                if _names_match(compact, str(profile.get("query_name") or profile.get("name") or ""))
                or _names_match(compact, str(profile.get("name") or ""))
            ),
            None,
        )
        if match_index is None:
            entries.append(
                {
                    "display_name": compact,
                    "profile_name": "",
                    "profile_url": "",
                    "affiliation": "",
                    "is_disambiguated": False,
                }
            )
            continue
        profile = remaining_profiles.pop(match_index)
        entries.append(
            {
                "display_name": compact,
                "profile_name": profile.get("name", ""),
                "profile_url": profile.get("profile_url", ""),
                "affiliation": profile.get("affiliation", ""),
                "is_disambiguated": bool(profile.get("profile_url")),
            }
        )
    return entries


def render_author_markdown(
    author_entries: list[dict[str, Any]],
    fallback_authors: list[str],
    aminer_author_profiles: list[dict[str, Any]],
    *,
    max_entries: int = 20,
) -> str:
    rendered: list[str] = []
    seen: set[str] = set()
    if author_entries:
        for entry in author_entries:
            display_name = _clean_text(entry.get("display_name"))
            if not display_name or display_name in seen:
                continue
            seen.add(display_name)
            profile_url = _clean_text(entry.get("profile_url"))
            rendered.append(f"[{display_name}]({profile_url})" if profile_url else display_name)
    if not rendered:
        linked_by_name = {
            _clean_text(profile.get("name")): profile
            for profile in aminer_author_profiles
            if _clean_text(profile.get("name"))
        }
        for author in fallback_authors:
            name = _clean_text(author)
            if not name or name in seen:
                continue
            seen.add(name)
            profile = linked_by_name.get(name, {})
            profile_url = _clean_text(profile.get("profile_url"))
            rendered.append(f"[{name}]({profile_url})" if profile_url else name)
    if not rendered:
        return "暂无"
    limited = rendered[:max_entries]
    result = "、".join(limited)
    return f"{result}、et al." if len(rendered) > max_entries else result


def is_famous_author_profile(profile: dict[str, Any]) -> bool:
    return int(profile.get("hindex", 0) or 0) >= FAMOUS_AUTHOR_HINDEX_THRESHOLD or int(
        profile.get("citations", 0) or 0
    ) >= FAMOUS_AUTHOR_CITATIONS_THRESHOLD


def select_famous_author_profiles(
    profiles: list[dict[str, Any]],
    *,
    max_authors: int = 2,
) -> list[dict[str, Any]]:
    candidates = [profile for profile in profiles if profile.get("name") and is_famous_author_profile(profile)]
    return sorted(
        candidates,
        key=lambda profile: (
            -int(profile.get("hindex", 0) or 0),
            -int(profile.get("citations", 0) or 0),
            str(profile.get("name") or ""),
        ),
    )[:max_authors]


def format_famous_author(profile: dict[str, Any]) -> str:
    name = _clean_text(profile.get("name_zh")) or _clean_text(profile.get("name"))
    parts = [name] if name else []
    affiliation = _clean_text(profile.get("affiliation"))
    if affiliation:
        parts.append(f"来自{affiliation}")
    position = _clean_text(profile.get("position"))
    if position:
        parts.append(position)
    honors = [item for item in (_clean_text(value) for value in list(profile.get("honors") or [])) if item]
    if honors:
        parts.append(f"荣誉包括{'、'.join(honors[:2])}")
    interests = [item for item in (_clean_text(value) for value in list(profile.get("interests") or [])) if item]
    if interests:
        parts.append(f"研究方向包括{'、'.join(interests[:3])}")
    hindex = int(profile.get("hindex", 0) or 0)
    if hindex:
        parts.append(f"h-index 为 {hindex}")
    citations = int(profile.get("citations", 0) or 0)
    if citations:
        parts.append(f"总被引 {citations} 次")
    sentence = "，".join(part for part in parts if part).strip("，")
    return f"{sentence}。" if sentence else ""
