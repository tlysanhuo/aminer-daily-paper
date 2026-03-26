from __future__ import annotations

import json
import ssl
import time
from typing import Any
import urllib.error
import urllib.request
from urllib.request import Request

from scripts.aminer_schema import build_author_entries, format_famous_author, select_famous_author_profiles
from scripts.common import utc_now_iso
from scripts.constants import DEFAULT_AMINER_AUTHOR_SEARCH_URL, DEFAULT_AMINER_DETAIL_URL, build_aminer_paper_url

DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRY_ATTEMPTS = 3
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _build_https_opener() -> urllib.request.OpenerDirector:
    ssl_context = ssl.create_default_context()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl_context),
    )


def _decode_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        payload = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        payload = ""
    return _clean_text(payload)


def _extract_error_reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.URLError):
        return _clean_text(exc.reason)
    return _clean_text(str(exc))


def _post_json(
    url: str,
    payload: Any,
    token: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    opener = _build_https_opener()
    last_error: BaseException | None = None
    max_attempts = max(int(retry_attempts or DEFAULT_RETRY_ATTEMPTS), 1)

    for attempt in range(1, max_attempts + 1):
        request = Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json;charset=utf-8",
                "Authorization": token,
                "User-Agent": "aminer-rec/1.0",
            },
            method="POST",
        )
        try:
            with opener.open(request, timeout=timeout) as response:  # nosec B310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _decode_http_error(exc)
            if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < max_attempts:
                last_error = exc
                time.sleep(0.4 * attempt)
                continue
            suffix = f":{detail[:200]}" if detail else ""
            raise RuntimeError(f"aminer_request_http_{exc.code}{suffix}") from exc
        except (urllib.error.URLError, ssl.SSLError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(0.4 * attempt)
                continue
            break
        except json.JSONDecodeError as exc:
            raise RuntimeError("aminer_request_invalid_json") from exc

    reason = _extract_error_reason(last_error or RuntimeError("unknown_error")) or "unknown_error"
    raise RuntimeError(f"aminer_request_unreachable:{reason}") from last_error


def _extract_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    for key in ("data", "items"):
        value = response.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_search_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    data = response.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and isinstance(first.get("items"), list):
            return [item for item in first.get("items") if isinstance(item, dict)]
    items = response.get("items")
    if isinstance(items, list):
        return [item for item in items if isinstance(item, dict)]
    return []


def _normalize_author_profile(person: dict[str, Any], query_name: str = "") -> dict[str, Any]:
    indices = person.get("indices") if isinstance(person.get("indices"), dict) else {}
    profile = person.get("profile") if isinstance(person.get("profile"), dict) else {}
    author_id = _clean_text(person.get("id") or person.get("_id") or person.get("person_id"))
    name = _clean_text(person.get("name"))
    honors = profile.get("honor") or []
    if isinstance(honors, str):
        honors = [honors]
    interests = profile.get("interests") or profile.get("topics") or profile.get("keywords") or []
    if isinstance(interests, str):
        interests = [interests]
    return {
        "name": name,
        "name_zh": _clean_text(person.get("name_zh")),
        "author_id": author_id,
        "profile_url": f"https://www.aminer.cn/profile/{author_id}" if author_id else "",
        "affiliation": _clean_text(profile.get("affiliation") or profile.get("organization_zh") or profile.get("org_zh")),
        "position": _clean_text(profile.get("position") or profile.get("position_zh")),
        "hindex": int(indices.get("hindex") or 0),
        "citations": int(indices.get("citations") or 0),
        "interests": [_clean_text(item) for item in interests if _clean_text(item)][:5],
        "honors": [_clean_text(item) for item in honors if _clean_text(item)][:3],
        "is_disambiguated": bool(author_id),
        "query_name": query_name or name,
    }


def _fetch_paper_details(aminer_ids: list[str], token: str, detail_url: str) -> dict[str, dict[str, Any]]:
    if not aminer_ids:
        return {}
    response = _post_json(detail_url, {"ids": aminer_ids}, token)
    items = _extract_items(response)
    details: dict[str, dict[str, Any]] = {}
    for item in items:
        paper_id = _clean_text(item.get("id") or item.get("_id") or item.get("aminer_id"))
        if paper_id:
            details[paper_id] = item
    return details


def _fetch_author_details(author_ids: list[str], token: str, author_search_url: str) -> dict[str, dict[str, Any]]:
    if not author_ids:
        return {}
    response = _post_json(
        author_search_url,
        [
            {
                "action": "search.search",
                "parameters": {"ids": author_ids, "switches": ["master"]},
                "schema": {
                    "person": [
                        "id",
                        "name",
                        "name_zh",
                        {"indices": ["hindex", "pubs", "citations"]},
                        {"profile": ["affiliation", "organization_zh", "org_zh", "position", "position_zh", "interests", "topics", "keywords", "honor"]},
                    ]
                },
            }
        ],
        token,
    )
    items = _extract_search_items(response)
    details: dict[str, dict[str, Any]] = {}
    for item in items:
        person_id = _clean_text(item.get("id") or item.get("_id") or item.get("person_id"))
        if person_id:
            details[person_id] = item
    return details


def enrich_ranked_payload_with_aminer_details(
    ranked_payload: dict[str, Any],
    *,
    token: str,
    detail_url: str = DEFAULT_AMINER_DETAIL_URL,
    author_search_url: str = DEFAULT_AMINER_AUTHOR_SEARCH_URL,
) -> dict[str, Any]:
    papers = list(ranked_payload.get("papers") or [])
    if not _clean_text(token):
        return ranked_payload

    aminer_ids = [_clean_text(paper.get("aminer_paper_id")) for paper in papers if _clean_text(paper.get("aminer_paper_id"))]
    details_by_id = _fetch_paper_details(sorted(set(aminer_ids)), token, detail_url)
    author_ids = sorted(
        {
            _clean_text(author.get("id") or author.get("_id") or author.get("author_id"))
            for detail in details_by_id.values()
            for author in list(detail.get("authors") or [])
            if isinstance(author, dict) and _clean_text(author.get("id") or author.get("_id") or author.get("author_id"))
        }
    )
    author_details_by_id = _fetch_author_details(author_ids, token, author_search_url)

    enriched_papers: list[dict[str, Any]] = []
    for paper in papers:
        aminer_id = _clean_text(paper.get("aminer_paper_id"))
        detail = details_by_id.get(aminer_id, {}) if aminer_id else {}
        detail_authors = [author for author in list(detail.get("authors") or []) if isinstance(author, dict)]
        source_authors = [_clean_text(author.get("name")) for author in detail_authors if _clean_text(author.get("name"))] or list(paper.get("authors") or [])
        profiles = []
        for author in detail_authors:
            author_id = _clean_text(author.get("id") or author.get("_id") or author.get("author_id"))
            if not author_id:
                continue
            profile = author_details_by_id.get(author_id)
            if not profile:
                continue
            profiles.append(_normalize_author_profile(profile, query_name=_clean_text(author.get("name"))))
        author_entries = build_author_entries(source_authors, profiles) if source_authors else list(paper.get("author_entries") or [])
        famous_authors = [format_famous_author(profile) for profile in select_famous_author_profiles(profiles) if format_famous_author(profile)]
        enriched_papers.append(
            {
                **paper,
                "authors": source_authors or list(paper.get("authors") or []),
                "author_entries": author_entries,
                "aminer_author_profiles": profiles,
                "famous_authors": famous_authors or list(paper.get("famous_authors") or []),
                "aminer_paper_url": build_aminer_paper_url(aminer_id) if aminer_id else _clean_text(paper.get("aminer_paper_url")),
                "venue": _clean_text(detail.get("venue") or paper.get("venue")),
                "year": _clean_text(detail.get("year") or paper.get("year")),
                "citations": int(detail.get("n_citation") or paper.get("citations") or 0),
                "aminer_enriched_at": utc_now_iso(),
            }
        )
    return {**ranked_payload, "papers": enriched_papers}
