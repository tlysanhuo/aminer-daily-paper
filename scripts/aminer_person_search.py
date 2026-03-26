from __future__ import annotations

import json
import os
import ssl
import time
from typing import Any
import urllib.error
import urllib.request
from urllib.request import Request

from scripts.common import clean_text, dedupe_preserve_order, utc_now_iso
from scripts.constants import (
    DEFAULT_AMINER_PERSON_PAPERS_URL,
    DEFAULT_AMINER_PERSON_SEARCH_URL,
    build_aminer_author_url,
    build_aminer_paper_url,
)

DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_RETRY_ATTEMPTS = 3
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_token(token: str = "", config: dict[str, Any] | None = None) -> str:
    explicit = clean_text(token)
    if explicit:
        return explicit
    aminer_config = (config or {}).get("aminer") if isinstance((config or {}).get("aminer"), dict) else {}
    return clean_text(aminer_config.get("token") or os.getenv("AMINER_TOKEN"))


def resolve_person_search_url(config: dict[str, Any] | None = None) -> str:
    aminer_config = (config or {}).get("aminer") if isinstance((config or {}).get("aminer"), dict) else {}
    return clean_text(aminer_config.get("person_search_url")) or DEFAULT_AMINER_PERSON_SEARCH_URL


def resolve_person_papers_url(config: dict[str, Any] | None = None) -> str:
    aminer_config = (config or {}).get("aminer") if isinstance((config or {}).get("aminer"), dict) else {}
    return clean_text(aminer_config.get("person_papers_url")) or DEFAULT_AMINER_PERSON_PAPERS_URL


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
    return clean_text(payload)


def _extract_error_reason(exc: BaseException) -> str:
    if isinstance(exc, urllib.error.URLError):
        return clean_text(exc.reason)
    return clean_text(str(exc))


def _post_json(
    url: str,
    payload: Any,
    *,
    headers: dict[str, str],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
) -> dict[str, Any]:
    request_body = json.dumps(payload).encode("utf-8")
    opener = _build_https_opener()
    last_error: BaseException | None = None
    max_attempts = max(int(retry_attempts or DEFAULT_RETRY_ATTEMPTS), 1)

    for attempt in range(1, max_attempts + 1):
        request = Request(
            url,
            data=request_body,
            headers=headers,
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


def search_persons(
    *,
    name: str,
    org: str = "",
    size: int = 10,
    offset: int = 0,
    token: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_name = clean_text(name)
    if not resolved_name:
        return {"status": "success", "count": 0, "total": 0, "persons": []}
    resolved_token = _resolve_token(token=token, config=config)
    if not resolved_token:
        return {"status": "degraded", "reason": "missing_token", "count": 0, "total": 0, "persons": []}
    payload: dict[str, Any] = {
        "name": resolved_name,
        "offset": max(_as_int(offset, 0), 0),
        "size": min(max(_as_int(size, 10), 1), 10),
    }
    if clean_text(org):
        payload["org"] = clean_text(org)
    response = _post_json(
        resolve_person_search_url(config=config),
        payload,
        headers={
            "Content-Type": "application/json;charset=utf-8",
            "Authorization": resolved_token,
            "User-Agent": "aminer-rec/1.0",
        },
    )
    persons = []
    for item in list(response.get("data") or []):
        if not isinstance(item, dict):
            continue
        person_id = clean_text(item.get("id"))
        english_name = clean_text(item.get("name"))
        chinese_name = clean_text(item.get("name_zh"))
        persons.append(
            {
                "id": person_id,
                "name": english_name,
                "name_zh": chinese_name,
                "display_name": chinese_name or english_name,
                "org": clean_text(item.get("org")),
                "org_zh": clean_text(item.get("org_zh")),
                "org_id": clean_text(item.get("org_id")),
                "interests": [clean_text(entry) for entry in list(item.get("interests") or []) if clean_text(entry)],
                "n_citation": _as_int(item.get("n_citation"), 0),
                "profile_url": build_aminer_author_url(person_id),
            }
        )
    return {
        "status": "success" if bool(response.get("success", True)) else "degraded",
        "count": len(persons),
        "total": _as_int(response.get("total"), len(persons)),
        "persons": persons,
        "raw_response": response,
    }


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [clean_text(item) for item in value if clean_text(item)]
        return dedupe_preserve_order(items)
    text = clean_text(value)
    return [text] if text else []


def _extract_year(paper: dict[str, Any]) -> int:
    versions = paper.get("versions")
    if isinstance(versions, list):
        years = sorted({_as_int(item.get("year"), 0) for item in versions if isinstance(item, dict) and _as_int(item.get("year"), 0)}, reverse=True)
        if years:
            return years[0]
    return _as_int((paper.get("year") or paper.get("pub_year") or paper.get("publication_year")), 0)


def _normalize_person_paper(item: dict[str, Any]) -> dict[str, Any]:
    authors = []
    author_entries = []
    author_profiles = []
    for author in list(item.get("authors") or []):
        if not isinstance(author, dict):
            continue
        name = clean_text(author.get("name"))
        author_id = clean_text(author.get("id"))
        org = clean_text(author.get("org"))
        if not name:
            continue
        authors.append(name)
        author_entries.append(
            {
                "display_name": name,
                "profile_url": build_aminer_author_url(author_id),
                "is_disambiguated": bool(author_id),
            }
        )
        author_profiles.append(
            {
                "name": name,
                "author_id": author_id,
                "profile_url": build_aminer_author_url(author_id),
                "affiliation": org,
            }
        )

    venue = item.get("venue") if isinstance(item.get("venue"), dict) else {}
    venue_info = venue.get("info") if isinstance(venue.get("info"), dict) else {}
    title = clean_text(item.get("title"))
    paper_id = clean_text(item.get("id"))
    abstract = clean_text(item.get("abstract"))
    keywords = _normalize_text_list(item.get("keywords"))
    year = _extract_year(item)

    return {
        "title": title,
        "summary": abstract,
        "abstract": abstract,
        "keywords": keywords,
        "authors": authors,
        "author_entries": author_entries,
        "aminer_author_profiles": author_profiles,
        "famous_authors": [],
        "aminer_paper_id": paper_id,
        "aminer_paper_url": build_aminer_paper_url(paper_id),
        "doi": clean_text(item.get("doi")),
        "venue": clean_text(venue_info.get("name")),
        "year": year,
        "published": str(year) if year else "",
        "citations": _as_int(item.get("num_citation"), 0),
        "source_metadata": {
            "source": "aminer_person_papers",
            "person_paper_fetched_at": utc_now_iso(),
        },
    }


def search_person_papers(
    *,
    person_id: str,
    size: int = 20,
    offset: int = 0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_person_id = clean_text(person_id)
    if not resolved_person_id:
        return {"status": "success", "count": 0, "papers": []}
    payload = [
        {
            "action": "person.SearchPersonPaper",
            "parameters": {
                "person_id": resolved_person_id,
                "size": min(max(_as_int(size, 20), 1), 100),
                "offset": max(_as_int(offset, 0), 0),
            },
        }
    ]
    response = _post_json(
        resolve_person_papers_url(config=config),
        payload,
        headers={
            "Content-Type": "application/json;charset=utf-8",
            "User-Agent": "aminer-rec/1.0",
        },
    )
    data_items = list(response.get("data") or [])
    first = data_items[0] if data_items and isinstance(data_items[0], dict) else {}
    first_data = first.get("data") if isinstance(first.get("data"), dict) else {}
    hit_list = [item for item in list(first_data.get("hitList") or []) if isinstance(item, dict)]
    papers = [_normalize_person_paper(item) for item in hit_list if clean_text(item.get("title"))]
    return {
        "status": "success" if not first.get("error") else "degraded",
        "count": len(papers),
        "papers": papers,
        "raw_response": response,
    }
