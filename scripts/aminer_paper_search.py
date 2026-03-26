from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from scripts.common import clean_text, dedupe_preserve_order, utc_now_iso
from scripts.constants import build_aminer_author_url, build_aminer_paper_search_url, build_aminer_paper_url


DEFAULT_PAPER_SEARCH_PRO_URL = "https://datacenter.aminer.cn/gateway/open_platform/api/paper/search/pro"
DEFAULT_PAPER_DETAIL_URL = "https://datacenter.aminer.cn/gateway/open_platform/api/paper/detail"
DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_DETAIL_MAX_PAPERS = 40
DEFAULT_DETAIL_MAX_WORKERS = 6
DEFAULT_RETRY_ATTEMPTS = 3
RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_non_negative(value: Any, default: int) -> int:
    return max(_as_int(value, default), 0)


def _coerce_positive(value: Any, default: int) -> int:
    return max(_as_int(value, default), 1)


def _resolve_token(token: str = "", config: dict[str, Any] | None = None) -> str:
    explicit = clean_text(token)
    if explicit:
        return explicit
    config = config or {}
    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    return clean_text(aminer_config.get("token") or os.getenv("AMINER_TOKEN"))


def _format_authorization_header(token: str) -> str:
    cleaned = clean_text(token)
    if not cleaned:
        return ""
    if cleaned.casefold().startswith("bearer "):
        return cleaned
    return f"Bearer {cleaned}"


def resolve_paper_search_pro_url(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    return clean_text(aminer_config.get("paper_search_pro_url") or os.getenv("AMINER_PAPER_SEARCH_PRO_URL")) or DEFAULT_PAPER_SEARCH_PRO_URL


def resolve_paper_detail_url(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    return clean_text(aminer_config.get("paper_detail_url") or os.getenv("AMINER_PAPER_DETAIL_URL")) or DEFAULT_PAPER_DETAIL_URL


def _normalize_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in re.split(r"[,;，；、|]+", value) if part.strip()]
        deduped: list[str] = []
        seen: set[str] = set()
        for item in parts:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = clean_text(
                    item.get("name")
                    or item.get("author_name")
                    or item.get("display_name")
                    or item.get("label")
                    or item.get("text")
                )
                if text:
                    items.append(text)
                continue
            text = clean_text(item)
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
    text = clean_text(value)
    return [text] if text else []


def _extract_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    for key in ("data", "items", "list", "records"):
        value = response.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _extract_items(value)
            if nested:
                return nested
    return []


def _extract_total(response: dict[str, Any], item_count: int) -> int:
    for key in ("total", "count", "total_count", "totalCount"):
        value = response.get(key)
        if value is not None:
            return _coerce_non_negative(value, item_count)
    data = response.get("data")
    if isinstance(data, dict):
        for key in ("total", "count", "total_count", "totalCount"):
            value = data.get(key)
            if value is not None:
                return _coerce_non_negative(value, item_count)
    return item_count


def _extract_authors(item: dict[str, Any]) -> list[str]:
    for key in ("authors", "author", "author_names", "authorNames", "authorsName", "coauthors", "creator"):
        value = item.get(key)
        authors = _normalize_text_list(value)
        if authors:
            return authors
    for key in ("authors_info", "authorInfo", "authorsInfo"):
        value = item.get(key)
        if isinstance(value, list):
            authors = [
                clean_text(author.get("name") or author.get("author_name") or author.get("display_name") or author.get("label"))
                for author in value
                if isinstance(author, dict)
                and clean_text(author.get("name") or author.get("author_name") or author.get("display_name") or author.get("label"))
            ]
            if authors:
                return authors
    return []


def _extract_keywords(item: dict[str, Any], fallback: str = "") -> list[str]:
    for key in ("keywords", "keyword", "tags", "topics", "subject", "subjects"):
        value = item.get(key)
        keywords = _normalize_text_list(value)
        if keywords:
            return keywords
    return _normalize_text_list(fallback)


def _extract_summary(item: dict[str, Any]) -> str:
    for key in ("abstract", "summary", "description", "desc", "intro", "introduce"):
        text = clean_text(item.get(key))
        if text:
            return text
    return ""


def _extract_paper_id(item: dict[str, Any]) -> str:
    for key in ("id", "paper_id", "paperId", "pub_id", "pubId"):
        text = clean_text(item.get(key))
        if text:
            return text
    return ""


def _extract_doi(item: dict[str, Any]) -> str:
    for key in ("doi", "DOI"):
        text = clean_text(item.get(key))
        if text:
            return text
    return ""


def _extract_explicit_url(item: dict[str, Any]) -> str:
    for key in ("aminer_paper_url", "paper_url", "url", "detail_url", "detailUrl", "pub_url"):
        text = clean_text(item.get(key))
        if text:
            return text
    return ""


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


def _load_json_request(
    request: urllib.request.Request,
    *,
    timeout_seconds: int,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
) -> dict[str, Any]:
    opener = _build_https_opener()
    last_error: BaseException | None = None
    max_attempts = max(_coerce_positive(retry_attempts, DEFAULT_RETRY_ATTEMPTS), 1)
    for attempt in range(1, max_attempts + 1):
        try:
            with opener.open(request, timeout=timeout_seconds) as response:  # nosec B310
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = _decode_http_error(exc)
            if exc.code in RETRYABLE_HTTP_STATUS_CODES and attempt < max_attempts:
                last_error = exc
                time.sleep(0.4 * attempt)
                continue
            suffix = f":{detail[:200]}" if detail else ""
            raise RuntimeError(f"aminer_http_{exc.code}{suffix}") from exc
        except (urllib.error.URLError, ssl.SSLError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(0.4 * attempt)
                continue
            break
        except json.JSONDecodeError as exc:
            raise RuntimeError("aminer_invalid_json") from exc
    reason = _extract_error_reason(last_error or RuntimeError("unknown_error")) or "unknown_error"
    raise RuntimeError(f"aminer_unreachable:{reason}") from last_error


def _extract_detail_payload(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data")
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
    items = _extract_items(response)
    return items[0] if items else {}


def _coalesce_text(*values: Any) -> str:
    for value in values:
        text = clean_text(value)
        if text:
            return text
    return ""


def _extract_detail_authors(item: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    authors: list[str] = []
    author_entries: list[dict[str, Any]] = []
    author_profiles: list[dict[str, Any]] = []
    for author in list(item.get("authors") or []):
        if not isinstance(author, dict):
            continue
        author_id = _coalesce_text(author.get("id"), author.get("author_id"))
        display_name = _coalesce_text(author.get("name_zh"), author.get("name"))
        if not display_name:
            continue
        authors.append(display_name)
        profile_url = build_aminer_author_url(author_id)
        author_entries.append(
            {
                "display_name": display_name,
                "profile_url": profile_url,
                "is_disambiguated": bool(author_id),
            }
        )
        author_profiles.append(
            {
                "name": display_name,
                "author_id": author_id,
                "profile_url": profile_url,
                "org": _coalesce_text(author.get("org"), author.get("org_en")),
                "org_zh": _coalesce_text(author.get("org_zh")),
                "is_corresponding": bool(author.get("is_corresponding")),
                "n_citation": _as_int(author.get("n_citation"), 0),
            }
        )
    return authors, author_entries, author_profiles


def _extract_detail_keywords(item: dict[str, Any], fallback: list[str]) -> list[str]:
    keywords = _normalize_text_list(item.get("keywords_zh"))
    if keywords:
        return keywords
    keywords = _normalize_text_list(item.get("keywords"))
    if keywords:
        return keywords
    return list(fallback)


def _merge_paper_detail(existing: dict[str, Any], detail_item: dict[str, Any]) -> dict[str, Any]:
    title = _coalesce_text(existing.get("title"), detail_item.get("title_zh"), detail_item.get("title"))
    abstract = _coalesce_text(detail_item.get("abstract_zh"), detail_item.get("abstract"), existing.get("abstract"), existing.get("summary"))
    authors, author_entries, author_profiles = _extract_detail_authors(detail_item)
    venue = _coalesce_text(
        detail_item.get("raw"),
        (detail_item.get("venue") or {}).get("name_zh") if isinstance(detail_item.get("venue"), dict) else "",
        (detail_item.get("venue") or {}).get("name") if isinstance(detail_item.get("venue"), dict) else "",
        existing.get("venue"),
    )
    merged = {
        **existing,
        "title": title,
        "summary": abstract,
        "abstract": abstract,
        "keywords": _extract_detail_keywords(detail_item, list(existing.get("keywords") or [])),
        "authors": authors or list(existing.get("authors") or []),
        "author_entries": author_entries or list(existing.get("author_entries") or []),
        "aminer_author_profiles": author_profiles or list(existing.get("aminer_author_profiles") or []),
        "doi": _coalesce_text(detail_item.get("doi"), existing.get("doi")),
        "venue": venue,
        "published": _coalesce_text(detail_item.get("year"), existing.get("published")),
        "source_metadata": {
            **(existing.get("source_metadata") or {}),
            "detail_enriched": True,
            "detail_raw_id": _coalesce_text(detail_item.get("id")),
        },
    }
    title_zh = _coalesce_text(detail_item.get("title_zh"))
    abstract_zh = _coalesce_text(detail_item.get("abstract_zh"))
    if title_zh:
        merged["title_zh"] = title_zh
    if abstract_zh:
        merged["abstract_zh"] = abstract_zh
    if detail_item.get("year") is not None:
        merged["year"] = _as_int(detail_item.get("year"), 0)
    return merged


def fetch_paper_detail(
    paper_id: str,
    *,
    token: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_id = clean_text(paper_id)
    if not resolved_id:
        raise ValueError("missing_paper_id")
    config = config or {}
    resolved_token = _resolve_token(token, config)
    url = resolve_paper_detail_url(config)
    request_url = f"{url}?{urllib.parse.urlencode({'id': resolved_id})}"
    headers = {
        "Content-Type": "application/json;charset=utf-8",
        "User-Agent": "aminer-rec/1.0",
        "X-Platform": "openclaw",
    }
    authorization = _format_authorization_header(resolved_token)
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(request_url, headers=headers, method="GET")
    timeout_seconds = _coerce_positive(
        (config.get("aminer") if isinstance(config.get("aminer"), dict) else {}).get("timeout_seconds"),
        DEFAULT_TIMEOUT_SECONDS,
    )
    response = _load_json_request(request, timeout_seconds=timeout_seconds)
    if not isinstance(response, dict):
        raise RuntimeError("paper_detail_invalid_payload")
    if response.get("success") is False:
        raise RuntimeError(clean_text(response.get("msg")) or clean_text(response.get("message")) or "paper_detail_unsuccessful")
    detail_item = _extract_detail_payload(response)
    if not detail_item:
        raise RuntimeError("paper_detail_empty")
    return detail_item


def enrich_papers_with_details(
    papers: list[dict[str, Any]],
    *,
    token: str = "",
    config: dict[str, Any] | None = None,
    max_papers: int = DEFAULT_DETAIL_MAX_PAPERS,
    max_workers: int = DEFAULT_DETAIL_MAX_WORKERS,
) -> list[dict[str, Any]]:
    if not papers:
        return []
    config = config or {}
    resolved_token = _resolve_token(token, config)
    if not resolved_token:
        return list(papers)

    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    detail_limit = min(max(_coerce_positive(aminer_config.get("detail_enrich_max_papers"), max_papers), 1), len(papers))
    worker_count = min(max(_coerce_positive(aminer_config.get("detail_enrich_max_workers"), max_workers), 1), detail_limit)

    enriched: list[dict[str, Any]] = [dict(paper) for paper in papers]
    futures: dict[Any, tuple[int, str]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for index, paper in enumerate(enriched[:detail_limit]):
            paper_id = _coalesce_text(paper.get("aminer_paper_id"), (paper.get("source_metadata") or {}).get("raw_id"))
            if not paper_id:
                continue
            futures[executor.submit(fetch_paper_detail, paper_id, token=resolved_token, config=config)] = (index, paper_id)
        for future in as_completed(futures):
            index, paper_id = futures[future]
            try:
                detail_item = future.result()
            except Exception as exc:
                enriched[index] = {
                    **enriched[index],
                    "source_metadata": {
                        **(enriched[index].get("source_metadata") or {}),
                        "detail_enriched": False,
                        "detail_error": _coalesce_text(exc),
                        "detail_raw_id": paper_id,
                    },
                }
                continue
            enriched[index] = _merge_paper_detail(enriched[index], detail_item)
    return enriched


def normalize_paper_search_pro_item(
    item: dict[str, Any],
    *,
    query_title: str = "",
    query_keyword: str = "",
) -> dict[str, Any]:
    title = clean_text(item.get("title")) or clean_text(query_title) or clean_text(query_keyword)
    summary = _extract_summary(item)
    authors = _extract_authors(item)
    paper_id = _extract_paper_id(item)
    doi = _extract_doi(item)
    explicit_url = _extract_explicit_url(item)
    aminer_paper_url = explicit_url or (build_aminer_paper_url(paper_id) if paper_id else build_aminer_paper_search_url(title))
    keywords = _extract_keywords(item, fallback=query_keyword)
    published = clean_text(item.get("published") or item.get("publish_time") or item.get("publishTime") or item.get("year"))
    year = _coerce_non_negative(item.get("year"), 0)
    venue = clean_text(item.get("venue_name") or item.get("venue"))
    citation_bucket = clean_text(item.get("n_citation_bucket"))
    first_author = clean_text(item.get("first_author"))
    normalized: dict[str, Any] = {
        "title": title,
        "summary": summary,
        "abstract": summary,
        "keywords": keywords,
        "authors": authors,
        "author_entries": [
            {"display_name": author, "profile_url": "", "is_disambiguated": False}
            for author in authors
        ],
        "aminer_author_profiles": [],
        "famous_authors": [],
        "aminer_paper_id": paper_id,
        "aminer_paper_url": aminer_paper_url,
        "doi": doi,
        "venue": venue,
        "n_citation_bucket": citation_bucket,
        "first_author": first_author,
        "source_metadata": {
            "source": "aminer_paper_search_pro",
            "raw_id": paper_id,
            "raw_doi": doi,
        },
    }
    if year:
        normalized["year"] = year
    if published:
        normalized["published"] = published
    if item.get("published_date"):
        normalized["published_date"] = item.get("published_date")
    return normalized


def normalize_paper_search_pro_response(
    response: dict[str, Any],
    *,
    query_title: str = "",
    query_keyword: str = "",
    page: int = 0,
    size: int = 10,
) -> dict[str, Any]:
    papers = [
        normalize_paper_search_pro_item(item, query_title=query_title, query_keyword=query_keyword)
        for item in _extract_items(response)
    ]
    return {
        "status": "success" if response.get("success", True) else "degraded",
        "generated_at": utc_now_iso(),
        "query": {
            "title": clean_text(query_title),
            "keyword": clean_text(query_keyword),
            "page": _coerce_non_negative(page, 0),
            "size": _coerce_positive(size, 10),
        },
        "total": _extract_total(response, len(papers)),
        "paper_count": len(papers),
        "papers": papers,
        "source_metadata": {
            "code": response.get("code"),
            "log_id": clean_text(response.get("log_id")),
            "msg": clean_text(response.get("msg")),
        },
    }


def search_papers_pro(
    *,
    title: str = "",
    keyword: str = "",
    order: str = "",
    page: int = 0,
    size: int = 10,
    token: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_title = clean_text(title)
    query_keyword = clean_text(keyword)
    if not query_title and not query_keyword:
        raise ValueError("missing_title_or_keyword")

    config = config or {}
    url = resolve_paper_search_pro_url(config)
    resolved_token = _resolve_token(token, config)
    params: dict[str, Any] = {
        "page": _coerce_non_negative(page, 0),
        "size": _coerce_positive(size, 10),
    }
    if query_title:
        params["title"] = query_title
    if query_keyword:
        params["keyword"] = query_keyword
    normalized_order = clean_text(order)
    if normalized_order:
        params["order"] = normalized_order
    request_url = f"{url}?{urllib.parse.urlencode(params)}"
    headers = {"Content-Type": "application/json"}
    authorization = _format_authorization_header(resolved_token)
    if authorization:
        headers["Authorization"] = authorization
    request = urllib.request.Request(request_url, headers=headers, method="GET")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    timeout_seconds = _coerce_positive(
        (config.get("aminer") if isinstance(config.get("aminer"), dict) else {}).get("timeout_seconds"),
        DEFAULT_TIMEOUT_SECONDS,
    )
    try:
        with opener.open(request, timeout=timeout_seconds) as response:  # nosec B310
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"paper_search_pro_http_{exc.code}:{detail[:200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"paper_search_pro_unreachable:{exc.reason}") from exc
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("paper_search_pro_invalid_json") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("paper_search_pro_invalid_payload")
    if parsed.get("success") is False:
        raise RuntimeError(clean_text(parsed.get("msg")) or clean_text(parsed.get("message")) or "paper_search_pro_unsuccessful")
    return normalize_paper_search_pro_response(
        parsed,
        query_title=query_title,
        query_keyword=query_keyword,
        page=page,
        size=size,
    )
