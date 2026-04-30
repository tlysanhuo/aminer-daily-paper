#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.arxiv_search import (
    default_recent_top_tier_quota,
    enrich_ranked_payload_with_aminer_paper_urls,
    fetch_arxiv_candidates,
    rank_arxiv_candidates,
    rebalance_recent_top_tier_papers,
)
from scripts.aminer_paper_search import enrich_papers_with_details, search_papers_pro
from scripts.common import read_json, write_json
from scripts.constants import DEFAULT_LLM_RERANK_TOP_N, DEFAULT_TOP_K
from scripts.enrich_with_aminer import enrich_ranked_payload_with_aminer_details
from scripts.llm_client import RerankGenerationError, SummaryGenerationError, llm_rerank_non_cs
from scripts.research_profile import build_research_profile, summarize_profile_request
from scripts.summarize_papers import summarize_papers as summarize_papers_locally


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _profile_match(text: str, phrase: str) -> bool:
    normalized_text = _clean_text(text).casefold()
    normalized_phrase = _clean_text(phrase).casefold()
    if not normalized_text or not normalized_phrase:
        return False
    if re.search(r"[\u4e00-\u9fff]", normalized_phrase):
        return normalized_phrase in normalized_text
    tokens = re.findall(r"[a-z0-9]+", normalized_phrase)
    return bool(tokens) and all(token in normalized_text for token in tokens)


def _build_recommendation_reason(paper: dict[str, Any], profile: dict[str, Any]) -> str:
    reasons: list[str] = []
    profile_topics = [str(item).strip() for item in list(profile.get("topics") or profile.get("retrieval_topics") or []) if str(item).strip()]
    profile_keywords = [str(item).strip() for item in list(profile.get("retrieval_keywords") or profile.get("keywords") or []) if str(item).strip()]
    term_weights = {
        _clean_text(key).casefold(): float(value)
        for key, value in dict(profile.get("retrieval_term_weights") or {}).items()
        if _clean_text(key)
    }
    preferred_authors = [str(item).strip() for item in list(profile.get("preferred_authors") or []) if str(item).strip()]
    preferred_venues = [str(item).strip() for item in list(profile.get("preferred_venues") or []) if str(item).strip()]
    matched_set = {_clean_text(item).casefold() for item in list(paper.get("matched_keywords") or []) if _clean_text(item)}
    matched_topics = [topic for topic in profile_topics if _clean_text(topic).casefold() in matched_set]
    matched_topics.sort(key=lambda item: (term_weights.get(_clean_text(item).casefold(), 0.0), len(_clean_text(item))), reverse=True)
    if matched_topics:
        reasons.append(f"匹配你的研究方向：{' / '.join(matched_topics[:3])}")
    matched_keywords = [keyword for keyword in profile_keywords if _clean_text(keyword).casefold() in matched_set]
    matched_keywords.sort(key=lambda item: (term_weights.get(_clean_text(item).casefold(), 0.0), len(_clean_text(item))), reverse=True)
    if matched_keywords and not matched_topics:
        reasons.append(f"匹配你的关注关键词：{' / '.join(matched_keywords[:3])}")
    paper_authors = [str(item).strip() for item in list(paper.get("authors") or []) if str(item).strip()]
    matched_authors = [author for author in preferred_authors if any(_profile_match(candidate, author) for candidate in paper_authors)]
    if matched_authors:
        reasons.append(f"包含你常关注的作者：{' / '.join(matched_authors[:2])}")
    venue = _clean_text(paper.get("venue"))
    matched_venues = [item for item in preferred_venues if _profile_match(venue, item)]
    if matched_venues:
        reasons.append(f"来自你常关注的 venue：{' / '.join(matched_venues[:2])}")
    if not reasons and profile_topics:
        reasons.append(f"与当前推荐方向 {profile_topics[0]} 语义接近")
    return "；".join(reasons[:2])


def _attach_recommendation_reasons(ranked_payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(ranked_payload)
    for key in ("papers", "ranked_candidates"):
        items = []
        for paper in list(ranked_payload.get(key) or []):
            items.append({**paper, "recommendation_reason": _build_recommendation_reason(paper, profile)})
        if items:
            enriched[key] = items
    return enriched


def _load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _run_python(script_path: Path, args: list[str]) -> None:
    completed = subprocess.run([sys.executable, str(script_path), *args], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"{script_path.name} failed"
        raise RuntimeError(detail)


def _run_python_with_timeout(script_path: Path, args: list[str], *, timeout_seconds: int | None = None) -> None:
    # Unit tests patch `_run_python` and use temporary script directories without real files.
    # Fall back to the original helper in that case so tests can still intercept subprocess calls.
    if not script_path.exists():
        _run_python(script_path, args)
        return
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_value = int(timeout_seconds or 0)
        raise RuntimeError(f"{script_path.name}_timeout:{timeout_value}s") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"{script_path.name} failed"
        raise RuntimeError(detail)


def _summary_subprocess_timeout_seconds(
    *,
    llm_timeout_seconds: int,
    fallback_timeout_seconds: int,
    paper_count: int,
    max_concurrent_requests: int,
    retry_attempts: int = 2,
) -> int:
    per_request = max(int(llm_timeout_seconds or 0), int(fallback_timeout_seconds or 0), 30)
    paper_total = max(int(paper_count or 0), 1)
    concurrency = max(int(max_concurrent_requests or 0), 1)
    retry_factor = max(int(retry_attempts or 0), 0) + 1
    wave_count = max(1, (paper_total + concurrency - 1) // concurrency)
    provider_attempts = 2
    baseline = (per_request * retry_factor * provider_attempts * wave_count) + 20
    return max(90, min(baseline, 300))


def _build_local_summary_fallback_payload(ranked_payload: dict[str, Any]) -> dict[str, Any]:
    fallback_payload = summarize_papers_locally(
        ranked_payload,
        api_key="",
        base_url="",
        model="local-fallback",
        timeout_seconds=1,
        fallback_api_key="",
        fallback_base_url="",
        fallback_model="local-fallback",
        fallback_timeout_seconds=1,
        max_concurrent_requests=1,
        retry_attempts=0,
    )
    papers = list(fallback_payload.get("papers") or [])
    if not papers:
        raise RuntimeError("summary_timeout_no_papers")
    if any(str(paper.get("summary_reason", "")).strip() == "missing_effective_abstract" for paper in papers):
        raise RuntimeError("summary_timeout_missing_effective_abstract")

    normalized_papers: list[dict[str, Any]] = []
    for paper in papers:
        normalized_papers.append(
            {
                **paper,
                "summary_status": "success",
                "summary_reason": "",
                "summary_provider": "local_fallback",
            }
        )
    return {
        **fallback_payload,
        "status": "success",
        "degraded_reason": "",
        "degraded_reasons": [],
        "papers": normalized_papers,
    }


def _paper_identity_key(paper: dict[str, Any]) -> str:
    for field in ("arxiv_id", "aminer_paper_id", "paper_id", "id", "title"):
        key = _clean_text(paper.get(field))
        if key:
            return key.casefold()
    return ""


def _merge_partial_summary_fallback_payload(
    ranked_payload: dict[str, Any],
    partial_payload: dict[str, Any],
) -> dict[str, Any]:
    fallback_payload = _build_local_summary_fallback_payload(ranked_payload)
    fallback_papers = list(fallback_payload.get("papers") or [])
    partial_by_key = {
        _paper_identity_key(paper): paper
        for paper in list(partial_payload.get("papers") or [])
        if _paper_identity_key(paper)
    }

    merged_papers: list[dict[str, Any]] = []
    for index, fallback_paper in enumerate(fallback_papers):
        selected = partial_by_key.get(_paper_identity_key(fallback_paper)) or fallback_paper
        merged_papers.append(selected)

    return {
        **fallback_payload,
        "generated_at": str(partial_payload.get("generated_at") or fallback_payload.get("generated_at") or ""),
        "paper_count": len(merged_papers),
        "profile_topics": list(partial_payload.get("profile_topics") or fallback_payload.get("profile_topics") or []),
        "profile_name": str(partial_payload.get("profile_name") or fallback_payload.get("profile_name") or ""),
        "profile_source": str(partial_payload.get("profile_source") or fallback_payload.get("profile_source") or ""),
        "papers": merged_papers,
    }


def _stage_error(stage: str, detail: Any) -> RuntimeError:
    compact = _clean_text(detail) or "unknown_error"
    return RuntimeError(f"{stage}_failed:{compact}")


def _format_papers_as_markdown(papers: list[dict[str, Any]], profile_topics: list[str]) -> str:
    lines: list[str] = []
    topic_hint = " / ".join(profile_topics[:5]) if profile_topics else ""
    header = f"为你推荐 {len(papers)} 篇相关论文"
    if topic_hint:
        header += f"（研究方向：{topic_hint}）"
    lines.append(header)

    for idx, paper in enumerate(papers, start=1):
        lines.append("")
        lines.append("---")
        lines.append("")
        title = _clean_text(paper.get("title") or "")
        url = _clean_text(paper.get("aminer_paper_url") or paper.get("abs_url") or "")
        title_line = f"**{idx}. [{title}]({url})**" if url else f"**{idx}. {title}**"
        lines.append(title_line)

        year = paper.get("year")
        keywords = paper.get("keywords") or []
        authors = paper.get("authors") or []
        summary = _clean_text(paper.get("summary") or paper.get("abstract") or "")
        meta_parts: list[str] = []
        if year:
            meta_parts.append(f"年份：{year}")
        if keywords:
            meta_parts.append(f"关键词：{' / '.join(str(k) for k in keywords[:5])}")
        if meta_parts:
            lines.append(" | ".join(meta_parts))
        if authors:
            author_str = "、".join(str(a) for a in authors[:6])
            if len(authors) > 6:
                author_str += " et al."
            lines.append(f"作者：{author_str}")

        reason = _clean_text(paper.get("recommendation_reason") or "")
        if reason:
            if len(reason) > 200:
                reason = reason[:200].rstrip() + "…"
            lines.append(f"推荐理由：{reason}")

        if summary:
            truncated = summary if len(summary) <= 300 else summary[:300].rstrip() + "…"
            lines.append("")
            lines.append(truncated)

    return "\n".join(lines)


def _apply_user_filters(ranked_payload: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    language_sort = _clean_text(profile.get("language_sort"))
    start_year = int(profile.get("start_year") or 0)
    end_year = int(profile.get("end_year") or 0)
    if not language_sort and not start_year and not end_year:
        return ranked_payload

    def _paper_year(paper: dict[str, Any]) -> int:
        for field in ("year", "published"):
            val = str(paper.get(field) or "").strip()[:4]
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
        return 0

    def _paper_matches(paper: dict[str, Any]) -> bool:
        if start_year > 0 or end_year > 0:
            py = _paper_year(paper)
            if py and start_year > 0 and py < start_year:
                return False
            if py and end_year > 0 and py > end_year:
                return False
        if language_sort == "en":
            title = str(paper.get("title") or "")
            if re.search(r"[\u4e00-\u9fff]", title):
                return False
        if language_sort == "zh":
            title = str(paper.get("title") or "")
            abstract = str(paper.get("abstract") or paper.get("summary") or "")
            if not re.search(r"[\u4e00-\u9fff]", title + abstract):
                return False
        return True

    filtered: dict[str, Any] = dict(ranked_payload)
    for key in ("papers", "ranked_candidates"):
        items = list(ranked_payload.get(key) or [])
        filtered[key] = [p for p in items if _paper_matches(p)]
    return filtered


def _first_non_success_reason(payload: dict[str, Any], *, status_key: str, reason_key: str, fallback: str) -> str:
    for paper in list(payload.get("papers") or []):
        if str(paper.get(status_key, "success")).strip() == "success":
            continue
        reason = _clean_text(paper.get(reason_key))
        if reason:
            return reason
    return fallback


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = _clean_text(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _extract_profile_llm_topics(profile: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = profile.get("source_metadata") if isinstance(profile.get("source_metadata"), dict) else {}
    collected: list[dict[str, Any]] = []
    direct_topics = metadata.get("llm_topics")
    if isinstance(direct_topics, list):
        collected.extend(item for item in direct_topics if isinstance(item, dict))
    components = metadata.get("components")
    if isinstance(components, list):
        for component in components:
            if not isinstance(component, dict):
                continue
            llm_topics = component.get("llm_topics")
            if isinstance(llm_topics, list):
                collected.extend(item for item in llm_topics if isinstance(item, dict))
    return collected


def _select_aminer_queries(profile: dict[str, Any], *, max_queries: int = 5) -> list[dict[str, str]]:
    topics = _dedupe_preserve_order(list(profile.get("retrieval_topics") or profile.get("topics") or []))
    keywords = _dedupe_preserve_order(list(profile.get("retrieval_keywords") or profile.get("keywords") or []))
    queries: list[dict[str, str]] = []
    llm_topics = _extract_profile_llm_topics(profile)
    if not bool(profile.get("is_cs_user")):
        for item in llm_topics:
            topic_name = _clean_text(item.get("name"))
            topic_keywords = _dedupe_preserve_order(list(item.get("keywords") or []))
            if topic_name:
                queries.append({"title": topic_name, "keyword": ""})
            if topic_name and topic_keywords:
                queries.append({"title": topic_name, "keyword": " ".join(topic_keywords[:2])})
            if topic_keywords:
                queries.append({"title": "", "keyword": " ".join(topic_keywords[: min(len(topic_keywords), 3)])})
    for topic in topics[:max_queries]:
        queries.append({"title": topic, "keyword": ""})
    remaining = max(max_queries - len(queries), 0)
    for keyword in keywords[: max(remaining, 0) + 2]:
        if len(queries) >= max_queries:
            break
        queries.append({"title": "", "keyword": keyword})
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in queries:
        key = (_clean_text(item.get("title")).casefold(), _clean_text(item.get("keyword")).casefold())
        if key in seen or not any(key):
            continue
        seen.add(key)
        deduped.append({"title": _clean_text(item.get("title")), "keyword": _clean_text(item.get("keyword"))})
        if len(deduped) >= max_queries:
            break
    return deduped


def _has_aminer_token(config: dict[str, Any]) -> bool:
    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    return bool(_clean_text(aminer_config.get("token")))


def _citation_bucket_score(bucket: Any) -> float:
    normalized = _clean_text(bucket)
    mapping = {
        "0": 0.0,
        "1-10": 0.8,
        "11-50": 1.4,
        "51-200": 2.0,
        "200-1000": 2.6,
        "1000-5000": 3.0,
        "5000+": 3.4,
    }
    return mapping.get(normalized, 0.0)


def _aminer_candidate_pre_detail_score(paper: dict[str, Any]) -> float:
    score = 0.0
    current_year = datetime.now(timezone.utc).year
    year_text = _clean_text(paper.get("year") or paper.get("published"))
    paper_year = 0
    try:
        paper_year = int(year_text[:4]) if year_text else 0
    except ValueError:
        paper_year = 0
    if paper_year:
        age = max(current_year - paper_year, 0)
        if age <= 1:
            score += 3.2
        elif age <= 3:
            score += 2.5
        elif age <= 5:
            score += 1.5
        elif age <= 8:
            score += 0.6
        else:
            score -= min((age - 8) * 0.18, 1.8)
    score += _citation_bucket_score(paper.get("n_citation_bucket"))
    if _clean_text(paper.get("venue")):
        score += 0.3
    if _clean_text(paper.get("doi")):
        score += 0.2
    title = _clean_text(paper.get("title"))
    if title:
        score += 0.1
    return round(score, 3)


def _prioritize_aminer_candidates_before_detail(papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for paper in papers:
        annotated.append(
            {
                **paper,
                "pre_detail_score": _aminer_candidate_pre_detail_score(paper),
            }
        )
    annotated.sort(
        key=lambda item: (
            float(item.get("pre_detail_score") or 0.0),
            int(item.get("year") or 0),
            _citation_bucket_score(item.get("n_citation_bucket")),
            _clean_text(item.get("title")),
        ),
        reverse=True,
    )
    return annotated


def _resolve_llm_candidates(config: dict[str, Any]) -> list[dict[str, str]]:
    llm_config = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    primary = {
        "api_key": _clean_text(llm_config.get("api_key")),
        "base_url": _clean_text(llm_config.get("base_url")),
        "model": _clean_text(llm_config.get("model")) or "gpt-5-mini",
        "timeout_seconds": str(int(llm_config.get("timeout_seconds") or 30)),
        "label": "primary",
    }
    fallback_config = llm_config.get("fallback") if isinstance(llm_config.get("fallback"), dict) else {}
    fallback = {
        "api_key": _clean_text(fallback_config.get("api_key")),
        "base_url": _clean_text(fallback_config.get("base_url")),
        "model": _clean_text(fallback_config.get("model")) or primary["model"],
        "timeout_seconds": str(int(fallback_config.get("timeout_seconds") or llm_config.get("timeout_seconds") or 30)),
        "label": "fallback",
    }
    return [primary, fallback]


def _paper_identity(paper: dict[str, Any]) -> str:
    return (
        _clean_text(paper.get("arxiv_id"))
        or _clean_text(paper.get("aminer_paper_id"))
        or _clean_text(paper.get("doi"))
        or _clean_text(paper.get("title")).casefold()
    )


def _merge_papers_by_identity(base_papers: list[dict[str, Any]], updated_papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated_by_identity = {
        _paper_identity(paper): paper
        for paper in updated_papers
        if _paper_identity(paper)
    }
    merged: list[dict[str, Any]] = []
    for paper in base_papers:
        identity = _paper_identity(paper)
        merged.append(updated_by_identity.get(identity, paper))
    return merged


def _ensure_recent_top_tier_mix(
    ranked_payload: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(ranked_payload.get("ranked_candidates") or ranked_payload.get("papers") or [])
    if not candidates:
        return ranked_payload
    search_config = config.get("search") if isinstance(config.get("search"), dict) else {}
    top_k = max(int(search_config.get("top_k") or DEFAULT_TOP_K), 1)
    min_recent_top_tier = int(search_config.get("min_recent_top_tier_count") or default_recent_top_tier_quota(top_k))
    recent_top_tier_window = max(int(search_config.get("recent_top_tier_window") or (top_k * 3)), top_k)
    candidate_window = candidates[:recent_top_tier_window]
    aminer_token = _clean_text(((config.get("aminer") if isinstance(config.get("aminer"), dict) else {}) or {}).get("token"))

    if aminer_token:
        enriched_window_payload = enrich_ranked_payload_with_aminer_paper_urls({"papers": candidate_window}, config=config)
        enriched_window_payload = enrich_ranked_payload_with_aminer_details(
            enriched_window_payload,
            token=aminer_token,
        )
        candidate_window = list(enriched_window_payload.get("papers") or candidate_window)

    annotated_window, selected_papers, policy = rebalance_recent_top_tier_papers(
        candidate_window,
        top_k=top_k,
        min_recent_top_tier=min_recent_top_tier,
    )
    merged_ranked_candidates = _merge_papers_by_identity(candidates, annotated_window)
    merged_selected_papers = _merge_papers_by_identity(selected_papers, list(ranked_payload.get("papers") or []))
    merged_selected_papers = _merge_papers_by_identity(merged_selected_papers, annotated_window)
    return {
        **ranked_payload,
        "ranked_candidates": merged_ranked_candidates,
        "papers": merged_selected_papers,
        "recent_top_tier_policy": {
            **policy,
            "window_size": min(recent_top_tier_window, len(candidates)),
        },
    }


def _maybe_llm_rerank_top_candidates(ranked_payload: dict[str, Any], profile: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    candidates = list(ranked_payload.get("ranked_candidates") or ranked_payload.get("papers") or [])
    if len(candidates) < 2:
        return ranked_payload
    rerank_top_n = min(
        int((((config.get("search") if isinstance(config.get("search"), dict) else {}) or {}).get("llm_rerank_top_n")) or DEFAULT_LLM_RERANK_TOP_N),
        len(candidates),
    )
    window = candidates[:rerank_top_n]
    llm_attempts = _resolve_llm_candidates(config)
    last_error = ""
    for attempt in llm_attempts:
        if not attempt["api_key"]:
            continue
        try:
            results, raw_output = llm_rerank_non_cs(
                profile,
                window,
                api_key=attempt["api_key"],
                base_url=attempt["base_url"],
                model=attempt["model"],
                timeout_seconds=int(attempt["timeout_seconds"]),
            )
            by_index = {int(item["index"]): item for item in results if 0 <= int(item["index"]) < len(window)}
            reranked_window: list[dict[str, Any]] = []
            for index, paper in enumerate(window):
                decision = by_index.get(index, {})
                relevance = int(decision.get("relevance", 0))
                quality = int(decision.get("quality", 0))
                llm_score = round(relevance * 0.65 + quality * 0.35, 2)
                combined_score = round(float(paper.get("recommendation_score") or 0.0) * 10.0 + llm_score * 0.8, 2)
                reranked_window.append(
                    {
                        **paper,
                        "llm_rerank_relevance": relevance,
                        "llm_rerank_quality": quality,
                        "llm_rerank_reason": str(decision.get("reason") or "").strip(),
                        "llm_rerank_score": llm_score,
                        "final_recommendation_score": combined_score,
                    }
                )
            reranked_window.sort(
                key=lambda item: (
                    float(item.get("final_recommendation_score") or 0.0),
                    float(item.get("recommendation_score") or 0.0),
                    item.get("title", ""),
                ),
                reverse=True,
            )
            reranked_candidates = reranked_window + candidates[rerank_top_n:]
            top_k = int((((config.get("search") if isinstance(config.get("search"), dict) else {}) or {}).get("top_k")) or DEFAULT_TOP_K)
            rerank_meta = {
                "status": "success",
                "provider": attempt["label"],
                "top_n": rerank_top_n,
                "raw_output": raw_output,
            }
            return {
                **ranked_payload,
                "papers": reranked_candidates[: max(top_k, 1)],
                "ranked_candidates": reranked_candidates,
                "llm_rerank": rerank_meta,
                "llm_rerank_non_cs": rerank_meta,
            }
        except (RerankGenerationError, SummaryGenerationError) as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = f"llm_client_error:{exc.__class__.__name__}"
    if not last_error:
        last_error = "missing_api_key"
    rerank_meta = {
        "status": "degraded",
        "reason": last_error,
        "top_n": rerank_top_n,
    }
    return {
        **ranked_payload,
        "llm_rerank": rerank_meta,
        "llm_rerank_non_cs": rerank_meta,
    }


def _fetch_aminer_candidates(profile: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    if not _has_aminer_token(config):
        return {
            "status": "degraded",
            "generated_at": "",
            "query": [],
            "candidate_count": 0,
            "papers": [],
            "recall_role": "disabled",
            "source": "aminer",
            "error": "missing_token",
        }
    queries = _select_aminer_queries(profile, max_queries=8 if not bool(profile.get("is_cs_user")) else 5)
    if not queries:
        return {
            "status": "degraded",
            "generated_at": "",
            "query": [],
            "candidate_count": 0,
            "papers": [],
            "recall_role": "disabled",
            "source": "aminer",
            "error": "missing_query_terms",
        }
    search_config = config.get("search") if isinstance(config.get("search"), dict) else {}
    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    size = int(aminer_config.get("search_size") or 100)
    collected: list[dict[str, Any]] = []
    query_trace: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    order = "year" if not bool(profile.get("is_cs_user")) else ""
    for query in queries:
        payload = search_papers_pro(
            title=query["title"],
            keyword=query["keyword"],
            order=order,
            page=0,
            size=min(max(size, 5), 100),
            config=config,
        )
        query_trace.append(payload.get("query") or query)
        for paper in payload.get("papers") or []:
            key = _clean_text(paper.get("aminer_paper_id")) or _clean_text(paper.get("doi")) or _clean_text(paper.get("title")).casefold()
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            paper["source_metadata"] = {
                **(paper.get("source_metadata") or {}),
                "recall_source": "aminer",
            }
            collected.append(paper)
    if collected and not bool(profile.get("is_cs_user")):
        collected = _prioritize_aminer_candidates_before_detail(collected)
    if collected and not bool(profile.get("is_cs_user")):
        collected = enrich_papers_with_details(collected, config=config)
    return {
        "status": "success" if collected else "degraded",
        "generated_at": "",
        "query": query_trace,
        "candidate_count": len(collected),
        "papers": collected,
        "recall_role": "primary" if str(profile.get("recall_primary_source")) == "aminer" else "supplemental",
        "source": "aminer",
    }


def _fetch_candidates_by_strategy(profile: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
    ordered_sources = [
        source
        for source in [str(profile.get("recall_primary_source") or ""), str(profile.get("recall_secondary_source") or "")]
        if source in {"arxiv", "aminer"}
    ]
    if not ordered_sources:
        ordered_sources = ["arxiv"]

    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in ordered_sources:
        if (
            source == "arxiv"
            and str(profile.get("recall_primary_source") or "") == "aminer"
            and any(_clean_text(payload.get("source")) == "aminer" and (payload.get("candidate_count") or 0) > 0 for payload in payloads)
        ):
            continue
        try:
            if source == "arxiv":
                payload = fetch_arxiv_candidates(profile, config=config)
                payload["source"] = "arxiv"
            else:
                payload = _fetch_aminer_candidates(profile, config=config)
                if payload.get("status") != "success":
                    reason = _clean_text(payload.get("error") or payload.get("status"))
                    if reason:
                        errors.append(f"aminer:{reason}")
            payloads.append(payload)
        except Exception as exc:
            errors.append(f"{source}:{exc}")

    merged_papers: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for payload in payloads:
        source = _clean_text(payload.get("source"))
        for paper in payload.get("papers") or []:
            identity = (
                _clean_text(paper.get("arxiv_id"))
                or _clean_text(paper.get("aminer_paper_id"))
                or _clean_text(paper.get("doi"))
                or _clean_text(paper.get("title")).casefold()
            )
            if not identity or identity in seen_keys:
                continue
            seen_keys.add(identity)
            paper["source_metadata"] = {
                **(paper.get("source_metadata") or {}),
                "recall_source": _clean_text((paper.get("source_metadata") or {}).get("recall_source")) or source,
            }
            merged_papers.append(paper)

    if not merged_papers:
        detail = "; ".join(errors) if errors else "no_candidates"
        raise RuntimeError(detail)

    return {
        "status": "success",
        "candidate_count": len(merged_papers),
        "papers": merged_papers,
        "recall_plan": ordered_sources,
        "source_payloads": payloads,
        "errors": errors,
    }


def run_pipeline(
    *,
    base_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    aminer_user_id: str,
    topics: list[str],
    scholar_name: str,
    scholar_org: str,
    paper_titles: list[str],
    papers_file: str,
    free_text: str,
    language_sort: str = "",
    start_year: int = 0,
    end_year: int = 0,
    target: str = "",
    account_id: str = "main",
    skip_dispatch: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if target.strip() and account_id.strip():
        write_json(output_dir / "manual_reply_route.json", {"target": target.strip(), "accountId": account_id.strip()})

    write_json(
        output_dir / "request_context.json",
        summarize_profile_request(
            aminer_user_id=aminer_user_id,
            topics=topics,
            scholar_name=scholar_name,
            scholar_org=scholar_org,
            paper_titles=paper_titles,
            papers_file=papers_file,
            free_text=free_text,
        ),
    )
    try:
        profile = build_research_profile(
            aminer_user_id=aminer_user_id,
            topics=topics,
            scholar_name=scholar_name,
            scholar_org=scholar_org,
            paper_titles=paper_titles,
            papers_file=papers_file,
            free_text=free_text,
            config=config,
        )
    except Exception as exc:
        raise _stage_error("profile", exc) from exc

    if profile.get("status") != "success":
        raise _stage_error("profile", profile.get("source_metadata", {}).get("reason") or "profile_unavailable")

    # Inject user-specified language and year preferences into profile
    if language_sort:
        profile["language_sort"] = language_sort
    if start_year > 0:
        profile["start_year"] = start_year
    if end_year > 0:
        profile["end_year"] = end_year

    try:
        candidate_payload = _fetch_candidates_by_strategy(profile, config=config)
    except Exception as exc:
        raise _stage_error("recall", exc) from exc
    try:
        ranked_payload = rank_arxiv_candidates(
            candidate_payload,
            profile,
            top_k=int((config.get("search") or {}).get("top_k") or DEFAULT_TOP_K),
        )
    except Exception as exc:
        raise _stage_error("rank", exc) from exc
    ranked_payload["profile_topics"] = list(profile.get("topics") or profile.get("retrieval_topics") or [])
    ranked_payload["profile_name"] = str(profile.get("profile_name") or "")
    ranked_payload["profile_source"] = str((profile.get("source_metadata") or {}).get("source") or "")
    ranked_payload["is_cs_user"] = bool(profile.get("is_cs_user"))
    ranked_payload["recall_primary_source"] = str(profile.get("recall_primary_source") or "")
    ranked_payload["recall_secondary_source"] = str(profile.get("recall_secondary_source") or "")
    ranked_payload["recall_strategy"] = dict(profile.get("recall_strategy") or {})
    ranked_payload["recall_plan"] = list(candidate_payload.get("recall_plan") or [])
    ranked_payload["recall_errors"] = list(candidate_payload.get("errors") or [])

    # Apply language_sort and year filters from user input
    ranked_payload = _apply_user_filters(ranked_payload, profile)

    try:
        ranked_payload = enrich_ranked_payload_with_aminer_paper_urls(ranked_payload, config=config)
        ranked_payload = enrich_ranked_payload_with_aminer_details(
            ranked_payload,
            token=_clean_text(((config.get("aminer") if isinstance(config.get("aminer"), dict) else {}) or {}).get("token")),
        )
        ranked_payload = _maybe_llm_rerank_top_candidates(ranked_payload, profile, config=config)
        ranked_payload = _ensure_recent_top_tier_mix(ranked_payload, config=config)
        ranked_payload = _attach_recommendation_reasons(ranked_payload, profile)
    except Exception as exc:
        raise _stage_error("rank", exc) from exc
    profile_path = output_dir / "user_profile.json"
    candidates_path = output_dir / "arxiv_candidates.json"
    ranked_path = output_dir / "papers_ranked.json"
    summarized_path = output_dir / "papers_summarized.json"
    messages_path = output_dir / "feishu_messages.json"

    write_json(profile_path, profile)
    write_json(candidates_path, candidate_payload)
    write_json(ranked_path, ranked_payload)

    llm_config = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    llm_fallback = llm_config.get("fallback") if isinstance(llm_config.get("fallback"), dict) else {}
    llm_timeout_seconds = int(llm_config.get("timeout_seconds") or 30)
    fallback_timeout_seconds = int(llm_fallback.get("timeout_seconds") or llm_timeout_seconds or 30)
    summary_retry_attempts = max(int(llm_config.get("retry_attempts") or 0), 0)
    max_concurrent_requests = int(llm_config.get("max_concurrent_requests") or 10)
    summary_timeout_seconds = _summary_subprocess_timeout_seconds(
        llm_timeout_seconds=llm_timeout_seconds,
        fallback_timeout_seconds=fallback_timeout_seconds,
        paper_count=len(list(ranked_payload.get("papers") or [])),
        max_concurrent_requests=max_concurrent_requests,
        retry_attempts=summary_retry_attempts,
    )

    script_dir = base_dir / "scripts"
    summary_args = [
        "--input",
        str(ranked_path),
        "--output",
        str(summarized_path),
        "--api-key",
        _clean_text(llm_config.get("api_key")),
        "--base-url",
        _clean_text(llm_config.get("base_url")),
        "--model",
        _clean_text(llm_config.get("model")) or "gpt-5-mini",
        "--timeout-seconds",
        str(llm_timeout_seconds),
        "--fallback-api-key",
        _clean_text(llm_fallback.get("api_key")),
        "--fallback-base-url",
        _clean_text(llm_fallback.get("base_url")),
        "--fallback-model",
        _clean_text(llm_fallback.get("model")) or _clean_text(llm_config.get("model")) or "gpt-5-mini",
        "--fallback-timeout-seconds",
        str(fallback_timeout_seconds),
        "--max-concurrent-requests",
        str(max_concurrent_requests),
        "--retry-attempts",
        str(summary_retry_attempts),
    ]
    try:
        _run_python_with_timeout(
            script_dir / "summarize_papers.py",
            summary_args,
            timeout_seconds=summary_timeout_seconds,
        )
    except Exception as exc:
        detail = str(exc).strip()
        if "summarize_papers.py_timeout:" in detail:
            try:
                if summarized_path.exists():
                    try:
                        partial_payload = read_json(summarized_path)
                    except Exception:
                        partial_payload = {}
                    if list(partial_payload.get("papers") or []):
                        write_json(
                            summarized_path,
                            _merge_partial_summary_fallback_payload(ranked_payload, partial_payload),
                        )
                    else:
                        write_json(summarized_path, _build_local_summary_fallback_payload(ranked_payload))
                else:
                    write_json(summarized_path, _build_local_summary_fallback_payload(ranked_payload))
            except Exception as fallback_exc:
                raise _stage_error("summary", fallback_exc) from fallback_exc
        else:
            raise _stage_error("summary", exc) from exc
    summarized_payload = read_json(summarized_path)
    summarized_status = str(summarized_payload.get("status", "success")).strip() or "success"
    if summarized_status not in {"success", "partial_success"}:
        raise _stage_error(
            "summary",
            _first_non_success_reason(
                summarized_payload,
                status_key="summary_status",
                reason_key="summary_reason",
                fallback=str(summarized_payload.get("status") or "summary_unavailable"),
            ),
        )
    # Non-Feishu channels: skip card dispatch and return Markdown text instead
    has_feishu_target = bool(target.strip()) and not skip_dispatch

    if not has_feishu_target:
        markdown_text = _format_papers_as_markdown(
            list(summarized_payload.get("papers") or []),
            list(profile.get("topics") or profile.get("retrieval_topics") or []),
        )
        return {
            "status": "success",
            "profile_path": str(profile_path),
            "candidates_path": str(candidates_path),
            "ranked_path": str(ranked_path),
            "summarized_path": str(summarized_path),
            "final_response": "TEXT",
            "reply_text": markdown_text,
            "mode": str(profile.get("profile_mode") or ("scholar_path" if _clean_text(aminer_user_id) else "topic_path")),
            "is_cs_user": bool(profile.get("is_cs_user")),
            "recall_primary_source": str(profile.get("recall_primary_source") or ""),
            "recall_secondary_source": str(profile.get("recall_secondary_source") or ""),
        }

    try:
        _run_python(
            script_dir / "render_feishu_messages.py",
            ["--input", str(summarized_path), "--output", str(messages_path)],
        )
    except Exception as exc:
        raise _stage_error("render", exc) from exc
    if not skip_dispatch:
        dispatch_args = ["--messages", str(messages_path), "--account-id", account_id]
        if target.strip():
            dispatch_args.extend(["--target", target.strip()])
        try:
            _run_python(script_dir / "dispatch_feishu_messages.py", dispatch_args)
        except Exception as exc:
            raise _stage_error("dispatch", exc) from exc

    return {
        "status": "success",
        "profile_path": str(profile_path),
        "candidates_path": str(candidates_path),
        "ranked_path": str(ranked_path),
        "summarized_path": str(summarized_path),
        "messages_path": str(messages_path),
        "final_response": "NO_REPLY" if not skip_dispatch else "READY_FOR_DISPATCH",
        "mode": str(profile.get("profile_mode") or ("scholar_path" if _clean_text(aminer_user_id) else "topic_path")),
        "is_cs_user": bool(profile.get("is_cs_user")),
        "recall_primary_source": str(profile.get("recall_primary_source") or ""),
        "recall_secondary_source": str(profile.get("recall_secondary_source") or ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run aminer-rec end-to-end pipeline.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--aminer-user-id", default="")
    parser.add_argument("--topics", nargs="*", default=[])
    parser.add_argument("--scholar-name", default="")
    parser.add_argument("--scholar-org", default="")
    parser.add_argument("--paper-title", action="append", dest="paper_titles", default=[])
    parser.add_argument("--papers-file", default="")
    parser.add_argument("--free-text", default="")
    parser.add_argument("--language-sort", default="")
    parser.add_argument("--start-year", type=int, default=0)
    parser.add_argument("--end-year", type=int, default=0)
    parser.add_argument("--target", default="")
    parser.add_argument("--account", default="main")
    parser.add_argument("--skip-dispatch", action="store_true")
    args = parser.parse_args()

    resolved_base_dir = args.base_dir.resolve()
    resolved_config = args.config.resolve() if args.config else (resolved_base_dir / "config.example.yaml")

    result = run_pipeline(
        base_dir=resolved_base_dir,
        output_dir=args.output_dir.resolve(),
        config=_load_yaml(resolved_config),
        aminer_user_id=args.aminer_user_id,
        topics=list(args.topics or []),
        scholar_name=args.scholar_name,
        scholar_org=args.scholar_org,
        paper_titles=list(args.paper_titles or []),
        papers_file=args.papers_file,
        free_text=args.free_text,
        language_sort=args.language_sort,
        start_year=args.start_year,
        end_year=args.end_year,
        target=args.target,
        account_id=args.account,
        skip_dispatch=args.skip_dispatch,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
