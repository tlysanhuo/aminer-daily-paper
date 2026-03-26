from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from scripts.aminer_person_search import search_person_papers, search_persons
from scripts.aminer_paper_search import search_papers_pro
from scripts.arxiv_search import fetch_arxiv_author_papers
from scripts.build_user_profile import (
    _build_recall_strategy,
    _categories_from_topics,
    _flatten_keyword_items,
    _maybe_apply_llm_topics,
    _normalize_topics,
    _prioritize_keywords,
    _query_from_topics,
    _resolve_llm_candidates,
    build_topics_profile,
    build_user_profile,
)
from scripts.common import clean_text, dedupe_preserve_order
from scripts.constants import (
    DEFAULT_SCHOLAR_PROFILE_MAX_PAPERS,
    DEFAULT_SCHOLAR_PROFILE_RECENT_YEARS,
    DEFAULT_SCHOLAR_PROFILE_SEED_PAPERS,
    DUAL_BUCKET_RECENT_LOOKBACK_DAYS,
    DUAL_BUCKET_RECENT_MAX_PAPERS,
    DUAL_BUCKET_ANCHOR_MAX_PAPERS,
    SOURCE_PRIOR_WEIGHTS,
    DUAL_BUCKET_MATCH_BONUS,
)
from scripts.datacenter_client import call_segmentation_pro
from scripts.enrich_with_aminer import enrich_ranked_payload_with_aminer_details
from scripts.llm_client import ScholarTermLabelError, llm_label_scholar_terms
from scripts.internal_profile_provider import (
    ENGLISH_FUNCTION_WORDS,
    _is_generic_direction_term,
    _is_generic_keyword_term,
    _is_low_signal_phrase,
    _normalize_term,
    _score_terms,
    _score_terms_for_bucket,
    _specificity_bonus,
)

LLM_SCHOLAR_ROLE_FACTORS = {
    "scholar_specific": 1.25,
    "core_domain": 1.0,
    "broad_superordinate": 0.35,
    "method": 0.18,
    "auxiliary": 0.45,
    "noise": 0.0,
}


def _should_skip_retrieval_term(term: str) -> bool:
    normalized = _normalize_term(term)
    if not normalized:
        return True
    if _is_low_signal_phrase(normalized) or _is_generic_direction_term(normalized) or _is_generic_keyword_term(normalized):
        return True
    lowered = normalized.casefold()
    if any(fragment in lowered for fragment in ("extensive experiments", "we propose", "we present", "show that", "our approach", "our method")):
        return True
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", lowered)
    if len(tokens) >= 3 and any(token in {"and", "for", "with", "via", "that", "our", "their", "this"} for token in tokens):
        return True
    if tokens[:2] in (
        ["we", "propose"],
        ["we", "present"],
        ["we", "study"],
        ["show", "that"],
        ["our", "approach"],
        ["our", "method"],
        ["conduct", "extensive"],
    ):
        return True
    if tokens and tokens[0] in {"conduct", "demonstrate", "evaluate", "experimental"}:
        return True
    return False


def _looks_like_method_topic(term: str) -> bool:
    normalized = _normalize_term(term).casefold()
    if not normalized:
        return True
    if normalized in {"benchmark", "benchmarks", "evaluation", "evaluations"}:
        return True
    if any(fragment in normalized for fragment in ("contrastive learning", "self-supervised learning", "few-shot learning", "transfer learning", "pre-training", "pretraining", "fine-tuning", "finetuning")):
        return True
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalized)
    if not tokens:
        return False
    if "benchmark" in tokens or "benchmarks" in tokens:
        return True
    if "learning" in tokens and not any(
        token in {"recognition", "linking", "disambiguation", "retrieval", "extraction", "recommendation", "grounding"}
        for token in tokens
    ):
        return True
    return False


def _looks_like_auxiliary_scholar_term(term: str) -> bool:
    normalized = _normalize_term(term).casefold()
    if not normalized:
        return False
    auxiliary_exact = {
        "heterogeneous networks",
        "information diffusion",
        "social influence",
        "social networks",
        "social networking (online)",
        "task relationships",
    }
    if normalized in auxiliary_exact:
        return True
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalized)
    if not tokens:
        return False
    auxiliary_tokens = {"diffusion", "influence", "network", "networks", "relationship", "relationships", "social"}
    return any(token in auxiliary_tokens for token in tokens)


def _looks_like_bucket_noise_topic(term: str) -> bool:
    normalized = _normalize_term(term)
    lowered = normalized.casefold()
    if not normalized:
        return True
    if re.fullmatch(r"[a-z]{2}\.[A-Z]{2}", normalized):
        return True
    if any(fragment in lowered for fragment in ("http", "https", "github", "code and data")):
        return True
    if _is_low_signal_phrase(normalized) or _should_skip_retrieval_term(normalized):
        return True
    if _looks_like_method_topic(normalized):
        return True
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", lowered)
    if not tokens:
        return True
    sentence_heads = {
        ("we", "propose"),
        ("we", "present"),
        ("we", "study"),
        ("show", "that"),
        ("have", "been"),
        ("effective", "and"),
        ("state-of", "the-art"),
        ("extensive", "experiments"),
    }
    if len(tokens) >= 2 and tuple(tokens[:2]) in sentence_heads:
        return True
    function_ratio = sum(1 for token in tokens if token in ENGLISH_FUNCTION_WORDS) / max(len(tokens), 1)
    if len(tokens) >= 4 and function_ratio >= 0.34:
        return True
    return False


def _clean_dual_bucket_topics(terms: list[str], *, limit: int = 8) -> list[str]:
    cleaned: list[str] = []
    for term in dedupe_preserve_order(list(terms or [])):
        normalized = _normalize_term(term)
        if _looks_like_bucket_noise_topic(normalized):
            continue
        cleaned.append(normalized)
        if len(cleaned) >= max(int(limit), 1):
            break
    return cleaned


def _build_scholar_research_domains(
    *,
    recall_topics: list[str],
    rerank_topics: list[str],
    rerank_keywords: list[str],
    categories: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    cleaned_categories = [clean_text(item) for item in categories if clean_text(item)]
    strong_terms = dedupe_preserve_order(
        [
            *[term for term in recall_topics if not _looks_like_auxiliary_scholar_term(term)],
            *[term for term in rerank_topics if not _looks_like_auxiliary_scholar_term(term)],
            *[term for term in rerank_keywords if not _looks_like_auxiliary_scholar_term(term)],
        ]
    )
    academic_graph_terms: list[str] = []
    entity_task_terms: list[str] = []
    method_terms: list[str] = []
    fallback_terms: list[str] = []
    for term in strong_terms:
        normalized = _normalize_term(term)
        lowered = normalized.casefold()
        if not normalized or _should_skip_retrieval_term(normalized):
            continue
        if _looks_like_method_topic(normalized):
            method_terms.append(normalized)
            continue
        if any(
            token in lowered
            for token in (
                "academic graph",
                "academic knowledge graph",
                "knowledge graph",
                "oag",
                "open academic graph",
                "author",
                "ambiguity",
                "disambiguation",
                "scholar",
                "paper comprehension",
            )
        ):
            academic_graph_terms.append(normalized)
            continue
        if any(
            token in lowered
            for token in (
                "entity linking",
                "concept linking",
                "named entity recognition",
                "recognition",
                "linking",
                "retrieval",
                "mining",
            )
        ):
            entity_task_terms.append(normalized)
            continue
        fallback_terms.append(normalized)

    domains: list[dict[str, Any]] = []
    if academic_graph_terms:
        domains.append(
            {
                "name": academic_graph_terms[0],
                "keywords": dedupe_preserve_order([*academic_graph_terms[:5], *entity_task_terms[:2]])[:6],
                "arxiv_categories": cleaned_categories,
                "priority": 5,
            }
        )
    if entity_task_terms:
        domains.append(
            {
                "name": entity_task_terms[0],
                "keywords": dedupe_preserve_order([*entity_task_terms[:5], *academic_graph_terms[:2]])[:6],
                "arxiv_categories": cleaned_categories,
                "priority": 4,
            }
        )
    if method_terms:
        domains.append(
            {
                "name": method_terms[0],
                "keywords": dedupe_preserve_order(method_terms[:4])[:4],
                "arxiv_categories": cleaned_categories,
                "priority": 2,
            }
        )
    if not domains and fallback_terms:
        domains.append(
            {
                "name": fallback_terms[0],
                "keywords": dedupe_preserve_order(fallback_terms[:6]),
                "arxiv_categories": cleaned_categories,
                "priority": 3,
            }
        )

    excluded_keywords = dedupe_preserve_order(
        [
            *[term for term in rerank_keywords if _looks_like_auxiliary_scholar_term(term)],
            "biomedical",
            "clinical",
            "financial",
            "patent",
            "payment",
            "drosophila",
            "crime",
            "humanitarian",
            "geolocation",
        ]
    )[:16]
    return domains[:3], excluded_keywords


def _canonicalize_scholar_entity_term(term: Any) -> str:
    normalized = _normalize_term(term)
    if not normalized:
        return ""
    normalized = re.sub(r"^[A-Za-z]+[-:]\s*", "", normalized)
    normalized = re.sub(r"\bterms?[-:]\s*", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bopen academic graph\b", "OAG", normalized, flags=re.IGNORECASE)
    return _normalize_term(normalized)


def _looks_like_scholar_entity_term(term: str) -> bool:
    normalized = _canonicalize_scholar_entity_term(term)
    if not normalized or _should_skip_retrieval_term(normalized) or _looks_like_method_topic(normalized):
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9-]{1,9}", normalized):
        return True
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", normalized.casefold())
    if not tokens:
        return False
    signal_tokens = {
        "academic",
        "author",
        "authors",
        "disambiguation",
        "ambiguity",
        "entity",
        "entities",
        "graph",
        "graphs",
        "knowledge",
        "linking",
        "ontology",
        "ontologies",
        "paper",
        "papers",
        "scholar",
        "retrieval",
        "comprehension",
    }
    if "systems" in tokens and "knowledge" in tokens:
        return False
    return any(token in signal_tokens for token in tokens)


def _term_token_set(term: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", _normalize_term(term).casefold()))
    normalized: set[str] = set()
    for token in tokens:
        normalized.add(token[:-1] if token.endswith("s") else token)
    return {token for token in normalized if token}


def _select_specific_terms(terms: list[str], *, limit: int) -> list[str]:
    selected: list[str] = []
    selected_tokens: list[set[str]] = []
    for term in terms:
        normalized = _normalize_term(term)
        tokens = _term_token_set(normalized)
        if not normalized or not tokens:
            continue
        redundant = False
        for existing_tokens in selected_tokens:
            if tokens.issubset(existing_tokens):
                redundant = True
                break
        if redundant:
            continue
        selected.append(normalized)
        selected_tokens.append(tokens)
        if len(selected) >= max(int(limit), 1):
            break
    return selected


def _extract_scholar_entity_terms(seed_papers: list[dict[str, Any]], core_topics: list[dict[str, Any]]) -> list[str]:
    candidates: list[str] = []
    for item in core_topics:
        if not isinstance(item, dict):
            continue
        candidates.append(_canonicalize_scholar_entity_term(item.get("name")))
        for keyword in list(item.get("keywords") or [])[:4]:
            candidates.append(_canonicalize_scholar_entity_term(keyword))
    for paper in seed_papers:
        if not isinstance(paper, dict):
            continue
        for keyword in list(paper.get("keywords") or []):
            candidates.append(_canonicalize_scholar_entity_term(keyword))
        title = _normalize_term(paper.get("title"))
        if ":" in title:
            prefix = _canonicalize_scholar_entity_term(title.split(":", 1)[0])
            candidates.append(prefix)
    terms = []
    for term in dedupe_preserve_order(candidates):
        if _looks_like_scholar_entity_term(term):
            terms.append(term)
    return terms[:6]


def _add_term_score(
    scores: dict[str, float],
    labels: dict[str, str],
    term: Any,
    *,
    weight: float,
) -> None:
    normalized = _normalize_term(term)
    if not normalized:
        return
    if _should_skip_retrieval_term(normalized):
        return
    key = normalized.casefold()
    labels.setdefault(key, normalized)
    scores[key] = max(scores.get(key, 0.0), round(float(weight) + _specificity_bonus(normalized), 3))


def _build_retrieval_signal(profile: dict[str, Any], *, max_topics: int = 6, max_keywords: int = 12) -> tuple[list[str], list[str], dict[str, float]]:
    scores: dict[str, float] = {}
    labels: dict[str, str] = {}

    for topic in list(profile.get("topics") or []):
        _add_term_score(scores, labels, topic, weight=2.6)
    for keyword in list(profile.get("keywords") or []):
        _add_term_score(scores, labels, keyword, weight=1.6)

    metadata = profile.get("source_metadata") if isinstance(profile.get("source_metadata"), dict) else {}
    components = metadata.get("components")
    if not isinstance(components, list):
        components = [metadata]

    topic_keys: set[str] = set()
    keyword_keys: set[str] = set()
    for component in components:
        if not isinstance(component, dict):
            continue
        for item in component.get("resolved_person_interests") or []:
            normalized = _normalize_term(item)
            if not normalized:
                continue
            _add_term_score(scores, labels, normalized, weight=3.8)
            topic_keys.add(normalized.casefold())
        for topic in component.get("core_topics") or []:
            if not isinstance(topic, dict):
                continue
            name = _normalize_term(topic.get("name"))
            support_count = int(topic.get("support_count") or 0)
            support_bonus = min(max(support_count, 0) * 0.25, 0.75)
            topic_bonus = min(float(topic.get("score") or 0.0) / 12.0, 1.25)
            if name:
                _add_term_score(scores, labels, name, weight=2.8 + support_bonus + topic_bonus)
                topic_keys.add(name.casefold())
            for keyword in list(topic.get("keywords") or [])[:4]:
                normalized = _normalize_term(keyword)
                if not normalized:
                    continue
                _add_term_score(scores, labels, normalized, weight=1.9 + support_bonus + min(topic_bonus, 0.8))
                keyword_keys.add(normalized.casefold())
        for item in component.get("llm_topics") or []:
            if not isinstance(item, dict):
                continue
            name = _normalize_term(item.get("name"))
            if name:
                _add_term_score(scores, labels, name, weight=3.1)
                topic_keys.add(name.casefold())
            for keyword in list(item.get("keywords") or [])[:4]:
                normalized = _normalize_term(keyword)
                if not normalized:
                    continue
                _add_term_score(scores, labels, normalized, weight=2.0)
                keyword_keys.add(normalized.casefold())

    ranked_keys = sorted(
        scores,
        key=lambda key: (
            scores[key],
            1 if key in topic_keys else 0,
            len(labels.get(key, "")),
            labels.get(key, "").casefold(),
        ),
        reverse=True,
    )
    retrieval_topics = [labels[key] for key in ranked_keys if key in topic_keys][: max(int(max_topics), 1)]
    retrieval_keywords = [labels[key] for key in ranked_keys if key not in topic_keys or key in keyword_keys][: max(int(max_keywords), 1)]

    if not retrieval_topics:
        retrieval_topics = [labels[key] for key in ranked_keys[: max(int(max_topics), 1)]]
    if not retrieval_keywords:
        retrieval_keywords = [labels[key] for key in ranked_keys[: max(int(max_keywords), 1)]]

    weight_map = {labels[key]: scores[key] for key in ranked_keys[: max(max_topics, max_keywords, 12)]}
    return retrieval_topics, retrieval_keywords, weight_map


def _build_scholar_signal_layers(
    *,
    explicit_topics: list[str],
    resolved_interests: list[str],
    core_topics: list[dict[str, Any]],
    seed_papers: list[dict[str, Any]],
    merged_topics: list[str],
    merged_keywords: list[str],
) -> tuple[list[str], list[str], list[str], dict[str, float]]:
    candidate_recall_topics = dedupe_preserve_order(
        [
            *_normalize_topics(explicit_topics),
            *[topic for topic in resolved_interests if not _should_skip_retrieval_term(topic)],
        ]
    )
    domain_recall_topics = _select_specific_terms(
        [topic for topic in candidate_recall_topics if not _looks_like_method_topic(topic)],
        limit=3,
    )
    domain_token_sets = [_term_token_set(topic) for topic in domain_recall_topics]
    entity_terms = [
        term
        for term in _select_specific_terms(_extract_scholar_entity_terms(seed_papers, core_topics), limit=6)
        if not any(_term_token_set(term).issubset(tokens) for tokens in domain_token_sets if tokens)
    ][:4]
    prioritized_domains = domain_recall_topics[:3]
    recall_topics = dedupe_preserve_order([*prioritized_domains, *entity_terms])[:6]
    rerank_topics = dedupe_preserve_order(
        [
            *_normalize_topics(explicit_topics),
            *[topic for topic in resolved_interests if not _is_low_signal_phrase(topic)],
            *entity_terms,
            *[
                _normalize_term(item.get("name"))
                for item in core_topics
                if isinstance(item, dict) and _normalize_term(item.get("name"))
            ],
            *merged_topics,
        ]
    )[:10]
    rerank_keywords = _prioritize_keywords(
        [
            *rerank_topics,
            *entity_terms,
            *[
                keyword
                for item in core_topics
                if isinstance(item, dict)
                for keyword in list(item.get("keywords") or [])[:4]
            ],
            *merged_keywords,
        ]
    )[:18]
    primary_rerank_keywords = [keyword for keyword in rerank_keywords if not _looks_like_auxiliary_scholar_term(keyword)]
    auxiliary_rerank_keywords = [keyword for keyword in rerank_keywords if _looks_like_auxiliary_scholar_term(keyword)]
    rerank_keywords = [*primary_rerank_keywords, *auxiliary_rerank_keywords][:18]
    retrieval_keywords = _prioritize_keywords(recall_topics)[:12]
    weight_map: dict[str, float] = {}
    for topic in recall_topics:
        weight_map[topic] = 5.0 + _specificity_bonus(topic)
    for term in entity_terms:
        weight_map[term] = max(weight_map.get(term, 0.0), 5.4 + _specificity_bonus(term))
    for topic in rerank_topics:
        if _looks_like_auxiliary_scholar_term(topic):
            weight_map.setdefault(topic, 0.9 + (_specificity_bonus(topic) * 0.25))
        else:
            weight_map.setdefault(topic, 3.2 + _specificity_bonus(topic))
    for keyword in rerank_keywords:
        if _looks_like_auxiliary_scholar_term(keyword):
            weight_map.setdefault(keyword, 0.6 + (_specificity_bonus(keyword) * 0.2))
        else:
            weight_map.setdefault(keyword, 1.6 + _specificity_bonus(keyword))
    return recall_topics, rerank_topics, rerank_keywords, weight_map


def _is_short_acronym(term: str) -> bool:
    """判断是否为短 acronym（1-3 个纯大写字母），这类词不应单独作为查询词"""
    normalized = _normalize_term(term)
    if not normalized:
        return False
    # 匹配 1-3 个纯大写字母的 acronym，如 NER, KG, IR, LLM, RAG
    return bool(re.fullmatch(r"[A-Z]{1,3}", normalized))


def _build_layered_recall_terms(
    labels: list[dict[str, Any]],
    *,
    recall_topics: list[str],
    rerank_topics: list[str],
    rerank_keywords: list[str],
    weight_map: dict[str, float],
) -> dict[str, Any]:
    """基于 LLM 标注构建分层召回术语

    返回:
        - primary_recall_terms: scholar_specific + core_domain（高特异性）
        - fallback_recall_terms: broad_superordinate（仅补召回）
        - blocked_query_terms: auxiliary + noise + method（不进入召回）
    """
    primary_roles = {"scholar_specific", "core_domain"}
    fallback_roles = {"broad_superordinate"}
    blocked_roles = {"auxiliary", "noise", "method"}

    label_map: dict[str, dict[str, Any]] = {}
    for item in labels:
        term = _normalize_term(item.get("term"))
        if not term:
            continue
        label_map[term.casefold()] = {**item, "term": term}

    all_terms = dedupe_preserve_order([*recall_topics, *rerank_topics, *rerank_keywords])

    primary_terms: list[str] = []
    fallback_terms: list[str] = []
    blocked_terms: list[str] = []
    unclassified_terms: list[str] = []

    for term in all_terms:
        normalized = _normalize_term(term)
        if not normalized:
            continue

        label = label_map.get(normalized.casefold())
        if not label:
            unclassified_terms.append(normalized)
            continue

        role = str(label.get("role") or "")

        # 短 acronym 强制进入 blocked
        if _is_short_acronym(normalized):
            blocked_terms.append(normalized)
            continue

        if role in primary_roles:
            primary_terms.append(normalized)
        elif role in fallback_roles:
            fallback_terms.append(normalized)
        elif role in blocked_roles:
            blocked_terms.append(normalized)
        else:
            unclassified_terms.append(normalized)

    def _primary_rank(item: str) -> tuple[float, float, float, float]:
        normalized = _normalize_term(item)
        label = label_map.get(normalized.casefold(), {})
        role = str(label.get("role") or "")
        role_priority = 2.0 if role == "scholar_specific" else 1.0 if role == "core_domain" else 0.0
        return (
            role_priority,
            float(label.get("weight") or 0.0),
            float(weight_map.get(normalized, 0.0)),
            float(len(normalized)),
        )

    def _fallback_rank(item: str) -> tuple[float, float, float]:
        normalized = _normalize_term(item)
        label = label_map.get(normalized.casefold(), {})
        return (
            float(label.get("weight") or 0.0),
            float(weight_map.get(normalized, 0.0)),
            float(len(normalized)),
        )

    # 按 LLM 标签优先级 + LLM 权重 + 原始权重排序
    def _sort_by_weight(items: list[str], *, mode: str) -> list[str]:
        return sorted(
            items,
            key=_primary_rank if mode == "primary" else _fallback_rank,
            reverse=True,
        )

    primary_terms = _sort_by_weight(primary_terms, mode="primary")
    fallback_terms = _sort_by_weight(fallback_terms, mode="fallback")
    blocked_terms = _sort_by_weight(blocked_terms, mode="fallback")[:10]

    # 未分类术语不参与召回，只保留在 blocked 里避免 query 被噪声带偏
    blocked_terms = dedupe_preserve_order([*blocked_terms, *_sort_by_weight(unclassified_terms, mode="fallback")[:8]])[:12]

    return {
        "primary_recall_terms": primary_terms[:5],
        "fallback_recall_terms": fallback_terms[:4],
        "blocked_query_terms": blocked_terms,
        "primary_count": len(primary_terms),
        "fallback_count": len(fallback_terms),
        "blocked_count": len(blocked_terms),
    }


def _maybe_apply_llm_scholar_term_labels(
    *,
    scholar_name: str,
    scholar_org: str,
    resolved_interests: list[str],
    core_topics: list[dict[str, Any]],
    seed_papers: list[dict[str, Any]],
    recall_topics: list[str],
    rerank_topics: list[str],
    rerank_keywords: list[str],
    weight_map: dict[str, float],
    config: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], list[str], dict[str, float], dict[str, Any]]:
    llm_config = dict((config or {}).get("llm") or {})
    if not bool(llm_config.get("enable_scholar_term_labeling")):
        return recall_topics, rerank_topics, rerank_keywords, weight_map, {"reason": "disabled"}

    llm_candidates = _resolve_llm_candidates(config)
    if not any(candidate.get("api_key") for candidate in llm_candidates):
        return recall_topics, rerank_topics, rerank_keywords, weight_map, {"reason": "missing_api_key"}

    candidate_terms = dedupe_preserve_order([*recall_topics, *rerank_topics, *rerank_keywords])[:18]
    if len(candidate_terms) < 4:
        return recall_topics, rerank_topics, rerank_keywords, weight_map, {"reason": "insufficient_terms"}

    payload = {
        "scholar_name": scholar_name,
        "scholar_org": scholar_org,
        "resolved_interests": resolved_interests[:8],
        "core_topics": core_topics[:8],
        "seed_papers": seed_papers[:5],
        "candidate_terms": candidate_terms,
    }
    labels: list[dict[str, Any]] = []
    raw_output = ""
    failure_reasons: list[str] = []
    for index, llm_config in enumerate(llm_candidates):
        if not llm_config.get("api_key"):
            continue
        label = "primary" if index == 0 else f"fallback_{index}"
        try:
            labels, raw_output = llm_label_scholar_terms(
                payload,
                api_key=str(llm_config["api_key"]),
                base_url=str(llm_config.get("base_url") or ""),
                model=str(llm_config.get("model") or ""),
                timeout_seconds=int(llm_config.get("timeout_seconds") or 30),
            )
            break
        except Exception as exc:
            failure_reasons.append(f"{label}:{exc}")
    if not labels:
        return recall_topics, rerank_topics, rerank_keywords, weight_map, {
            "reason": "; ".join(failure_reasons) if failure_reasons else "llm_label_unavailable"
        }

    label_map: dict[str, dict[str, Any]] = {}
    for item in labels:
        term = _normalize_term(item.get("term"))
        if not term:
            continue
        label_map[term.casefold()] = {**item, "term": term}

    adjusted_weights: dict[str, float] = {}
    for term, weight in weight_map.items():
        normalized = _normalize_term(term)
        if not normalized:
            continue
        label = label_map.get(normalized.casefold())
        current_weight = float(weight or 0.0)
        if not label:
            adjusted_weights[normalized] = current_weight
            continue
        role = str(label.get("role") or "")
        base_factor = LLM_SCHOLAR_ROLE_FACTORS.get(role, 1.0)
        llm_weight = float(label.get("weight") or 0.0)
        factor = max(0.0, min(base_factor * (llm_weight if llm_weight > 0 else 1.0), 1.5))
        adjusted_weights[normalized] = round(current_weight * factor, 3)

    def _sort_terms(items: list[str]) -> list[str]:
        return sorted(
            dedupe_preserve_order(items),
            key=lambda item: (
                adjusted_weights.get(_normalize_term(item), 0.0),
                len(_normalize_term(item)),
                _normalize_term(item).casefold(),
            ),
            reverse=True,
        )

    filtered_recall_topics = [
        term
        for term in _sort_terms(recall_topics)
        if adjusted_weights.get(_normalize_term(term), 0.0) > 0
    ][:6]
    filtered_rerank_topics = [
        term
        for term in _sort_terms(rerank_topics)
        if adjusted_weights.get(_normalize_term(term), 0.0) > 0
    ][:10]
    filtered_rerank_keywords = [
        term
        for term in _sort_terms(rerank_keywords)
        if adjusted_weights.get(_normalize_term(term), 0.0) > 0
    ][:18]

    # 构建分层召回术语
    layered_terms = _build_layered_recall_terms(
        labels,
        recall_topics=filtered_recall_topics or recall_topics,
        rerank_topics=filtered_rerank_topics or rerank_topics,
        rerank_keywords=filtered_rerank_keywords or rerank_keywords,
        weight_map=adjusted_weights,
    )

    metadata = {
        "reason": "success",
        "labels": labels,
        "raw_output": raw_output,
        "layered_recall_terms": layered_terms,
    }
    return filtered_recall_topics or recall_topics, filtered_rerank_topics or rerank_topics, filtered_rerank_keywords or rerank_keywords, adjusted_weights, metadata


def _is_research_exclude_term(term: str) -> bool:
    normalized = _normalize_term(term)
    if not normalized:
        return True
    if _should_skip_retrieval_term(normalized):
        return True
    if _looks_like_method_topic(normalized):
        return True
    if _looks_like_auxiliary_scholar_term(normalized):
        return True
    return False


def _normalize_research_term(term: Any) -> str:
    normalized = _normalize_term(term)
    if not normalized:
        return ""
    normalized = re.sub(r"^(?:terms?|keywords?)\s*[-:]\s*", "", normalized, flags=re.IGNORECASE)
    if re.fullmatch(r"[A-Z0-9]{2,10}-", normalized):
        return normalized[:-1]
    if normalized == "OAG":
        return "Open Academic Graph"
    return normalized


def _is_overbroad_research_term(term: str) -> bool:
    normalized = _normalize_research_term(term).casefold()
    if not normalized:
        return True
    return normalized in {
        "named entity recognition",
        "entity linking",
        "concept linking",
        "knowledge graph",
        "knowledge graphs",
    }


def _domain_specificity_bonus(term: str) -> float:
    normalized = _normalize_research_term(term).casefold()
    if not normalized:
        return 0.0
    bonus = 0.0
    if any(fragment in normalized for fragment in ("academic", "scholar", "scholarly", "open academic graph")):
        bonus += 2.2
    if any(fragment in normalized for fragment in ("author", "disambiguation", "ambiguity")):
        bonus += 1.8
    if any(fragment in normalized for fragment in ("graph mining", "knowledge graph")):
        bonus += 0.8
    if _is_overbroad_research_term(normalized):
        bonus -= 1.8
    return bonus


def _infer_domain_arxiv_categories(domain_name: str, domain_keywords: list[str], fallback_categories: list[str]) -> list[str]:
    text = " ".join([_normalize_research_term(domain_name), *[_normalize_research_term(item) for item in domain_keywords]]).casefold()
    categories: list[str] = []
    if any(fragment in text for fragment in ("academic", "graph", "knowledge graph", "retrieval", "search")):
        categories.extend(["cs.IR", "cs.AI"])
    if any(fragment in text for fragment in ("entity linking", "named entity", "disambiguation", "ambiguity", "language")):
        categories.extend(["cs.CL", "cs.IR"])
    if any(fragment in text for fragment in ("mining", "graph mining")):
        categories.extend(["cs.IR", "cs.LG"])
    categories = dedupe_preserve_order([item for item in categories if clean_text(item)])
    if categories:
        return categories[:3]
    fallback = [clean_text(item) for item in fallback_categories if clean_text(item)]
    return fallback[:3]


def _collect_scholar_term_roles(components: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, float]]:
    role_map: dict[str, str] = {}
    weight_map: dict[str, float] = {}
    role_priority = {
        "scholar_specific": 5,
        "core_domain": 4,
        "broad_superordinate": 3,
        "method": 2,
        "auxiliary": 1,
        "noise": 0,
    }
    for component in components:
        if not isinstance(component, dict):
            continue
        scholar_term_labeling = component.get("scholar_term_labeling") or {}
        if not isinstance(scholar_term_labeling, dict):
            continue
        for item in list(scholar_term_labeling.get("labels") or []):
            if not isinstance(item, dict):
                continue
            term = _normalize_research_term(item.get("term"))
            role = clean_text(item.get("role"))
            if not term or not role:
                continue
            key = term.casefold()
            current_role = role_map.get(key, "")
            if role_priority.get(role, -1) >= role_priority.get(current_role, -1):
                role_map[key] = role
            try:
                weight = float(item.get("weight") or 0.0)
            except (TypeError, ValueError):
                weight = 0.0
            weight_map[key] = max(weight_map.get(key, 0.0), weight)
    return role_map, weight_map


def _phrase_similarity(left: str, right: str) -> float:
    normalized_left = _normalize_term(left)
    normalized_right = _normalize_term(right)
    if not normalized_left or not normalized_right:
        return 0.0
    left_lower = normalized_left.casefold()
    right_lower = normalized_right.casefold()
    if left_lower == right_lower:
        return 4.0
    left_tokens = _term_token_set(normalized_left)
    right_tokens = _term_token_set(normalized_right)
    overlap = len(left_tokens.intersection(right_tokens))
    phrase_bonus = 0.0
    if left_lower in right_lower or right_lower in left_lower:
        phrase_bonus = 1.5
    return overlap + phrase_bonus + SequenceMatcher(None, left_lower, right_lower).ratio()


def _best_matching_core_topic(domain_name: str, core_topics: list[dict[str, Any]]) -> dict[str, Any] | None:
    best_topic: dict[str, Any] | None = None
    best_score = 0.0
    for topic in core_topics:
        if not isinstance(topic, dict):
            continue
        name = _normalize_term(topic.get("name"))
        if not name:
            continue
        score = _phrase_similarity(domain_name, name)
        for keyword in list(topic.get("keywords") or [])[:4]:
            score = max(score, _phrase_similarity(domain_name, keyword) + 0.4)
        if score > best_score:
            best_score = score
            best_topic = topic
    if best_score < 1.1:
        return None
    return best_topic


def _build_domain_keywords(
    *,
    domain_name: str,
    core_topic: dict[str, Any] | None,
    keyword_pool: list[str],
    exclude_terms: list[str],
    term_roles: dict[str, str] | None = None,
    domain_name_set: set[str] | None = None,
    limit: int = 6,
) -> list[str]:
    normalized_domain = _normalize_term(domain_name)
    domain_key = normalized_domain.casefold()
    domain_tokens = _term_token_set(normalized_domain)
    exclude_keys = {_normalize_term(term).casefold() for term in exclude_terms if _normalize_term(term)}
    domain_name_keys = {key.casefold() for key in (domain_name_set or set()) if key}
    selected: list[str] = []
    selected_keys: set[str] = set()

    def _add_term(term: str) -> None:
        normalized = _normalize_research_term(term)
        if not normalized:
            return
        key = normalized.casefold()
        role = clean_text((term_roles or {}).get(key))
        if role in {"method", "auxiliary", "noise"}:
            return
        if _is_research_exclude_term(normalized):
            return
        if _is_overbroad_research_term(normalized) and key != domain_key:
            return
        if key in exclude_keys or key in selected_keys:
            return
        if key in domain_name_keys and key != domain_key:
            return
        selected.append(normalized)
        selected_keys.add(key)

    _add_term(normalized_domain)
    if core_topic and isinstance(core_topic, dict):
        _add_term(core_topic.get("name"))
        for keyword in list(core_topic.get("keywords") or [])[:4]:
            _add_term(keyword)

    scored_terms: list[tuple[float, str]] = []
    core_tokens = _term_token_set(core_topic.get("name")) if core_topic else set()
    core_keywords = {
        _normalize_term(keyword).casefold()
        for keyword in (list(core_topic.get("keywords") or []) if isinstance(core_topic, dict) else [])
        if _normalize_term(keyword)
    }
    for term in keyword_pool:
        normalized = _normalize_research_term(term)
        if not normalized:
            continue
        key = normalized.casefold()
        role = clean_text((term_roles or {}).get(key))
        if key in exclude_keys or key in selected_keys:
            continue
        if key in domain_name_keys and key != domain_key:
            continue
        if role in {"method", "auxiliary", "noise"} or _is_research_exclude_term(normalized):
            continue
        tokens = _term_token_set(normalized)
        score = 0.0
        score += len(domain_tokens.intersection(tokens)) * 1.4
        score += len(core_tokens.intersection(tokens)) * 0.9
        if key in core_keywords:
            score += 2.5
        if role == "scholar_specific":
            score += 1.6
        elif role == "core_domain":
            score += 1.0
        elif role == "broad_superordinate":
            score += 0.2
        score += _domain_specificity_bonus(normalized)
        score += _specificity_bonus(normalized) * 0.5
        if score <= 0.15:
            continue
        scored_terms.append((score, normalized))

    scored_terms.sort(key=lambda item: (item[0], len(_term_token_set(item[1])), item[1].casefold()), reverse=True)
    for _, term in scored_terms:
        _add_term(term)
        if len(selected) >= max(int(limit), 1):
            break
    return selected[: max(int(limit), 1)]


def _build_research_domain_profile(
    merged_profile: dict[str, Any],
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    domain_scores: dict[str, float] = defaultdict(float)
    domain_labels: dict[str, str] = {}
    keyword_scores: dict[str, float] = defaultdict(float)
    keyword_labels: dict[str, str] = {}
    exclude_scores: dict[str, float] = defaultdict(float)
    exclude_labels: dict[str, str] = {}
    core_topics_pool: list[dict[str, Any]] = []
    term_roles, term_label_weights = _collect_scholar_term_roles(components)

    def add_score(
        scores: dict[str, float],
        labels: dict[str, str],
        term: Any,
        *,
        weight: float,
    ) -> None:
        normalized = _normalize_research_term(term)
        if not normalized:
            return
        key = normalized.casefold()
        role = clean_text(term_roles.get(key))
        if role in {"method", "auxiliary", "noise"}:
            scores = exclude_scores
            labels = exclude_labels
            weight = max(float(weight), 2.0)
        elif scores is domain_scores and role == "broad_superordinate":
            weight *= 0.35
        elif role == "scholar_specific":
            weight *= 1.35
        elif role == "core_domain":
            weight *= 1.1
        key = normalized.casefold()
        labels.setdefault(key, normalized)
        scores[key] += float(weight) + _specificity_bonus(normalized) + min(term_label_weights.get(key, 0.0), 2.0)

    for component in components:
        if not isinstance(component, dict):
            continue
        resolved_interests = list(component.get("resolved_person_interests") or [])
        scholar_recall_topics = list(component.get("scholar_recall_topics") or [])
        scholar_rerank_topics = list(component.get("scholar_rerank_topics") or [])
        scholar_rerank_keywords = list(component.get("scholar_rerank_keywords") or [])
        core_topics = [item for item in list((component.get("core_topics") or [])) if isinstance(item, dict)]
        llm_topics = list((component.get("internal_profile") or {}).get("llm_topics") or [])
        core_topics_pool.extend(core_topics)

        for term in resolved_interests:
            if _is_research_exclude_term(term):
                add_score(exclude_scores, exclude_labels, term, weight=4.2)
                continue
            add_score(domain_scores, domain_labels, term, weight=5.2)

        for term in scholar_recall_topics:
            if _is_research_exclude_term(term):
                add_score(exclude_scores, exclude_labels, term, weight=3.6)
                continue
            add_score(domain_scores, domain_labels, term, weight=4.6)

        for item in core_topics:
            name = _normalize_term(item.get("name"))
            if not name:
                continue
            if _is_research_exclude_term(name):
                add_score(exclude_scores, exclude_labels, name, weight=4.0)
                continue
            support_count = int(item.get("support_count") or 0)
            support_bonus = min(max(support_count, 0) * 0.3, 1.2)
            topic_bonus = min(float(item.get("score") or 0.0) / 6.0, 1.6)
            add_score(domain_scores, domain_labels, name, weight=4.8 + support_bonus + topic_bonus)
            for keyword in list(item.get("keywords") or [])[:4]:
                if _is_research_exclude_term(keyword):
                    add_score(exclude_scores, exclude_labels, keyword, weight=2.5)
                    continue
                add_score(keyword_scores, keyword_labels, keyword, weight=3.0 + support_bonus + min(topic_bonus, 0.8))

        for term in scholar_rerank_topics:
            if _is_research_exclude_term(term):
                add_score(exclude_scores, exclude_labels, term, weight=3.0)
                continue
            add_score(domain_scores, domain_labels, term, weight=3.4)

        for term in scholar_rerank_keywords:
            if _is_research_exclude_term(term):
                add_score(exclude_scores, exclude_labels, term, weight=2.6)
                continue
            add_score(keyword_scores, keyword_labels, term, weight=2.2)

        for item in llm_topics:
            if not isinstance(item, dict):
                continue
            name = _normalize_term(item.get("name"))
            if name:
                if _is_research_exclude_term(name):
                    add_score(exclude_scores, exclude_labels, name, weight=3.0)
                else:
                    add_score(domain_scores, domain_labels, name, weight=3.6)
            for keyword in list(item.get("keywords") or [])[:4]:
                if _is_research_exclude_term(keyword):
                    add_score(exclude_scores, exclude_labels, keyword, weight=2.0)
                    continue
                add_score(keyword_scores, keyword_labels, keyword, weight=2.0)

        for term in list(component.get("topics") or []):
            if _is_research_exclude_term(term):
                add_score(exclude_scores, exclude_labels, term, weight=2.6)
                continue
            add_score(domain_scores, domain_labels, term, weight=3.2)

        for term in list(component.get("keywords") or []):
            if _is_research_exclude_term(term):
                add_score(exclude_scores, exclude_labels, term, weight=2.0)
                continue
            add_score(keyword_scores, keyword_labels, term, weight=1.8)

    for term in list(merged_profile.get("topics") or []):
        if _is_research_exclude_term(term):
            add_score(exclude_scores, exclude_labels, term, weight=2.2)
            continue
        add_score(domain_scores, domain_labels, term, weight=2.8)
    for term in list(merged_profile.get("keywords") or []):
        if _is_research_exclude_term(term):
            add_score(exclude_scores, exclude_labels, term, weight=1.8)
            continue
        add_score(keyword_scores, keyword_labels, term, weight=1.6)

    fallback_categories = list(merged_profile.get("arxiv_categories") or [])
    domain_ranked = sorted(
        domain_scores,
        key=lambda key: (
            domain_scores[key],
            _domain_specificity_bonus(domain_labels.get(key, "")),
            1 if term_roles.get(key) in {"scholar_specific", "core_domain"} else 0,
            0 if _is_overbroad_research_term(domain_labels.get(key, "")) else 1,
            len(_term_token_set(domain_labels.get(key, ""))),
            len(domain_labels.get(key, "")),
            domain_labels.get(key, "").casefold(),
        ),
        reverse=True,
    )
    preferred_domain_names = [
        domain_labels[key]
        for key in domain_ranked
        if term_roles.get(key) in {"scholar_specific", "core_domain"} and not _is_overbroad_research_term(domain_labels.get(key, ""))
    ]
    domain_names = _select_specific_terms(preferred_domain_names, limit=4)
    if not domain_names:
        narrowed_names = [domain_labels[key] for key in domain_ranked if not _is_overbroad_research_term(domain_labels.get(key, ""))]
        domain_names = _select_specific_terms(narrowed_names, limit=4)
    if not domain_names:
        domain_names = _select_specific_terms([domain_labels[key] for key in domain_ranked], limit=4)
    if not domain_names:
        domain_names = _select_specific_terms(
            [
                _normalize_term(term)
                for term in list(merged_profile.get("topics") or [])
                if not _is_research_exclude_term(term)
            ],
            limit=3,
        )
    if not domain_names:
        domain_names = _select_specific_terms(
            [
                _normalize_term(term)
                for term in list(merged_profile.get("keywords") or [])
                if not _is_research_exclude_term(term)
            ],
            limit=3,
        )

    keyword_ranked = sorted(
        keyword_scores,
        key=lambda key: (
            keyword_scores[key],
            len(_term_token_set(keyword_labels.get(key, ""))),
            len(keyword_labels.get(key, "")),
            keyword_labels.get(key, "").casefold(),
        ),
        reverse=True,
    )
    keyword_pool = [keyword_labels[key] for key in keyword_ranked]
    exclude_ranked = sorted(
        exclude_scores,
        key=lambda key: (
            exclude_scores[key],
            len(_term_token_set(exclude_labels.get(key, ""))),
            len(exclude_labels.get(key, "")),
            exclude_labels.get(key, "").casefold(),
        ),
        reverse=True,
    )
    excluded_keywords = _select_specific_terms(
        [
            exclude_labels[key]
            for key in exclude_ranked
            if not any(
                fragment in _normalize_research_term(exclude_labels[key]).casefold()
                for fragment in ("academic", "author", "disambiguation", "ambiguity", "open academic graph")
            )
        ],
        limit=12,
    )
    research_domains: list[dict[str, Any]] = []
    domain_weights: dict[str, float] = {}
    domain_name_set = {domain_name.casefold() for domain_name in domain_names}
    for index, domain_name in enumerate(domain_names[:4]):
        matched_topic = _best_matching_core_topic(domain_name, core_topics_pool)
        domain_keywords = _build_domain_keywords(
            domain_name=domain_name,
            core_topic=matched_topic,
            keyword_pool=keyword_pool,
            exclude_terms=excluded_keywords,
            term_roles=term_roles,
            domain_name_set=domain_name_set,
            limit=6,
        )
        if not domain_keywords:
            domain_keywords = [domain_name]
        priority_key = domain_name.casefold()
        base_priority = float(domain_scores.get(priority_key, 0.0))
        if matched_topic:
            base_priority = max(base_priority, float(matched_topic.get("score") or 0.0))
        priority = round(max(base_priority, 1.0), 3)
        domain_categories = _infer_domain_arxiv_categories(domain_name, domain_keywords, fallback_categories)
        research_domains.append(
            {
                "name": domain_name,
                "keywords": domain_keywords,
                "arxiv_categories": domain_categories,
                "exclude_keywords": excluded_keywords[:12],
                "priority": priority,
            }
        )
        domain_weights[domain_name] = priority
        for keyword in domain_keywords:
            domain_weights.setdefault(keyword, max(priority * 0.6, 1.0))
    flat_keywords = _prioritize_keywords(
        [
            keyword
            for domain in research_domains
            for keyword in list(domain.get("keywords") or [])
        ]
    )[:18]
    flat_domain_names = [domain["name"] for domain in research_domains]
    return {
        "research_domains": research_domains,
        "excluded_keywords": excluded_keywords,
        "retrieval_topics": flat_domain_names,
        "retrieval_keywords": flat_keywords,
        "retrieval_term_weights": domain_weights,
        "ranking_topics": flat_domain_names[:10],
        "ranking_keywords": flat_keywords,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _split_text_items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[,;/；，、|\n]+", value) if item.strip()]
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            if isinstance(item, dict):
                text = clean_text(
                    item.get("name")
                    or item.get("display_name")
                    or item.get("label")
                    or item.get("text")
                    or item.get("value")
                )
                if text:
                    items.append(text)
                continue
            text = clean_text(item)
            if text:
                items.append(text)
        return dedupe_preserve_order(items)
    return [clean_text(value)] if clean_text(value) else []


def _extract_year(value: Any) -> int:
    text = clean_text(value)
    if not text:
        return 0
    match = re.search(r"(19|20)\d{2}", text)
    return int(match.group(0)) if match else _safe_int(value)


def _normalize_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", clean_text(text)).casefold()


def _matches_name(candidate: str, scholar_name: str) -> bool:
    normalized_candidate = _normalize_phrase(candidate)
    normalized_name = _normalize_phrase(scholar_name)
    if not normalized_candidate or not normalized_name:
        return False
    if normalized_name in normalized_candidate or normalized_candidate in normalized_name:
        return True
    name_tokens = [token for token in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized_name) if token]
    return bool(name_tokens) and all(token in normalized_candidate for token in name_tokens)


def _profile_match(text: str, phrase: str) -> bool:
    normalized_text = _normalize_phrase(text)
    normalized_phrase = _normalize_phrase(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    if re.search(r"[\u4e00-\u9fff]", normalized_phrase):
        return normalized_phrase in normalized_text
    tokens = re.findall(r"[a-z0-9]+", normalized_phrase)
    return bool(tokens) and all(token in normalized_text for token in tokens)


def _resolve_scholar_profile_recent_years(config: dict[str, Any] | None = None) -> int:
    search_config = (config or {}).get("search") if isinstance((config or {}).get("search"), dict) else {}
    try:
        value = int(search_config.get("scholar_profile_recent_years") or DEFAULT_SCHOLAR_PROFILE_RECENT_YEARS)
    except (TypeError, ValueError):
        value = DEFAULT_SCHOLAR_PROFILE_RECENT_YEARS
    return max(value, 1)


def _resolve_scholar_profile_max_papers(config: dict[str, Any] | None = None) -> int:
    search_config = (config or {}).get("search") if isinstance((config or {}).get("search"), dict) else {}
    try:
        value = int(search_config.get("scholar_profile_max_papers") or DEFAULT_SCHOLAR_PROFILE_MAX_PAPERS)
    except (TypeError, ValueError):
        value = DEFAULT_SCHOLAR_PROFILE_MAX_PAPERS
    return max(value, 8)


def _select_scholar_profile_papers(
    papers: list[dict[str, Any]],
    *,
    max_papers: int,
    recent_year_window: int,
) -> list[dict[str, Any]]:
    if not papers:
        return []

    current_year = datetime.now(timezone.utc).year
    recent_cutoff_year = current_year - max(recent_year_window, 1) + 1

    def _paper_sort_key(paper: dict[str, Any]) -> tuple[int, int, int, str]:
        citations = _safe_int(paper.get("citations") or paper.get("n_citation"))
        year = _extract_year(paper.get("year") or paper.get("published") or paper.get("published_date"))
        title = clean_text(paper.get("title"))
        return (
            year,
            citations,
            len(_author_names_from_paper(paper)),
            title.casefold(),
        )

    def _anchor_sort_key(paper: dict[str, Any]) -> tuple[int, int, int, str]:
        citations = _safe_int(paper.get("citations") or paper.get("n_citation"))
        year = _extract_year(paper.get("year") or paper.get("published") or paper.get("published_date"))
        title = clean_text(paper.get("title"))
        return (
            citations,
            year,
            len(_author_names_from_paper(paper)),
            title.casefold(),
        )

    recent: list[dict[str, Any]] = []
    older: list[dict[str, Any]] = []
    unknown_year: list[dict[str, Any]] = []
    for paper in papers:
        year = _extract_year(paper.get("year") or paper.get("published") or paper.get("published_date"))
        if not year:
            unknown_year.append(paper)
        elif year >= recent_cutoff_year:
            recent.append(paper)
        else:
            older.append(paper)

    recent_sorted = sorted(recent, key=_paper_sort_key, reverse=True)
    older_sorted = sorted(older, key=_anchor_sort_key, reverse=True)
    unknown_sorted = sorted(unknown_year, key=_anchor_sort_key, reverse=True)

    first_block_size = min(DEFAULT_SCHOLAR_PROFILE_SEED_PAPERS, max_papers)
    recent_front_count = min(len(recent_sorted), max(4, first_block_size - 4))
    recent_front = recent_sorted[:recent_front_count]
    remaining_recent = recent_sorted[recent_front_count:]

    recent_front_ids = {
        clean_text(paper.get("aminer_paper_id") or paper.get("paper_id") or paper.get("title"))
        for paper in recent_front
        if clean_text(paper.get("aminer_paper_id") or paper.get("paper_id") or paper.get("title"))
    }
    anchor_pool = [
        paper
        for paper in [*older_sorted, *remaining_recent, *unknown_sorted]
        if clean_text(paper.get("aminer_paper_id") or paper.get("paper_id") or paper.get("title")) not in recent_front_ids
    ]
    anchor_pool.sort(key=_anchor_sort_key, reverse=True)
    anchor_front_count = min(len(anchor_pool), max(2, first_block_size - len(recent_front)))
    anchor_front = anchor_pool[:anchor_front_count]
    anchor_front_ids = {
        clean_text(paper.get("aminer_paper_id") or paper.get("paper_id") or paper.get("title"))
        for paper in anchor_front
        if clean_text(paper.get("aminer_paper_id") or paper.get("paper_id") or paper.get("title"))
    }

    remaining_pool = [
        paper
        for paper in [*remaining_recent, *older_sorted, *unknown_sorted]
        if clean_text(paper.get("aminer_paper_id") or paper.get("paper_id") or paper.get("title")) not in anchor_front_ids
    ]
    selected = [*recent_front, *anchor_front, *remaining_pool]

    deduped: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for paper in selected:
        key = clean_text(paper.get("aminer_paper_id") or paper.get("paper_id") or paper.get("title"))
        if not key or key in seen_ids:
            continue
        seen_ids.add(key)
        deduped.append(paper)
        if len(deduped) >= max_papers:
            break
    return deduped


def _author_names_from_paper(paper: dict[str, Any]) -> list[str]:
    names = _split_text_items(paper.get("authors"))
    for entry in paper.get("author_entries") or []:
        if isinstance(entry, dict):
            text = clean_text(entry.get("display_name") or entry.get("name"))
            if text:
                names.append(text)
    for profile in paper.get("aminer_author_profiles") or []:
        if isinstance(profile, dict):
            text = clean_text(profile.get("name") or profile.get("name_zh"))
            if text:
                names.append(text)
    return dedupe_preserve_order(names)


def _paper_matches_person_id(paper: dict[str, Any], person_id: str) -> bool:
    resolved_person_id = clean_text(person_id)
    if not resolved_person_id:
        return False
    for profile in paper.get("aminer_author_profiles") or []:
        if not isinstance(profile, dict):
            continue
        if clean_text(profile.get("author_id")) == resolved_person_id:
            return True
    for entry in paper.get("author_entries") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("profile_url", "").rstrip("/").endswith(f"/{resolved_person_id}"):
            return True
    return False


def _paper_similarity(query_title: str, paper: dict[str, Any]) -> float:
    title = clean_text(paper.get("title"))
    if not title:
        return 0.0
    normalized_query = _normalize_phrase(query_title)
    normalized_title = _normalize_phrase(title)
    if normalized_query and normalized_query == normalized_title:
        return 1.0
    return SequenceMatcher(None, normalized_query, normalized_title).ratio()


def _pick_best_title_match(query_title: str, papers: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked = sorted(
        papers,
        key=lambda item: (
            _paper_similarity(query_title, item),
            _safe_int(item.get("citations") or item.get("n_citation")),
            _extract_year(item.get("year") or item.get("published") or item.get("published_date")),
        ),
        reverse=True,
    )
    if not ranked or _paper_similarity(query_title, ranked[0]) < 0.58:
        return None
    return ranked[0]


def _enrich_papers_with_aminer_details(papers: list[dict[str, Any]], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    aminer_config = (config or {}).get("aminer") if isinstance((config or {}).get("aminer"), dict) else {}
    token = clean_text(aminer_config.get("token"))
    if not papers or not token:
        return papers
    try:
        enriched = enrich_ranked_payload_with_aminer_details({"papers": papers}, token=token)
    except Exception:
        return papers
    return list(enriched.get("papers") or papers)


def _normalize_seed_paper(paper: dict[str, Any], scholar_name: str = "") -> dict[str, Any]:
    authors = _author_names_from_paper(paper)
    coauthors = [author for author in authors if not scholar_name or not _matches_name(author, scholar_name)]
    fields = _split_text_items(paper.get("fields")) + _split_text_items(paper.get("subjects")) + _split_text_items(paper.get("categories"))
    topics = _split_text_items(paper.get("topics"))
    keywords = _split_text_items(paper.get("keywords")) + _split_text_items(paper.get("interests"))
    for profile in paper.get("aminer_author_profiles") or []:
        if not isinstance(profile, dict):
            continue
        if scholar_name and not _matches_name(profile.get("name") or profile.get("name_zh") or "", scholar_name):
            continue
        keywords.extend(_split_text_items(profile.get("interests")))
        topics.extend(_split_text_items(profile.get("interests")))
    return {
        "paper_id": clean_text(paper.get("paper_id") or paper.get("aminer_paper_id") or paper.get("arxiv_id") or paper.get("title")),
        "title": clean_text(paper.get("title")),
        "abstract": clean_text(paper.get("abstract") or paper.get("summary")),
        "keywords": dedupe_preserve_order(keywords),
        "fields": dedupe_preserve_order(fields),
        "topics": dedupe_preserve_order(topics),
        "venue": clean_text(paper.get("venue")),
        "year": _extract_year(paper.get("year") or paper.get("published") or paper.get("published_date")),
        "n_citation": _safe_int(paper.get("citations") or paper.get("n_citation")),
        "coauthor_names": dedupe_preserve_order(coauthors),
    }


def _fetch_dual_bucket_seed_papers(
    scholar_name: str,
    scholar_org: str,
    *,
    resolved_person: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Fetch dual-bucket seed papers for scholar profiling.

    Args:
        scholar_name: Scholar name to search for
        scholar_org: Scholar organization (optional)
        resolved_person: Pre-resolved person info (optional)
        config: Configuration dictionary

    Returns:
        Tuple of (recent_seed_papers, anchor_seed_papers, resolved_person)
        - recent_seed_papers: arXiv papers from last 3 years (max 6)
        - anchor_seed_papers: AMiner high-citation papers (max 4)
    """
    config = config or {}
    name = clean_text(scholar_name)
    if not name:
        return [], [], {}

    # Resolve person if not provided
    if not resolved_person:
        person_payload = search_persons(name=name, org=scholar_org, size=10, config=config)
        persons = list(person_payload.get("persons") or [])
        if not persons and clean_text(scholar_org):
            person_payload = search_persons(name=name, org="", size=10, config=config)
            persons = list(person_payload.get("persons") or [])

        if persons:
            ranked_people = sorted(
                persons,
                key=lambda person: (
                    _safe_int(person.get("n_citation")),
                    1 if _profile_match(clean_text(person.get("org")), scholar_org) else 0,
                ),
                reverse=True,
            )
            resolved_person = ranked_people[0] if ranked_people else {}

    person_id = clean_text((resolved_person or {}).get("id"))
    arxiv_author_name = clean_text(
        (resolved_person or {}).get("name")
        or (resolved_person or {}).get("display_name")
        or (resolved_person or {}).get("name_zh")
        or scholar_name
    )
    recent_seed_papers: list[dict[str, Any]] = []
    anchor_seed_papers: list[dict[str, Any]] = []

    # Bucket 1: Recent high-priority papers from arXiv author search
    try:
        arxiv_papers = fetch_arxiv_author_papers(
            arxiv_author_name,
            lookback_days=DUAL_BUCKET_RECENT_LOOKBACK_DAYS,
            max_results=30,
            config=config,
        )
        # Normalize and dedupe
        seen_titles: set[str] = set()
        for paper in arxiv_papers:
            title = clean_text(paper.get("title"))
            if not title or title.casefold() in seen_titles:
                continue
            seen_titles.add(title.casefold())
            recent_seed_papers.append(_normalize_seed_paper(paper, scholar_name=name))
            if len(recent_seed_papers) >= DUAL_BUCKET_RECENT_MAX_PAPERS:
                break
    except Exception:
        pass

    # Bucket 2: High-citation anchor papers from AMiner
    if person_id:
        try:
            page_size = 20
            all_papers: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            stagnant_pages = 0
            max_pages = 8
            for page_index in range(max_pages):
                before_count = len(all_papers)
                papers_payload = search_person_papers(
                    person_id=person_id,
                    size=page_size,
                    offset=page_index * page_size,
                    config=config,
                )
                page_papers = list(papers_payload.get("papers") or [])
                if not page_papers:
                    break
                for paper in page_papers:
                    key = clean_text(paper.get("aminer_paper_id") or paper.get("title"))
                    if not key or key in seen_ids:
                        continue
                    seen_ids.add(key)
                    all_papers.append(paper)
                if len(all_papers) == before_count:
                    stagnant_pages += 1
                else:
                    stagnant_pages = 0
                if len(all_papers) >= max(page_size * 2, DUAL_BUCKET_ANCHOR_MAX_PAPERS * 2):
                    break
                if stagnant_pages >= 2:
                    break

            # Sort by citations (descending) to get anchor papers
            sorted_papers = sorted(
                all_papers,
                key=lambda p: (
                    _safe_int(p.get("citations") or p.get("n_citation")),
                    _extract_year(p.get("year") or p.get("published")),
                ),
                reverse=True,
            )

            # Exclude papers already in recent bucket
            recent_titles = {p.get("title", "").casefold() for p in recent_seed_papers if p.get("title")}
            for paper in sorted_papers:
                title = clean_text(paper.get("title"))
                if not title or title.casefold() in recent_titles:
                    continue
                anchor_seed_papers.append(_normalize_seed_paper(paper, scholar_name=name))
                if len(anchor_seed_papers) >= DUAL_BUCKET_ANCHOR_MAX_PAPERS:
                    break
        except Exception:
            pass

    return recent_seed_papers, anchor_seed_papers, resolved_person or {}


def _compute_bucket_topic_intersection(
    anchor_core_topics: list[dict[str, Any]],
    recent_core_topics: list[dict[str, Any]],
    similarity_threshold: float = 0.75,
) -> tuple[list[str], list[str], list[str]]:
    """Compute intersection and difference of topics between buckets.

    Uses token set similarity rather than exact matching.

    Args:
        anchor_core_topics: Core topics from anchor bucket
        recent_core_topics: Core topics from recent bucket
        similarity_threshold: Threshold for considering topics similar

    Returns:
        Tuple of (primary_topics, anchor_only_topics, recent_only_topics)
    """
    def _token_set(topic_name: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", _normalize_term(topic_name).casefold()))

    def _topics_similar(t1: str, t2: str) -> bool:
        tokens1 = _token_set(t1)
        tokens2 = _token_set(t2)
        if not tokens1 or not tokens2:
            return False
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        return (intersection / union) >= similarity_threshold if union > 0 else False

    anchor_topic_names = [t.get("name", "") for t in anchor_core_topics if t.get("name")]
    recent_topic_names = [t.get("name", "") for t in recent_core_topics if t.get("name")]

    primary_topics: list[str] = []
    anchor_only_topics: list[str] = []
    recent_only_topics: list[str] = []

    # Find primary (intersection) topics
    matched_anchor_indices: set[int] = set()
    matched_recent_indices: set[int] = set()

    for a_idx, anchor_topic in enumerate(anchor_topic_names):
        for r_idx, recent_topic in enumerate(recent_topic_names):
            if r_idx in matched_recent_indices:
                continue
            if _topics_similar(anchor_topic, recent_topic):
                primary_topics.append(anchor_topic)
                matched_anchor_indices.add(a_idx)
                matched_recent_indices.add(r_idx)
                break

    # Collect anchor-only topics
    for a_idx, anchor_topic in enumerate(anchor_topic_names):
        if a_idx not in matched_anchor_indices:
            anchor_only_topics.append(anchor_topic)

    # Collect recent-only topics
    for r_idx, recent_topic in enumerate(recent_topic_names):
        if r_idx not in matched_recent_indices:
            recent_only_topics.append(recent_topic)

    return primary_topics, anchor_only_topics, recent_only_topics


def _build_dual_bucket_signal_layers(
    *,
    recent_topics: list[str],
    recent_keywords: list[str],
    recent_core_topics: list[dict[str, Any]],
    anchor_topics: list[str],
    anchor_keywords: list[str],
    anchor_core_topics: list[dict[str, Any]],
    explicit_topics: list[str],
    resolved_interests: list[str],
) -> dict[str, Any]:
    """Build layered recall terms from dual-bucket signals.

    Returns dict with primary/anchor_only/recent_only layers.
    """
    # Compute intersection
    primary_topics, anchor_only_topics, recent_only_topics = _compute_bucket_topic_intersection(
        anchor_core_topics, recent_core_topics
    )

    cleaned_primary_topics = _clean_dual_bucket_topics(primary_topics, limit=6)
    cleaned_anchor_topics = _clean_dual_bucket_topics(anchor_only_topics, limit=8)
    cleaned_recent_topics = _clean_dual_bucket_topics(recent_only_topics, limit=8)
    cleaned_explicit_topics = _clean_dual_bucket_topics(explicit_topics, limit=4)

    # Explicit topics can shape primary recall, but resolved AMiner interests should
    # not directly enter recall layers before they are supported by paper buckets.
    primary_topics = dedupe_preserve_order([*cleaned_explicit_topics, *cleaned_primary_topics])[:6]
    anchor_only_topics = cleaned_anchor_topics[:8]
    recent_only_topics = cleaned_recent_topics[:8]

    # Build recall terms for each layer
    primary_recall_terms = _select_specific_terms(primary_topics, limit=5)

    # Anchor-only terms (long-term identity)
    anchor_recall_terms = _select_specific_terms(
        [t for t in anchor_only_topics if not _should_skip_retrieval_term(t)],
        limit=4
    )

    # Recent-only terms (new directions) - limit to prevent recent dominance
    recent_recall_terms = _select_specific_terms(
        [t for t in recent_only_topics if not _should_skip_retrieval_term(t) and not _looks_like_method_topic(t)],
        limit=3
    )

    # Build weight map
    weight_map: dict[str, float] = {}
    for term in primary_recall_terms:
        weight_map[term] = SOURCE_PRIOR_WEIGHTS["primary"] * (5.0 + _specificity_bonus(term))
    for term in anchor_recall_terms:
        weight_map[term] = SOURCE_PRIOR_WEIGHTS["anchor"] * (4.0 + _specificity_bonus(term))
    for term in recent_recall_terms:
        weight_map[term] = SOURCE_PRIOR_WEIGHTS["recent"] * (2.0 + _specificity_bonus(term))

    # Combined topics/keywords for compatibility
    cleaned_resolved_interests = _clean_dual_bucket_topics(resolved_interests, limit=6)
    all_topics = dedupe_preserve_order([*primary_topics, *anchor_only_topics, *recent_only_topics, *cleaned_resolved_interests])[:10]
    all_keywords = dedupe_preserve_order(
        [
            *[keyword for keyword in anchor_keywords if not _looks_like_bucket_noise_topic(keyword)],
            *[keyword for keyword in recent_keywords if not _looks_like_bucket_noise_topic(keyword)],
        ]
    )[:16]

    return {
        "primary_recall_terms": primary_recall_terms,
        "anchor_recall_terms": anchor_recall_terms,
        "recent_recall_terms": recent_recall_terms,
        "primary_topics": primary_topics,
        "anchor_only_topics": anchor_only_topics,
        "recent_only_topics": recent_only_topics,
        "topics": all_topics,
        "keywords": all_keywords,
        "weight_map": weight_map,
        "source_priors": dict(SOURCE_PRIOR_WEIGHTS),
        "dual_match_bonus": DUAL_BUCKET_MATCH_BONUS,
    }


def _profile_from_dual_bucket_papers(
    recent_seed_papers: list[dict[str, Any]],
    anchor_seed_papers: list[dict[str, Any]],
    *,
    profile_name: str,
    source: str,
    explicit_topics: list[str],
    scholar_name: str,
    scholar_org: str,
    bind_scholar_id: str = "",
    resolved_person: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build scholar profile from dual-bucket papers.

    Args:
        recent_seed_papers: arXiv recent papers (last 3 years)
        anchor_seed_papers: AMiner high-citation papers
        profile_name: Name for the profile
        source: Source identifier
        explicit_topics: User-specified topics
        scholar_name: Scholar name
        scholar_org: Scholar organization
        bind_scholar_id: Bound scholar ID
        resolved_person: Resolved person info
        config: Configuration

    Returns:
        Profile dictionary with dual-bucket metadata
    """
    config = config or {}

    # Score each bucket separately
    recent_topics, recent_keywords, recent_core_topics = _score_terms_for_bucket(
        recent_seed_papers, "recent"
    )
    anchor_topics, anchor_keywords, anchor_core_topics = _score_terms_for_bucket(
        anchor_seed_papers, "anchor"
    )

    # Get resolved interests
    resolved_interests = _normalize_topics(_split_text_items((resolved_person or {}).get("interests")))

    # Build dual-bucket signal layers
    dual_bucket_terms = _build_dual_bucket_signal_layers(
        recent_topics=recent_topics,
        recent_keywords=recent_keywords,
        recent_core_topics=recent_core_topics,
        anchor_topics=anchor_topics,
        anchor_keywords=anchor_keywords,
        anchor_core_topics=anchor_core_topics,
        explicit_topics=explicit_topics,
        resolved_interests=resolved_interests,
    )

    # Constraint: recent-only cannot dominate primary recall
    if not dual_bucket_terms.get("primary_recall_terms") and dual_bucket_terms.get("recent_recall_terms"):
        # Downgrade: limit recent-only to 2 terms
        dual_bucket_terms["recent_recall_terms"] = dual_bucket_terms["recent_recall_terms"][:2]

    # Build combined seed papers
    combined_seed_papers = [*recent_seed_papers, *anchor_seed_papers]

    # Get categories
    categories = _categories_from_topics(
        dual_bucket_terms.get("topics", []),
        dual_bucket_terms.get("keywords", [])
    )

    # Build recall strategy
    recall_strategy = _build_recall_strategy(categories)

    # Build rerank keywords
    rerank_keywords = _prioritize_keywords([
        *dual_bucket_terms.get("primary_recall_terms", []),
        *dual_bucket_terms.get("anchor_recall_terms", []),
        *dual_bucket_terms.get("keywords", []),
    ])[:16]

    return {
        "status": "success",
        "profile_mode": "scholar_path",
        "profile_name": profile_name,
        "user_name": profile_name,
        "bind_scholar_ids": [bind_scholar_id] if bind_scholar_id else [],
        "topics": dual_bucket_terms.get("topics", []),
        "keywords": dual_bucket_terms.get("keywords", []),
        "arxiv_categories": categories,
        "recall_strategy": recall_strategy,
        # Dual-bucket specific fields
        "recent_seed_papers": recent_seed_papers,
        "anchor_seed_papers": anchor_seed_papers,
        "seed_papers": combined_seed_papers,
        "primary_topics": dual_bucket_terms.get("primary_topics", []),
        "anchor_topics": dual_bucket_terms.get("anchor_only_topics", []),
        "recent_topics": dual_bucket_terms.get("recent_only_topics", []),
        "primary_keywords": dual_bucket_terms.get("primary_recall_terms", []),
        "anchor_keywords": dual_bucket_terms.get("anchor_recall_terms", []),
        "recent_keywords": dual_bucket_terms.get("recent_recall_terms", []),
        "dual_bucket_layered_recall_terms": {
            "primary_recall_terms": dual_bucket_terms.get("primary_recall_terms", []),
            "anchor_recall_terms": dual_bucket_terms.get("anchor_recall_terms", []),
            "recent_recall_terms": dual_bucket_terms.get("recent_recall_terms", []),
        },
        "dual_bucket_source_priors": {
            "primary": SOURCE_PRIOR_WEIGHTS["primary"],
            "anchor": SOURCE_PRIOR_WEIGHTS["anchor"],
            "recent": SOURCE_PRIOR_WEIGHTS["recent"],
            "dual_match_bonus": DUAL_BUCKET_MATCH_BONUS,
        },
        "scholar_recall_topics": dual_bucket_terms.get("primary_recall_terms", []),
        "scholar_rerank_topics": dual_bucket_terms.get("topics", [])[:8],
        "scholar_rerank_keywords": rerank_keywords,
        "scholar_term_weights": dual_bucket_terms.get("weight_map", {}),
        "ranking_topics": dual_bucket_terms.get("topics", [])[:8],
        "ranking_keywords": rerank_keywords,
        "source_metadata": {
            "source": source,
            "scholar_name": scholar_name,
            "scholar_org": scholar_org,
            "recent_paper_count": len(recent_seed_papers),
            "anchor_paper_count": len(anchor_seed_papers),
            "resolved_person_id": clean_text((resolved_person or {}).get("id")),
            "resolved_interests": resolved_interests,
            "recent_core_topics": recent_core_topics,
            "anchor_core_topics": anchor_core_topics,
        },
    }


def _profile_from_authored_papers(
    papers: list[dict[str, Any]],
    *,
    profile_name: str,
    source: str,
    explicit_topics: list[str],
    scholar_name: str,
    scholar_org: str,
    bind_scholar_id: str = "",
    resolved_person: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed_paper_limit = DEFAULT_SCHOLAR_PROFILE_SEED_PAPERS
    selected_papers = _select_scholar_profile_papers(
        [paper for paper in papers if clean_text(paper.get("title"))],
        max_papers=_resolve_scholar_profile_max_papers(config=config),
        recent_year_window=_resolve_scholar_profile_recent_years(config=config),
    )
    normalized_papers = [_normalize_seed_paper(paper, scholar_name=scholar_name) for paper in selected_papers]
    if not normalized_papers and not explicit_topics:
        return {"status": "degraded", "enabled": False, "source_metadata": {"reason": "missing_paper_signal"}}

    inferred_topics, inferred_keywords, preferred_authors, preferred_venues, core_topics = _score_terms(normalized_papers, [])
    resolved_interests = _normalize_topics(_split_text_items((resolved_person or {}).get("interests")))
    internal_profile = {
        "status": "success",
        "user_name": profile_name,
        "bind_scholar_ids": [],
        "topics": inferred_topics,
        "keywords": inferred_keywords,
        "preferred_authors": preferred_authors,
        "preferred_venues": preferred_venues,
        "seed_papers": normalized_papers[:seed_paper_limit],
        "source_metadata": {
            "source": source,
            "authored_paper_count": len(normalized_papers),
            "sampled_paper_count": len(selected_papers),
            "recent_year_window": _resolve_scholar_profile_recent_years(config=config),
            "core_topics": core_topics,
            "scholar_name": clean_text(scholar_name),
            "scholar_org": clean_text(scholar_org),
        },
    }
    llm_topic_reason = ""
    should_attempt_llm_topics = bool(normalized_papers) and len(resolved_interests) < 3 and len(inferred_topics) < 5
    if should_attempt_llm_topics:
        internal_profile, llm_topic_reason = _maybe_apply_llm_topics(internal_profile, config=config)
    elif normalized_papers:
        llm_topic_reason = "skipped_resolved_person_interests"

    cleaned_internal_topics = _normalize_topics(list(internal_profile.get("topics") or []))
    cleaned_explicit_topics = _normalize_topics(explicit_topics)
    merged_topics = dedupe_preserve_order([*cleaned_explicit_topics, *resolved_interests, *cleaned_internal_topics])[:8]
    if not merged_topics and resolved_interests:
        merged_topics = resolved_interests[:5]
    topic_profile = build_topics_profile(merged_topics or explicit_topics, config=config, enable_llm_topics=False)
    merged_keywords = _prioritize_keywords(
        [*resolved_interests, *list(topic_profile.get("keywords") or []), *list(internal_profile.get("keywords") or []), *merged_topics]
    )
    scholar_recall_topics, scholar_rerank_topics, scholar_rerank_keywords, scholar_term_weights = _build_scholar_signal_layers(
        explicit_topics=cleaned_explicit_topics,
        resolved_interests=resolved_interests,
        core_topics=core_topics,
        seed_papers=list(internal_profile.get("seed_papers") or []),
        merged_topics=merged_topics,
        merged_keywords=merged_keywords,
    )
    (
        scholar_recall_topics,
        scholar_rerank_topics,
        scholar_rerank_keywords,
        scholar_term_weights,
        scholar_term_labeling,
    ) = _maybe_apply_llm_scholar_term_labels(
        scholar_name=scholar_name,
        scholar_org=scholar_org,
        resolved_interests=resolved_interests,
        core_topics=core_topics,
        seed_papers=list(internal_profile.get("seed_papers") or []),
        recall_topics=scholar_recall_topics,
        rerank_topics=scholar_rerank_topics,
        rerank_keywords=scholar_rerank_keywords,
        weight_map=scholar_term_weights,
        config=config,
    )
    # 提取分层召回术语
    layered_recall_terms = scholar_term_labeling.get("layered_recall_terms") if isinstance(scholar_term_labeling, dict) else None
    categories = _categories_from_topics(merged_topics, merged_keywords)
    retrieval_keywords = _prioritize_keywords(
        [
            *scholar_recall_topics,
            *[
                keyword
                for keyword in scholar_rerank_keywords
                if not _looks_like_method_topic(keyword) and not _should_skip_retrieval_term(keyword)
            ],
        ]
    )[:12]
    recall_strategy = _build_recall_strategy(categories)

    # 构建分层召回术语
    profile_layered_terms = layered_recall_terms or {
        "primary_recall_terms": scholar_recall_topics[:5],
        "fallback_recall_terms": [],
        "blocked_query_terms": [],
        "primary_count": len(scholar_recall_topics[:5]),
        "fallback_count": 0,
        "blocked_count": 0,
    }

    return {
        "status": "success",
        "enabled": True,
        "user_id": "",
        "profile_name": clean_text(profile_name) or "scholar_profile",
        "bind_scholar_ids": [clean_text(bind_scholar_id)] if clean_text(bind_scholar_id) else [],
        "topics": merged_topics,
        "keywords": merged_keywords[:18],
        "arxiv_categories": categories,
        "is_cs_user": recall_strategy["is_cs_user"],
        "recall_primary_source": recall_strategy["primary_recall_source"],
        "recall_secondary_source": recall_strategy["secondary_recall_source"],
        "recall_strategy": recall_strategy,
        "preferred_authors": list(internal_profile.get("preferred_authors") or [])[:8],
        "preferred_venues": list(internal_profile.get("preferred_venues") or [])[:6],
        "seed_papers": list(internal_profile.get("seed_papers") or [])[:seed_paper_limit],
        "profile_mode": "scholar_path",
        "retrieval_topics": scholar_recall_topics[:6],
        "retrieval_keywords": retrieval_keywords,
        "retrieval_term_weights": scholar_term_weights,
        "ranking_topics": scholar_rerank_topics[:10],
        "ranking_keywords": scholar_rerank_keywords[:18],
        "layered_recall_terms": profile_layered_terms,
        "source_metadata": {
            **dict(internal_profile.get("source_metadata") or {}),
            "source": source,
            "llm_topic_reason": llm_topic_reason,
            "resolved_person_interests": resolved_interests,
            "scholar_recall_topics": scholar_recall_topics,
            "scholar_rerank_topics": scholar_rerank_topics,
            "scholar_rerank_keywords": scholar_rerank_keywords,
            "scholar_term_weights": scholar_term_weights,
            "scholar_term_labeling": scholar_term_labeling,
            "segmented_keyword_count": int((topic_profile.get("source_metadata") or {}).get("segmented_keyword_count") or 0),
            "resolved_person": dict(resolved_person or {}),
        },
    }


def _search_papers_by_titles(paper_titles: list[str], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for title in dedupe_preserve_order([clean_text(item) for item in paper_titles]):
        if not title:
            continue
        response = search_papers_pro(title=title, keyword="", size=8, config=config or {})
        best = _pick_best_title_match(title, list(response.get("papers") or []))
        if best:
            selected.append(best)
    return _enrich_papers_with_aminer_details(selected, config=config)


def _score_person_candidate(person: dict[str, Any], scholar_name: str, scholar_org: str) -> float:
    score = 0.0
    if _matches_name(person.get("name_zh") or person.get("display_name") or "", scholar_name):
        score += 3.0
    if _matches_name(person.get("name") or "", scholar_name):
        score += 2.0
    normalized_org = _normalize_phrase(scholar_org)
    org_text = " ".join(
        [
            clean_text(person.get("org")),
            clean_text(person.get("org_zh")),
        ]
    )
    if normalized_org and _profile_match(org_text, scholar_org):
        score += 4.0
    score += min(_safe_int(person.get("n_citation")), 10000) / 10000.0
    return score


def _search_papers_by_author_hint(
    scholar_name: str,
    scholar_org: str = "",
    *,
    config: dict[str, Any] | None = None,
    max_papers: int = DEFAULT_SCHOLAR_PROFILE_MAX_PAPERS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    name = clean_text(scholar_name)
    if not name:
        return {}, []

    person_payload = search_persons(name=name, org=scholar_org, size=10, config=config)
    persons = list(person_payload.get("persons") or [])
    if not persons and clean_text(scholar_org):
        person_payload = search_persons(name=name, org="", size=10, config=config)
        persons = list(person_payload.get("persons") or [])
    ranked_people = sorted(
        persons,
        key=lambda person: _score_person_candidate(person, name, scholar_org),
        reverse=True,
    )
    for person in ranked_people[:3]:
        person_id = clean_text(person.get("id"))
        if not person_id:
            continue
        page_size = min(max(max_papers, 12), 30)
        collected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for page_index in range(4):
            papers_payload = search_person_papers(
                person_id=person_id,
                size=page_size,
                offset=page_index * page_size,
                config=config,
            )
            page_papers = list(papers_payload.get("papers") or [])
            if not page_papers:
                break
            papers = _enrich_papers_with_aminer_details(page_papers, config=config)
            filtered = [
                paper
                for paper in papers
                if _paper_matches_person_id(paper, person_id) or any(_matches_name(author, name) for author in _author_names_from_paper(paper))
            ]
            for paper in filtered:
                key = clean_text(paper.get("aminer_paper_id") or paper.get("paper_id") or paper.get("title"))
                if not key or key in seen_ids:
                    continue
                seen_ids.add(key)
                collected.append(paper)
            if len(page_papers) < page_size or len(collected) >= max(max_papers * 2, page_size):
                break
        filtered = collected
        if filtered:
            selected = _select_scholar_profile_papers(
                filtered,
                max_papers=max_papers,
                recent_year_window=_resolve_scholar_profile_recent_years(config=config),
            )
            return person, selected

    response = search_papers_pro(title="", keyword=name, size=30, config=config or {})
    papers = _enrich_papers_with_aminer_details(list(response.get("papers") or []), config=config)
    normalized_org = _normalize_phrase(scholar_org)
    scored: list[tuple[float, dict[str, Any]]] = []
    for paper in papers:
        if not any(_matches_name(author, name) for author in _author_names_from_paper(paper)):
            continue
        score = 1.0 + min(_safe_int(paper.get("citations")), 200) * 0.001
        for profile in paper.get("aminer_author_profiles") or []:
            if not isinstance(profile, dict):
                continue
            if not _matches_name(profile.get("name") or profile.get("name_zh") or "", name):
                continue
            score += 1.0
            affiliation = _normalize_phrase(profile.get("affiliation") or "")
            if normalized_org and normalized_org in affiliation:
                score += 1.5
        scored.append((score, paper))
    scored.sort(key=lambda item: item[0], reverse=True)
    selected = _select_scholar_profile_papers(
        [paper for _, paper in scored],
        max_papers=max_papers,
        recent_year_window=_resolve_scholar_profile_recent_years(config=config),
    )
    return {}, selected


def _extract_papers_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("papers", "selected_papers", "ranked_candidates", "seed_papers"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _load_papers_from_file(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"papers_file_not_found:{path}")
    if path.suffix.lower() != ".json":
        raise ValueError(f"unsupported_papers_file:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _extract_papers_from_payload(payload)


def _infer_topics_from_free_text(text: str, config: dict[str, Any] | None = None) -> list[str]:
    normalized = clean_text(text)
    if not normalized:
        return []
    topics: list[str] = []
    try:
        segmentation = call_segmentation_pro(query=normalized, user_id="", config=config)
    except Exception:
        segmentation = {}
    topics.extend(_flatten_keyword_items(segmentation.get("keywords")))
    for fragment in re.split(r"[,;/；，、\n]|(?:我做|我是|研究|方向|关注|想看|推荐)", normalized):
        compact = clean_text(fragment)
        if compact and not re.fullmatch(r"(帮我|给我|帮忙|请推荐|最近|最新|最近论文|最新论文|论文)", compact):
            topics.append(compact)
    return _normalize_topics(topics)[:8]


def _merge_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    available = [profile for profile in profiles if profile and profile.get("status") == "success"]
    if not available:
        degraded = next((profile for profile in profiles if profile), {})
        return degraded or {"status": "degraded", "source_metadata": {"reason": "profile_unavailable"}}
    topics = dedupe_preserve_order([item for profile in available for item in list(profile.get("topics") or [])])
    keywords = _prioritize_keywords([item for profile in available for item in list(profile.get("keywords") or [])] + topics)
    categories = _categories_from_topics(topics, keywords)
    recall_strategy = _build_recall_strategy(categories)
    primary = available[0]
    merged_profile = {
        "status": "success",
        "enabled": True,
        "user_id": clean_text(primary.get("user_id")),
        "profile_name": clean_text(primary.get("profile_name")) or "aminer_rec5_profile",
        "bind_scholar_ids": dedupe_preserve_order([item for profile in available for item in list(profile.get("bind_scholar_ids") or [])]),
        "topics": topics[:10],
        "keywords": keywords[:18],
        "arxiv_categories": categories,
        "is_cs_user": recall_strategy["is_cs_user"],
        "recall_primary_source": recall_strategy["primary_recall_source"],
        "recall_secondary_source": recall_strategy["secondary_recall_source"],
        "recall_strategy": recall_strategy,
        "preferred_authors": dedupe_preserve_order([item for profile in available for item in list(profile.get("preferred_authors") or [])])[:8],
        "preferred_venues": dedupe_preserve_order([item for profile in available for item in list(profile.get("preferred_venues") or [])])[:6],
        "seed_papers": [paper for profile in available for paper in list(profile.get("seed_papers") or [])][:DEFAULT_SCHOLAR_PROFILE_SEED_PAPERS],
        "profile_mode": "scholar_path" if any(str(profile.get("profile_mode")) == "scholar_path" for profile in available) else "topic_path",
        "source_metadata": {
            "source": "merged_research_profile",
            "sources": [str((profile.get("source_metadata") or {}).get("source") or "") for profile in available],
            "profile_count": len(available),
            "components": [profile.get("source_metadata") or {} for profile in available],
        },
    }
    components = list((merged_profile.get("source_metadata") or {}).get("components") or [])
    authored_paper_count = max(
        [int(component.get("authored_paper_count") or 0) for component in components if isinstance(component, dict)] or [0]
    )
    recent_year_window = max(
        [int(component.get("recent_year_window") or 0) for component in components if isinstance(component, dict)] or [0]
    )
    if authored_paper_count:
        merged_profile["source_metadata"]["authored_paper_count"] = authored_paper_count
    if recent_year_window:
        merged_profile["source_metadata"]["recent_year_window"] = recent_year_window

    def _profile_or_component_list(key: str) -> list[str]:
        direct_values = [
            item
            for profile in available
            for item in list(profile.get(key) or [])
            if clean_text(item)
        ]
        if direct_values:
            return dedupe_preserve_order(direct_values)
        return dedupe_preserve_order(
            [
                item
                for component in components
                if isinstance(component, dict)
                for item in list(component.get(key) or [])
                if clean_text(item)
            ]
        )

    scholar_recall_topics = _profile_or_component_list("scholar_recall_topics")
    scholar_rerank_topics = _profile_or_component_list("scholar_rerank_topics")
    scholar_rerank_keywords = _prioritize_keywords(_profile_or_component_list("scholar_rerank_keywords"))
    scholar_term_weights: dict[str, float] = {}
    for profile in available:
        for term, weight in dict(profile.get("scholar_term_weights") or {}).items():
            normalized = clean_text(term)
            if not normalized:
                continue
            scholar_term_weights[normalized] = max(float(weight or 0.0), float(scholar_term_weights.get(normalized, 0.0)))
    for component in components:
        if not isinstance(component, dict):
            continue
        for term, weight in dict(component.get("scholar_term_weights") or {}).items():
            normalized = clean_text(term)
            if not normalized:
                continue
            scholar_term_weights[normalized] = max(float(weight or 0.0), float(scholar_term_weights.get(normalized, 0.0)))
    if scholar_recall_topics or scholar_rerank_keywords:
        retrieval_topics = scholar_recall_topics[:6] or list(merged_profile.get("topics") or [])[:6]
        retrieval_keywords = _prioritize_keywords(
            [
                *retrieval_topics,
                *[
                    keyword
                    for keyword in scholar_rerank_keywords
                    if not _looks_like_method_topic(keyword) and not _should_skip_retrieval_term(keyword)
                ],
            ]
        )[:12]
        merged_profile["retrieval_topics"] = retrieval_topics
        merged_profile["retrieval_keywords"] = retrieval_keywords
        merged_profile["retrieval_term_weights"] = scholar_term_weights
        merged_profile["ranking_topics"] = scholar_rerank_topics[:10] or list(merged_profile.get("topics") or [])
        merged_profile["ranking_keywords"] = scholar_rerank_keywords[:18] or list(merged_profile.get("keywords") or [])
    else:
        retrieval_topics, retrieval_keywords, retrieval_term_weights = _build_retrieval_signal(merged_profile)
        merged_profile["retrieval_topics"] = retrieval_topics
        merged_profile["retrieval_keywords"] = retrieval_keywords
        merged_profile["retrieval_term_weights"] = retrieval_term_weights
        merged_profile["ranking_topics"] = list(merged_profile.get("topics") or [])
        merged_profile["ranking_keywords"] = list(merged_profile.get("keywords") or [])
    layered_recall_terms = next(
        (
            component.get("layered_recall_terms")
            or (
                dict(component.get("scholar_term_labeling") or {}).get("layered_recall_terms")
                if isinstance(component, dict)
                else None
            )
            for component in components
            if isinstance(component, dict)
            and (
                isinstance(component.get("layered_recall_terms"), dict)
                or isinstance(dict(component.get("scholar_term_labeling") or {}).get("layered_recall_terms"), dict)
            )
        ),
        None,
    )
    if layered_recall_terms:
        merged_profile["layered_recall_terms"] = layered_recall_terms
    scholar_term_labeling = next(
        (
            component.get("scholar_term_labeling")
            for component in components
            if isinstance(component, dict) and component.get("scholar_term_labeling")
        ),
        None,
    )
    if scholar_term_labeling:
        merged_profile["source_metadata"]["scholar_term_labeling"] = scholar_term_labeling

    def _first_profile_dict(key: str) -> dict[str, Any] | None:
        for profile in available:
            value = profile.get(key)
            if isinstance(value, dict) and value:
                return value
        return None

    def _merged_profile_list(key: str, limit: int | None = None) -> list[Any]:
        raw_values = [
            item
            for profile in available
            for item in list(profile.get(key) or [])
            if item is not None
        ]
        if not raw_values:
            return []
        if all(isinstance(item, str) for item in raw_values):
            merged_values = dedupe_preserve_order(raw_values)
        else:
            merged_values = []
            seen_keys: set[str] = set()
            for item in raw_values:
                if isinstance(item, dict):
                    key_value = clean_text(
                        item.get("aminer_paper_id")
                        or item.get("paper_id")
                        or item.get("arxiv_id")
                        or item.get("title")
                    )
                else:
                    key_value = clean_text(item)
                if not key_value:
                    continue
                dedupe_key = key_value.casefold()
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                merged_values.append(item)
        return merged_values[:limit] if limit else merged_values

    dual_bucket_terms = _first_profile_dict("dual_bucket_layered_recall_terms")
    if dual_bucket_terms:
        merged_profile["dual_bucket_layered_recall_terms"] = dual_bucket_terms
        merged_profile["recent_seed_papers"] = _merged_profile_list("recent_seed_papers", limit=DUAL_BUCKET_RECENT_MAX_PAPERS)
        merged_profile["anchor_seed_papers"] = _merged_profile_list("anchor_seed_papers", limit=DUAL_BUCKET_ANCHOR_MAX_PAPERS)
        merged_profile["primary_topics"] = _merged_profile_list("primary_topics")
        merged_profile["anchor_topics"] = _merged_profile_list("anchor_topics")
        merged_profile["recent_topics"] = _merged_profile_list("recent_topics")
        merged_profile["primary_keywords"] = _merged_profile_list("primary_keywords")
        merged_profile["anchor_keywords"] = _merged_profile_list("anchor_keywords")
        merged_profile["recent_keywords"] = _merged_profile_list("recent_keywords")
        dual_bucket_priors = _first_profile_dict("dual_bucket_source_priors")
        if dual_bucket_priors:
            merged_profile["dual_bucket_source_priors"] = dual_bucket_priors

    if merged_profile.get("profile_mode") == "scholar_path":
        research_domain_profile = _build_research_domain_profile(merged_profile, components)
        if list(research_domain_profile.get("research_domains") or []):
            merged_profile["research_domains"] = list(research_domain_profile.get("research_domains") or [])
            merged_profile["excluded_keywords"] = list(research_domain_profile.get("excluded_keywords") or [])
            merged_profile["source_metadata"]["research_domain_profile"] = research_domain_profile
    return merged_profile


def build_research_profile(
    *,
    aminer_user_id: str = "",
    topics: list[str] | None = None,
    scholar_name: str = "",
    scholar_org: str = "",
    paper_titles: list[str] | None = None,
    papers_file: str = "",
    free_text: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    explicit_topics = _normalize_topics(list(topics or []))
    free_text_topics = _infer_topics_from_free_text(free_text, config=config) if free_text and not explicit_topics else []
    resolved_topics = dedupe_preserve_order([*explicit_topics, *free_text_topics])

    profiles: list[dict[str, Any]] = []
    cleaned_uid = clean_text(aminer_user_id)
    if cleaned_uid:
        uid_profile = build_user_profile(cleaned_uid, resolved_topics, config=config)
        if uid_profile.get("status") == "success":
            uid_profile["profile_mode"] = "scholar_path"
        profiles.append(uid_profile)

    scholar_papers: list[dict[str, Any]] = []
    paper_titles = [clean_text(item) for item in list(paper_titles or []) if clean_text(item)]
    if papers_file:
        try:
            scholar_papers.extend(_load_papers_from_file(papers_file))
        except Exception as exc:
            profiles.append({"status": "degraded", "source_metadata": {"reason": str(exc)}})
    if paper_titles:
        scholar_papers.extend(_search_papers_by_titles(paper_titles, config=config))
    resolved_person: dict[str, Any] = {}

    # Use dual-bucket approach when scholar_name is provided
    dual_bucket_enabled = bool(scholar_name and not scholar_papers and not cleaned_uid)

    if dual_bucket_enabled:
        recent_papers, anchor_papers, resolved_person = _fetch_dual_bucket_seed_papers(
            scholar_name, scholar_org, config=config
        )
        if recent_papers or anchor_papers:
            resolved_name = clean_text(
                resolved_person.get("display_name")
                or resolved_person.get("name_zh")
                or resolved_person.get("name")
                or scholar_name
            )
            profiles.append(
                _profile_from_dual_bucket_papers(
                    recent_seed_papers=recent_papers,
                    anchor_seed_papers=anchor_papers,
                    profile_name=resolved_name or "scholar_profile",
                    source="dual_bucket_profile",
                    explicit_topics=resolved_topics,
                    scholar_name=scholar_name,
                    scholar_org=scholar_org,
                    bind_scholar_id=clean_text(resolved_person.get("id")),
                    resolved_person=resolved_person,
                    config=config,
                )
            )
    elif scholar_name and not scholar_papers:
        # Fallback to original approach when dual-bucket not enabled
        resolved_person, hinted_papers = _search_papers_by_author_hint(scholar_name, scholar_org, config=config)
        scholar_papers.extend(hinted_papers)

    scholar_papers = [paper for index, paper in enumerate(scholar_papers) if clean_text(paper.get("title")) and paper not in scholar_papers[:index]]

    if scholar_papers and not dual_bucket_enabled:
        resolved_name = clean_text(resolved_person.get("display_name") or resolved_person.get("name_zh") or resolved_person.get("name") or scholar_name)
        profiles.append(
            _profile_from_authored_papers(
                scholar_papers,
                profile_name=resolved_name or clean_text((scholar_papers[0].get("authors") or [""])[0]) or "scholar_profile",
                source="manual_scholar_profile" if paper_titles or papers_file else "aminer_author_hint_profile",
                explicit_topics=resolved_topics,
                scholar_name=scholar_name,
                scholar_org=scholar_org,
                bind_scholar_id=clean_text(resolved_person.get("id")),
                resolved_person=resolved_person,
                config=config,
            )
        )

    if not scholar_papers and not cleaned_uid:
        topic_inputs = resolved_topics or _infer_topics_from_free_text(free_text, config=config)
        if topic_inputs:
            topic_profile = build_topics_profile(topic_inputs, config=config)
            topic_profile["profile_mode"] = "topic_path"
            topic_profile["source_metadata"] = {
                **dict(topic_profile.get("source_metadata") or {}),
                "source": "free_text_topics" if free_text and not explicit_topics else str((topic_profile.get("source_metadata") or {}).get("source") or "topics_fallback"),
                "free_text": clean_text(free_text),
            }
            profiles.append(topic_profile)

    merged = _merge_profiles(profiles)
    if merged.get("status") == "success" and not merged.get("topics"):
        return {"status": "degraded", "enabled": False, "source_metadata": {"reason": "missing_topics"}}
    return merged


def summarize_profile_request(
    *,
    aminer_user_id: str = "",
    topics: list[str] | None = None,
    scholar_name: str = "",
    scholar_org: str = "",
    paper_titles: list[str] | None = None,
    papers_file: str = "",
    free_text: str = "",
) -> dict[str, Any]:
    cleaned_topics = [clean_text(item) for item in list(topics or []) if clean_text(item)]
    return {
        "aminer_user_id": clean_text(aminer_user_id),
        "topics": cleaned_topics,
        "scholar_name": clean_text(scholar_name),
        "scholar_org": clean_text(scholar_org),
        "paper_titles": [clean_text(item) for item in list(paper_titles or []) if clean_text(item)],
        "papers_file": clean_text(papers_file),
        "free_text": clean_text(free_text),
        "query": _query_from_topics(cleaned_topics),
    }
