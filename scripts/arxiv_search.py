from __future__ import annotations

import math
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

from scripts.common import normalize_arxiv_id, utc_now_iso
from scripts.common import dedupe_preserve_order
from scripts.constants import (
    DEFAULT_ARXIV_API_URL,
    DEFAULT_AMINER_MAP_URL,
    DEFAULT_ARXIV_LOOKBACK_DAYS,
    DEFAULT_ARXIV_MAX_RESULTS,
    DEFAULT_TOP_K,
    DUAL_BUCKET_RECENT_LOOKBACK_DAYS,
    DUAL_BUCKET_RECENT_MAX_PAPERS,
    DUAL_BUCKET_ANCHOR_MAX_PAPERS,
    SOURCE_PRIOR_WEIGHTS,
    DUAL_BUCKET_MATCH_BONUS,
    build_aminer_paper_url,
    build_aminer_paper_search_url,
)

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
RECENT_PAPER_YEAR_WINDOW = 2
LOW_SIGNAL_QUERY_TERMS = {
    "been deployed",
    "deep research",
    "extensive experiments",
    "has been",
    "we present",
}
TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "based",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "the",
    "to",
    "using",
    "via",
    "with",
}
APPLICATION_DOMAIN_DRIFT_TERMS = {
    "biomedical",
    "clinical",
    "crime",
    "drosophila",
    "financial",
    "food",
    "geolocation",
    "humanitarian",
    "medical",
    "patient",
    "patent",
    "payment",
}
QUALITY_STRONG_INNOVATION_TERMS = (
    "state-of-the-art",
    "sota",
    "breakthrough",
    "first",
    "surpass",
    "outperform",
    "pioneering",
)
QUALITY_WEAK_INNOVATION_TERMS = (
    "novel",
    "propose",
    "introduce",
    "new approach",
    "new method",
    "innovative",
)
QUALITY_METHOD_TERMS = (
    "framework",
    "architecture",
    "algorithm",
    "mechanism",
    "pipeline",
    "end-to-end",
)
QUALITY_QUANTITATIVE_TERMS = (
    "outperforms",
    "improves by",
    "achieves",
    "accuracy",
    "f1",
    "bleu",
    "rouge",
    "beats",
    "surpasses",
)
QUALITY_EXPERIMENT_TERMS = (
    "experiment",
    "evaluation",
    "benchmark",
    "ablation",
    "baseline",
    "comparison",
)
TOP_TIER_VENUE_ALIASES: dict[str, tuple[str, ...]] = {
    "NeurIPS": ("neurips", "nips", "advances in neural information processing systems"),
    "ICML": ("icml", "international conference on machine learning"),
    "ICLR": ("iclr", "international conference on learning representations"),
    "AAAI": ("aaai", "aaai conference on artificial intelligence"),
    "IJCAI": ("ijcai", "international joint conference on artificial intelligence"),
    "CVPR": ("cvpr", "conference on computer vision and pattern recognition"),
    "ICCV": ("iccv", "international conference on computer vision"),
    "ECCV": ("eccv", "european conference on computer vision"),
    "ACL": ("acl", "annual meeting of the association for computational linguistics"),
    "EMNLP": ("emnlp", "conference on empirical methods in natural language processing"),
    "NAACL": ("naacl", "north american chapter of the association for computational linguistics"),
    "COLING": ("coling", "international conference on computational linguistics"),
    "KDD": ("kdd", "sigkdd conference on knowledge discovery and data mining"),
    "WWW": ("www", "the web conference", "world wide web conference"),
    "SIGIR": ("sigir", "international acm sigir conference on research and development in information retrieval"),
    "SIGMOD": ("sigmod", "international conference on management of data", "acm sigmod"),
    "VLDB": ("vldb", "very large data bases", "proceedings of the vldb endowment", "pvldb"),
    "ICDE": ("icde", "international conference on data engineering"),
    "MICCAI": ("miccai", "medical image computing and computer assisted intervention"),
    "ICRA": ("icra", "international conference on robotics and automation"),
    "IROS": ("iros", "intelligent robots and systems"),
    "RSS": ("rss", "robotics science and systems"),
    "TPAMI": ("tpami", "transactions on pattern analysis and machine intelligence"),
    "IJCV": ("ijcv", "international journal of computer vision"),
    "JMLR": ("jmlr", "journal of machine learning research"),
    "TACL": ("tacl", "transactions of the association for computational linguistics"),
    "Nature": ("nature",),
    "Science": ("science",),
    "Nature MI": ("nature machine intelligence",),
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _paper_identity(paper: dict[str, Any]) -> str:
    return (
        _clean_text(paper.get("arxiv_id"))
        or _clean_text(paper.get("aminer_paper_id"))
        or _clean_text(paper.get("doi"))
        or _clean_text(paper.get("title")).casefold()
    )


def _normalize_venue_text(value: Any) -> str:
    lowered = _clean_text(value).casefold()
    if not lowered:
        return ""
    return f" {re.sub(r'[^a-z0-9]+', ' ', lowered).strip()} "


def _has_effective_abstract(paper: dict[str, Any]) -> bool:
    title = _clean_text(paper.get("title")).casefold()
    abstract = _clean_text(paper.get("abstract") or paper.get("summary"))
    if not abstract:
        return False
    normalized_abstract = abstract.casefold()
    if title and normalized_abstract == title:
        return False
    if len(abstract) < 80:
        return False
    token_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", abstract))
    return token_count >= 12


def _resolve_paper_year(paper: dict[str, Any]) -> int:
    year_text = _clean_text(paper.get("year"))
    try:
        if year_text:
            return int(year_text)
    except Exception:
        return 0
    published_date = paper.get("published_date")
    if isinstance(published_date, datetime):
        return int(published_date.year)
    return 0


def identify_top_tier_venue(venue: Any) -> str:
    normalized = _normalize_venue_text(venue)
    if not normalized:
        return ""
    for canonical, aliases in TOP_TIER_VENUE_ALIASES.items():
        for alias in aliases:
            alias_normalized = _normalize_venue_text(alias)
            if alias_normalized and alias_normalized in normalized:
                return canonical
    return ""


def is_recent_top_tier_paper(
    paper: dict[str, Any],
    *,
    current_year: int | None = None,
    recent_year_window: int = RECENT_PAPER_YEAR_WINDOW,
) -> bool:
    venue_label = identify_top_tier_venue(paper.get("venue"))
    if not venue_label:
        return False
    resolved_year = _resolve_paper_year(paper)
    if not resolved_year:
        return False
    if current_year is None:
        current_year = datetime.now(timezone.utc).year
    earliest_year = current_year - max(recent_year_window - 1, 0)
    return resolved_year >= earliest_year


def annotate_recent_top_tier_metadata(paper: dict[str, Any]) -> dict[str, Any]:
    venue_label = identify_top_tier_venue(paper.get("venue"))
    resolved_year = _resolve_paper_year(paper)
    return {
        **paper,
        "top_tier_venue": venue_label,
        "paper_year": resolved_year or _clean_text(paper.get("year")),
        "is_recent_top_tier": bool(venue_label and is_recent_top_tier_paper(paper)),
    }


def default_recent_top_tier_quota(top_k: int) -> int:
    normalized_top_k = max(int(top_k or 0), 1)
    return max(1, min(2, normalized_top_k // 2 or 1))


def rebalance_recent_top_tier_papers(
    papers: list[dict[str, Any]],
    *,
    top_k: int,
    min_recent_top_tier: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    annotated = [annotate_recent_top_tier_metadata(paper) for paper in papers]
    normalized_top_k = max(int(top_k or 0), 1)
    selected_indices = list(range(min(len(annotated), normalized_top_k)))
    if min_recent_top_tier is None:
        min_recent_top_tier = default_recent_top_tier_quota(normalized_top_k)
    eligible_indices = [index for index, paper in enumerate(annotated) if paper.get("is_recent_top_tier")]
    required = min(max(int(min_recent_top_tier), 0), normalized_top_k, len(eligible_indices))
    selected_eligible = [index for index in selected_indices if annotated[index].get("is_recent_top_tier")]
    promoted_indices: list[int] = []
    if len(selected_eligible) < required:
        replacement_candidates = [index for index in eligible_indices if index not in selected_indices]
        removable_indices = [index for index in reversed(selected_indices) if not annotated[index].get("is_recent_top_tier")]
        swap_count = min(required - len(selected_eligible), len(replacement_candidates), len(removable_indices))
        if swap_count > 0:
            removed = set(removable_indices[:swap_count])
            promoted_indices = replacement_candidates[:swap_count]
            selected_indices = [index for index in selected_indices if index not in removed]
            selected_indices.extend(promoted_indices)
            selected_indices.sort()
    selected_papers = [annotated[index] for index in selected_indices[:normalized_top_k]]
    return annotated, selected_papers, {
        "enabled": True,
        "top_k": normalized_top_k,
        "min_recent_top_tier": required,
        "candidate_count": len(annotated),
        "eligible_count": len(eligible_indices),
        "selected_recent_top_tier_count": sum(1 for paper in selected_papers if paper.get("is_recent_top_tier")),
        "promoted_count": len(promoted_indices),
        "promoted_titles": [annotated[index].get("title", "") for index in promoted_indices],
    }


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9-]+", text.lower()) if len(token) > 1}


def _term_tokens(term: str) -> set[str]:
    raw = _tokenize(term)
    filtered = {token for token in raw if token not in TOPIC_STOPWORDS}
    return filtered or raw


def _token_matches(text_tokens: set[str], topic_token: str) -> bool:
    if topic_token in text_tokens:
        return True
    singular = topic_token[:-1] if topic_token.endswith("s") else topic_token
    plural = f"{topic_token}s" if not topic_token.endswith("s") else topic_token
    if singular and singular in text_tokens:
        return True
    if plural in text_tokens:
        return True
    if topic_token == "ai" and {"artificial", "intelligence"}.issubset(text_tokens):
        return True
    if topic_token == "llm" and (
        "llm" in text_tokens or "llms" in text_tokens or {"large", "language", "model"}.issubset(text_tokens)
    ):
        return True
    if topic_token == "rl" and ("rl" in text_tokens or {"reinforcement", "learning"}.issubset(text_tokens)):
        return True
    return False


def _term_overlap_count(text_tokens: set[str], term_tokens: set[str]) -> int:
    return sum(1 for token in term_tokens if _token_matches(text_tokens, token))


def _term_min_match_count(term_tokens: set[str]) -> int:
    if len(term_tokens) <= 1:
        return 1
    return 2


def _term_phrase_match(text: str, term: str) -> bool:
    normalized_term = " ".join(term.lower().split())
    if not normalized_term:
        return False
    normalized_text = " ".join(text.lower().split())
    term_tokens = normalized_term.split()
    if not term_tokens:
        return False
    if len(term_tokens) == 1:
        return bool(re.search(rf"\b{re.escape(term_tokens[0])}\b", normalized_text))
    phrase_pattern = r"\b" + r"\s+".join(re.escape(token) for token in term_tokens) + r"\b"
    return bool(re.search(phrase_pattern, normalized_text))


def _escape_arxiv_query(text: str) -> str:
    return text.replace('"', "")


def _normalize_query_term(value: Any) -> str:
    return _clean_text(value)


def _english_query_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9-]+", text.lower())


def _is_low_signal_query_term(text: str) -> bool:
    normalized = _normalize_query_term(text)
    if not normalized:
        return True
    lowered = normalized.casefold()
    if lowered in LOW_SIGNAL_QUERY_TERMS:
        return True
    tokens = _english_query_tokens(normalized)
    if not tokens:
        return True
    if len(tokens) == 1 and tokens[0] in {"paper", "papers", "research", "study", "studies"}:
        return True
    if len(tokens) >= 2 and tokens[0] in {"we", "our", "this", "these"}:
        return True
    return False


def _dedupe_query_terms(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = _normalize_query_term(item)
        key = normalized.casefold()
        if not normalized or key in seen or _is_low_signal_query_term(normalized):
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _profile_llm_topic_keywords(profile: dict[str, Any]) -> list[str]:
    metadata = profile.get("source_metadata") if isinstance(profile.get("source_metadata"), dict) else {}
    internal_profile = metadata.get("internal_profile") if isinstance(metadata.get("internal_profile"), dict) else {}
    llm_topics = internal_profile.get("llm_topics") if isinstance(internal_profile.get("llm_topics"), list) else []
    keywords: list[str] = []
    for topic in llm_topics:
        if not isinstance(topic, dict):
            continue
        name = _normalize_query_term(topic.get("name"))
        if name:
            keywords.append(name)
        for keyword in topic.get("keywords") or []:
            normalized = _normalize_query_term(keyword)
            if normalized:
                keywords.append(normalized)
    return _dedupe_query_terms(keywords)


def _profile_scholar_term_labels(profile: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = profile.get("source_metadata") if isinstance(profile.get("source_metadata"), dict) else {}
    labeling = metadata.get("scholar_term_labeling") if isinstance(metadata.get("scholar_term_labeling"), dict) else {}
    labels = labeling.get("labels") if isinstance(labeling.get("labels"), list) else []
    normalized: list[dict[str, Any]] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        term = _normalize_query_term(item.get("term"))
        role = _clean_text(item.get("role"))
        try:
            weight = float(item.get("weight"))
        except (TypeError, ValueError):
            weight = 0.0
        if not term or not role:
            continue
        normalized.append({"term": term, "role": role, "weight": weight})
    return normalized


def _profile_layered_recall_terms(profile: dict[str, Any]) -> dict[str, Any] | None:
    layered = profile.get("layered_recall_terms")
    if isinstance(layered, dict):
        return layered
    metadata = profile.get("source_metadata") if isinstance(profile.get("source_metadata"), dict) else {}
    scholar_term_labeling = metadata.get("scholar_term_labeling") if isinstance(metadata.get("scholar_term_labeling"), dict) else {}
    nested_layered = scholar_term_labeling.get("layered_recall_terms")
    return nested_layered if isinstance(nested_layered, dict) else None


def _is_meta_keyword(term: str) -> bool:
    normalized = _clean_text(term).casefold()
    if not normalized:
        return True
    meta_terms = {
        "benchmark",
        "benchmarks",
        "contrastive learning",
        "self-supervised learning",
        "extensive experiments",
        "open academic graph",
    }
    if normalized in meta_terms:
        return True
    return any(fragment in normalized for fragment in ("extensive experiments", "benchmark", "self-supervised", "contrastive"))


def _is_generic_domain_keyword(term: str) -> bool:
    normalized = _clean_text(term).casefold()
    if not normalized:
        return True
    return normalized in {
        "named entity recognition",
        "entity linking",
        "concept linking",
        "knowledge graph",
        "knowledge graphs",
        "academic knowledge graph",
        "academic graph mining",
    }


def _is_scholar_specific_keyword(term: str, seed_tokens: set[str]) -> bool:
    normalized = _clean_text(term)
    lowered = normalized.casefold()
    if not normalized or _is_meta_keyword(normalized):
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9-]{1,9}", normalized):
        return True
    if any(fragment in lowered for fragment in ("disambiguation", "ambiguity", "oag", "author name")):
        return True
    term_tokens = _tokenize(normalized)
    if not term_tokens:
        return False
    overlap = len(term_tokens.intersection(seed_tokens))
    return overlap >= 2 and not _is_generic_domain_keyword(normalized)


def _application_domain_drift_penalty(text_tokens: set[str], profile_tokens: set[str]) -> float:
    drift_hits = [token for token in APPLICATION_DOMAIN_DRIFT_TERMS if token in text_tokens and token not in profile_tokens]
    if not drift_hits:
        return 0.0
    return min(1.8, 0.9 + (0.35 * max(len(drift_hits) - 1, 0)))


def _title_domain_drift_penalty(title_tokens: set[str], profile_tokens: set[str]) -> float:
    drift_hits = [token for token in APPLICATION_DOMAIN_DRIFT_TERMS if token in title_tokens and token not in profile_tokens]
    if not drift_hits:
        return 0.0
    return min(3.0, 2.0 + (0.45 * max(len(drift_hits) - 1, 0)))


def _generic_only_scholar_penalty(
    *,
    is_scholar_profile: bool,
    primary_match_count: int,
    high_signal_match_count: int,
    generic_primary_match_count: int,
    text_tokens: set[str],
    profile_tokens: set[str],
) -> float:
    if not is_scholar_profile:
        return 0.0
    if primary_match_count <= 0 or high_signal_match_count > 0 or generic_primary_match_count <= 0:
        return 0.0
    profile_overlap = len(text_tokens.intersection(profile_tokens))
    if profile_overlap >= 3:
        return 0.0
    return min(2.4, 1.2 + (0.45 * max(generic_primary_match_count - 1, 0)))


def _calculate_recency_score(published_date: datetime | None) -> float:
    if published_date is None:
        return 0.0
    days = _days_since(published_date)
    if days <= 30:
        return 3.0
    if days <= 90:
        return 2.0
    if days <= 180:
        return 1.0
    return 0.0


def _calculate_quality_score(text: str) -> float:
    summary = _clean_text(text).casefold()
    if not summary:
        return 0.0
    score = 0.0
    strong_count = sum(1 for term in QUALITY_STRONG_INNOVATION_TERMS if term in summary)
    if strong_count >= 2:
        score += 1.0
    elif strong_count == 1:
        score += 0.7
    else:
        weak_count = sum(1 for term in QUALITY_WEAK_INNOVATION_TERMS if term in summary)
        if weak_count > 0:
            score += 0.3
    if any(term in summary for term in QUALITY_METHOD_TERMS):
        score += 0.5
    if any(term in summary for term in QUALITY_QUANTITATIVE_TERMS):
        score += 0.8
    elif any(term in summary for term in QUALITY_EXPERIMENT_TERMS):
        score += 0.4
    return min(score, 3.0)


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _dedupe_terms_keep_noise(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = _clean_text(item)
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _normalize_research_domains(profile: dict[str, Any]) -> list[dict[str, Any]]:
    raw_domains = profile.get("research_domains")
    if not isinstance(raw_domains, list):
        return []
    cleaned_domains: list[dict[str, Any]] = []
    for item in raw_domains:
        if not isinstance(item, dict):
            continue
        name = _clean_text(item.get("name"))
        if not name:
            continue
        keywords = _dedupe_terms_keep_noise(
            [
                keyword
                for keyword in [
                    name,
                    *list(item.get("keywords") or []),
                ]
                if _clean_text(keyword) and not _is_low_signal_query_term(keyword)
            ]
        )
        exclude_keywords = _dedupe_terms_keep_noise(
            [
                term
                for term in list(item.get("exclude_keywords") or [])
                if _clean_text(term)
            ]
        )
        priority = float(item.get("priority") or 1.0)
        cleaned_domains.append(
            {
                "name": name,
                "keywords": keywords[:8] or [name],
                "exclude_keywords": exclude_keywords[:12],
                "priority": max(priority, 1.0),
            }
        )
    return cleaned_domains


def _flatten_research_domain_terms(research_domains: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    domain_terms: list[str] = []
    keyword_terms: list[str] = []
    for domain in research_domains:
        if not isinstance(domain, dict):
            continue
        domain_name = _clean_text(domain.get("name"))
        if domain_name:
            domain_terms.append(domain_name)
        for keyword in list(domain.get("keywords") or []):
            normalized = _clean_text(keyword)
            if normalized:
                keyword_terms.append(normalized)
    return _dedupe_terms_keep_noise(domain_terms), _dedupe_terms_keep_noise(keyword_terms)


def _research_domain_profile_tokens(research_domains: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for domain in research_domains:
        if not isinstance(domain, dict):
            continue
        tokens.update(_tokenize(_clean_text(domain.get("name"))))
        for keyword in list(domain.get("keywords") or []):
            tokens.update(_tokenize(_clean_text(keyword)))
    return tokens


def _score_research_domain_match(
    paper: dict[str, Any],
    research_domains: list[dict[str, Any]],
    excluded_keywords: list[str],
) -> tuple[float, list[str], int, int]:
    title = _clean_text(paper.get("title"))
    abstract = _clean_text(paper.get("abstract") or paper.get("summary"))
    paper_text = f"{title}\n{abstract}"
    text_tokens = _tokenize(paper_text)
    title_tokens = _tokenize(title)
    matched_terms: list[str] = []
    matched_seen: set[str] = set()
    primary_match_count = 0
    high_signal_match_count = 0
    score = 0.0
    exclude_set = {_clean_text(item).casefold() for item in excluded_keywords if _clean_text(item)}

    def _record_match(term: str) -> None:
        normalized = _clean_text(term)
        if not normalized:
            return
        key = normalized.casefold()
        if key in matched_seen:
            return
        matched_seen.add(key)
        matched_terms.append(normalized)

    for domain in research_domains:
        if not isinstance(domain, dict):
            continue
        domain_name = _clean_text(domain.get("name"))
        if not domain_name:
            continue
        priority = max(float(domain.get("priority") or 1.0), 1.0)
        domain_weight = min(priority, 8.0) / 4.0
        domain_tokens = _tokenize(domain_name)
        domain_matched = 0
        exact_domain_title = _term_phrase_match(title, domain_name)
        exact_domain_abstract = _term_phrase_match(abstract, domain_name)
        if exact_domain_title:
            gain = 2.4 * domain_weight
            score += gain
            _record_match(domain_name)
            primary_match_count += 1
            high_signal_match_count += 1
            domain_matched += 1
        elif exact_domain_abstract:
            gain = 1.6 * domain_weight
            score += gain
            _record_match(domain_name)
            primary_match_count += 1
            high_signal_match_count += 1
            domain_matched += 1
        elif _term_overlap_count(text_tokens, domain_tokens) >= _term_min_match_count(domain_tokens):
            gain = 0.55 * domain_weight
            score += gain
            _record_match(domain_name)
            primary_match_count += 1
            domain_matched += 1

        keyword_matches = 0
        for keyword in list(domain.get("keywords") or []):
            normalized_keyword = _clean_text(keyword)
            if not normalized_keyword or normalized_keyword.casefold() == domain_name.casefold():
                continue
            if normalized_keyword.casefold() in exclude_set:
                continue
            keyword_tokens = _tokenize(normalized_keyword)
            exact_title = _term_phrase_match(title, normalized_keyword)
            exact_abstract = _term_phrase_match(abstract, normalized_keyword)
            if exact_title:
                gain = 1.35 * domain_weight
                score += gain
                _record_match(normalized_keyword)
                primary_match_count += 1
                high_signal_match_count += 1
                keyword_matches += 1
            elif exact_abstract:
                gain = 0.95 * domain_weight
                score += gain
                _record_match(normalized_keyword)
                primary_match_count += 1
                high_signal_match_count += 1
                keyword_matches += 1
            elif _term_overlap_count(text_tokens, keyword_tokens) >= _term_min_match_count(keyword_tokens):
                gain = 0.32 * domain_weight
                score += gain
                _record_match(normalized_keyword)
                primary_match_count += 1
                keyword_matches += 1

        exclude_hits = 0
        for term in list(domain.get("exclude_keywords") or []):
            normalized_term = _clean_text(term)
            if not normalized_term:
                continue
            if normalized_term.casefold() in exclude_set or _term_phrase_match(paper_text, normalized_term):
                score -= 1.25 * domain_weight
                exclude_hits += 1
        if keyword_matches > 1:
            score += min(0.8, 0.18 * (keyword_matches - 1))
        if domain_matched and keyword_matches == 0:
            score += 0.15 * domain_weight

    if primary_match_count <= 0:
        score -= 0.9
    return max(score, 0.0), matched_terms[:8], primary_match_count, high_signal_match_count


def _popularity_score(paper: dict[str, Any]) -> float:
    citations = _safe_int(paper.get("citations") or paper.get("n_citation"))
    if citations <= 0:
        return 0.0
    return min(math.log1p(citations) / 4.0, 3.0)


def _filter_and_score_papers(
    papers: list[dict[str, Any]],
    profile: dict[str, Any],
    research_domains: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    excluded_keywords = _dedupe_terms_keep_noise(
        [
            *list(profile.get("excluded_keywords") or []),
            *[
                term
                for domain in research_domains
                for term in list(domain.get("exclude_keywords") or [])
            ],
        ]
    )
    profile_text_tokens = _research_domain_profile_tokens(research_domains)
    category_set = {str(item).strip() for item in profile.get("arxiv_categories") or [] if str(item).strip()}
    preferred_authors = {str(item).casefold() for item in profile.get("preferred_authors") or [] if str(item).strip()}
    preferred_venues = {str(item).casefold() for item in profile.get("preferred_venues") or [] if str(item).strip()}
    seed_tokens = _seed_terms(profile)

    ranked: list[dict[str, Any]] = []
    for paper in papers:
        title = _clean_text(paper.get("title"))
        abstract = _clean_text(paper.get("abstract") or paper.get("summary"))
        paper_text = f"{title}\n{abstract}"
        title_tokens = _tokenize(title)
        text_tokens = _tokenize(paper_text)
        has_effective_abstract = _has_effective_abstract(paper)
        relevance_raw, matched_keywords, primary_match_count, high_signal_match_count = _score_research_domain_match(
            paper,
            research_domains,
            excluded_keywords,
        )
        matched_categories = [category for category in paper.get("categories") or [] if category in category_set]
        relevance_raw += min(len(matched_categories) * 0.35, 1.0)
        matched_authors = [
            author
            for author in paper.get("authors") or []
            if str(author).casefold() in preferred_authors
        ]
        relevance_raw += len(matched_authors) * 0.8
        venue = _clean_text(paper.get("venue"))
        if venue and venue.casefold() in preferred_venues:
            relevance_raw += 0.45
        seed_overlap = len(text_tokens.intersection(seed_tokens))
        if seed_overlap:
            relevance_raw += min(0.8, 0.12 * seed_overlap)
        relevance_raw -= _application_domain_drift_penalty(text_tokens, profile_text_tokens)
        relevance_raw -= _title_domain_drift_penalty(title_tokens, profile_text_tokens)
        relevance_raw = max(relevance_raw, 0.0)

        recency_score = _calculate_recency_score(paper.get("published_date"))
        popularity_score = _popularity_score(paper)
        quality_score = _calculate_quality_score(abstract)
        abstract_score = 3.0 if has_effective_abstract else 0.0

        normalized_relevance = min(relevance_raw, 8.0) / 8.0 * 10.0
        normalized_recency = recency_score / 3.0 * 10.0
        normalized_popularity = popularity_score / 3.0 * 10.0
        normalized_quality = quality_score / 3.0 * 10.0
        normalized_abstract = abstract_score / 3.0 * 10.0
        score = (
            normalized_relevance * 0.50
            + normalized_recency * 0.20
            + normalized_popularity * 0.15
            + normalized_quality * 0.10
            + normalized_abstract * 0.05
        )
        if high_signal_match_count > 0:
            score += min(1.2, 0.35 * high_signal_match_count)
        if primary_match_count <= 0:
            score -= 0.8

        ranked.append(
            {
                **paper,
                "has_effective_abstract": has_effective_abstract,
                "matched_keywords": matched_keywords[:8],
                "primary_match_count": primary_match_count,
                "high_signal_match_count": high_signal_match_count,
                "matched_categories": matched_categories,
                "matched_authors": matched_authors,
                "relevance_score": round(normalized_relevance, 2),
                "recency_score": round(normalized_recency, 2),
                "popularity_score": round(normalized_popularity, 2),
                "quality_score": round(normalized_quality, 2),
                "abstract_score": round(normalized_abstract, 2),
                "recommendation_score": round(score, 2),
                "aminer_comment": "；".join(
                    part
                    for part in (
                        f"命中关键词: {', '.join(matched_keywords[:4])}" if matched_keywords else "",
                        f"命中分类: {', '.join(matched_categories[:3])}" if matched_categories else "",
                        f"命中作者: {', '.join(matched_authors[:3])}" if matched_authors else "",
                    )
                    if part
                ),
                "author_entries": list(paper.get("author_entries") or [{"display_name": author, "profile_url": "", "is_disambiguated": False} for author in (paper.get("authors") or [])]),
                "aminer_author_profiles": list(paper.get("aminer_author_profiles") or []),
                "aminer_paper_url": _clean_text(paper.get("aminer_paper_url")) or build_aminer_paper_url_for_arxiv_paper(paper),
                "famous_authors": list(paper.get("famous_authors") or matched_authors[:3]),
            }
        )

    ranked.sort(
        key=lambda item: (
            float(item.get("recommendation_score") or 0.0),
            item.get("high_signal_match_count", 0),
            item.get("primary_match_count", 0),
            -_days_since(item.get("published_date")),
            item.get("title", ""),
        ),
        reverse=True,
    )
    return ranked[: max(int(top_k), 1)]


def _calculate_domain_relevance_score(
    paper: dict[str, Any],
    research_domains: list[dict[str, Any]],
    excluded_keywords: list[str],
) -> tuple[float, str, list[str]]:
    title = _clean_text(paper.get("title")).casefold()
    summary = _clean_text(paper.get("abstract") or paper.get("summary")).casefold()
    categories = {str(item).strip() for item in paper.get("categories") or [] if str(item).strip()}
    for keyword in excluded_keywords:
        normalized = _clean_text(keyword).casefold()
        if normalized and (normalized in title or normalized in summary):
            return 0.0, "", []
    best_score = 0.0
    best_domain = ""
    best_matches: list[str] = []
    for domain in research_domains:
        if not isinstance(domain, dict):
            continue
        domain_name = _clean_text(domain.get("name"))
        score = 0.0
        matches: list[str] = []
        for keyword in list(domain.get("keywords") or []):
            normalized = _clean_text(keyword)
            lowered = normalized.casefold()
            if not lowered:
                continue
            if lowered in title:
                score += 0.8
                matches.append(normalized)
            elif lowered in summary:
                score += 0.45
                matches.append(normalized)
        for category in list(domain.get("arxiv_categories") or []):
            if str(category).strip() in categories:
                score += 1.0
                break
        score += min(float(domain.get("priority") or 0), 5.0) * 0.08
        if score > best_score:
            best_score = score
            best_domain = domain_name
            best_matches = matches
    return min(best_score, 3.0), best_domain, dedupe_preserve_order(best_matches)[:6]


def _is_short_acronym_query_term(term: str) -> bool:
    """判断是否为短 acronym（1-3 个纯大写字母），这类词不应单独作为查询词"""
    normalized = _normalize_query_term(term)
    if not normalized:
        return False
    # 匹配 1-3 个纯大写字母的 acronym，如 NER, KG, IR, LLM, RAG
    return bool(re.fullmatch(r"[A-Z]{1,3}", normalized))


def _build_query_term_plans(profile: dict[str, Any], *, top_k: int = DEFAULT_TOP_K) -> list[list[str]]:
    """
    构建分层查询计划，支持两阶段召回：
    1. Primary bundle: scholar_specific + core_domain（高特异性术语优先召回）
    2. Fallback bundle: broad_superordinate + auxiliary（仅在 primary 不足时补充）

    每个查询计划携带 plan_role 标记，用于后续排序加权。
    """
    term_weights = {
        _normalize_query_term(key).casefold(): float(value)
        for key, value in dict(profile.get("retrieval_term_weights") or {}).items()
        if _normalize_query_term(key)
    }
    retrieval_topics = list(profile.get("retrieval_topics") or profile.get("topics") or [])
    retrieval_keywords = list(profile.get("retrieval_keywords") or profile.get("keywords") or [])

    # 检查是否有预计算的分层召回术语
    layered_terms = _profile_layered_recall_terms(profile)

    if str(profile.get("profile_mode") or "") == "scholar_path":
        primary_terms: list[str] = []
        fallback_terms: list[str] = []

        # 优先使用预计算的分层术语
        if layered_terms:
            primary_terms = list(layered_terms.get("primary_recall_terms") or [])
            fallback_terms = list(layered_terms.get("fallback_recall_terms") or [])
            blocked_terms = set(
                _normalize_query_term(term).casefold()
                for term in (layered_terms.get("blocked_query_terms") or [])
            )

            # 过滤掉 blocked 术语
            primary_terms = [t for t in primary_terms if _normalize_query_term(t).casefold() not in blocked_terms]
            fallback_terms = [t for t in fallback_terms if _normalize_query_term(t).casefold() not in blocked_terms]

            # 过滤短 acronym（不单独作为查询词）
            primary_terms = [t for t in primary_terms if not _is_short_acronym_query_term(t)]
            fallback_terms = [t for t in fallback_terms if not _is_short_acronym_query_term(t)]
        else:
            # 兼容旧逻辑：从 LLM labels 推导
            llm_labels = _profile_scholar_term_labels(profile)
            if llm_labels:
                allowed_roles = {"scholar_specific", "core_domain", "broad_superordinate"}
                role_priority = {"scholar_specific": 3, "core_domain": 2, "broad_superordinate": 1}
                label_map = {
                    _normalize_query_term(item.get("term")).casefold(): item
                    for item in llm_labels
                    if _normalize_query_term(item.get("term")) and str(item.get("role") or "") in allowed_roles
                }
                labeled_terms = _dedupe_query_terms([*retrieval_topics, *retrieval_keywords])
                labeled_terms = [term for term in labeled_terms if not _is_short_acronym_query_term(term)]

                def _term_rank(item: str) -> tuple[float, float, float]:
                    label = label_map.get(_normalize_query_term(item).casefold(), {})
                    return (
                        float(role_priority.get(str(label.get("role") or ""), 0)),
                        float(label.get("weight") or 0.0),
                        float(term_weights.get(_normalize_query_term(item).casefold(), 0.0)),
                    )

                primary_terms = [
                    term
                    for term in sorted(labeled_terms, key=_term_rank, reverse=True)
                    if str((label_map.get(_normalize_query_term(term).casefold()) or {}).get("role") or "") in {"scholar_specific", "core_domain"}
                ]
                fallback_terms = [
                    term
                    for term in sorted(labeled_terms, key=_term_rank, reverse=True)
                    if str((label_map.get(_normalize_query_term(term).casefold()) or {}).get("role") or "") == "broad_superordinate"
                ]
            else:
                # 无 LLM labels，使用权重排序
                ordered_terms = _dedupe_query_terms([*retrieval_keywords, *retrieval_topics])
                ordered_terms = [t for t in ordered_terms if not _is_short_acronym_query_term(t)]
                ordered_terms = sorted(
                    ordered_terms,
                    key=lambda item: (
                        term_weights.get(_normalize_query_term(item).casefold(), 0.0),
                        len(_normalize_query_term(item)),
                    ),
                    reverse=True,
                )
                primary_terms = ordered_terms[:5]
                fallback_terms = ordered_terms[5:10]

        # 构建查询计划：primary bundle 优先，fallback 只做补召回
        plans: list[list[str]] = []
        if primary_terms:
            primary_bundle_size = 3
            plans.append(primary_terms[:primary_bundle_size])
            remainder = primary_terms[primary_bundle_size : primary_bundle_size + 2]
            if remainder:
                plans.append(remainder)
        if fallback_terms:
            plans.append(fallback_terms[:2])
        if plans:
            return plans

    # 非 scholar_path 或无关键词时，使用通用逻辑
    broad_terms = _dedupe_query_terms(
        [
            *_profile_llm_topic_keywords(profile),
            *retrieval_topics,
        ]
    )
    specific_terms = _dedupe_query_terms(retrieval_keywords)
    broad_terms = sorted(
        broad_terms,
        key=lambda item: (
            term_weights.get(_normalize_query_term(item).casefold(), 0.0),
            len(_normalize_query_term(item)),
        ),
        reverse=True,
    )
    specific_terms = sorted(
        specific_terms,
        key=lambda item: (
            term_weights.get(_normalize_query_term(item).casefold(), 0.0),
            len(_normalize_query_term(item)),
        ),
        reverse=True,
    )
    primary_terms = _dedupe_query_terms([*broad_terms[:4], *specific_terms[:6]])[:8]
    fallback_terms = _dedupe_query_terms([*specific_terms[:10], *broad_terms[:8]])[:12]
    plans: list[list[str]] = []
    for candidate in (primary_terms, fallback_terms, []):
        if candidate in plans:
            continue
        if candidate or not plans:
            plans.append(candidate)
    if top_k > 6 and fallback_terms and fallback_terms not in plans:
        plans.append(fallback_terms)
    return plans


def _build_query_plans_with_role(profile: dict[str, Any], *, top_k: int = DEFAULT_TOP_K) -> list[tuple[list[str], str]]:
    research_domains = _normalize_research_domains(profile)
    if str(profile.get("profile_mode") or "") == "scholar_path" and research_domains:
        excluded = {
            _normalize_query_term(term).casefold()
            for term in list(profile.get("excluded_keywords") or [])
            if _normalize_query_term(term)
        }
        domain_names = [
            _clean_text(domain.get("name"))
            for domain in research_domains
            if _clean_text(domain.get("name")) and not _is_short_acronym_query_term(_clean_text(domain.get("name")))
        ]
        fallback_keywords: list[str] = []
        for domain in research_domains:
            if not isinstance(domain, dict):
                continue
            for keyword in list(domain.get("keywords") or []):
                normalized = _normalize_query_term(keyword)
                if not normalized:
                    continue
                if normalized.casefold() in excluded:
                    continue
                if _is_short_acronym_query_term(normalized):
                    continue
                if normalized.casefold() in {item.casefold() for item in domain_names}:
                    continue
                if _is_generic_domain_keyword(normalized):
                    continue
                fallback_keywords.append(normalized)
        domain_names = dedupe_preserve_order(domain_names)[:4]
        fallback_keywords = dedupe_preserve_order(fallback_keywords)[:4]
        plans: list[tuple[list[str], str]] = []
        if domain_names[:2]:
            plans.append((domain_names[:2], "primary"))
        if domain_names[2:4]:
            plans.append((domain_names[2:4], "primary"))
        if fallback_keywords:
            plans.append((fallback_keywords[:2], "fallback"))
        if plans:
            return plans

    layered_terms = _profile_layered_recall_terms(profile)
    if str(profile.get("profile_mode") or "") == "scholar_path" and layered_terms:
        primary_terms = [
            term
            for term in list(layered_terms.get("primary_recall_terms") or [])
            if term and not _is_short_acronym_query_term(term)
        ]
        fallback_terms = [
            term
            for term in list(layered_terms.get("fallback_recall_terms") or [])
            if term and not _is_short_acronym_query_term(term)
        ]
        plans: list[tuple[list[str], str]] = []
        if primary_terms:
            primary_bundle_size = 3
            first = primary_terms[:primary_bundle_size]
            if first:
                plans.append((first, "primary"))
            remainder = primary_terms[primary_bundle_size : primary_bundle_size + 2]
            if remainder:
                plans.append((remainder, "primary"))
        if fallback_terms:
            plans.append((fallback_terms[:2], "fallback"))
        if plans:
            return plans
    return [(plan, "primary") for plan in _build_query_term_plans(profile, top_k=top_k) if plan]


def _build_dual_bucket_query_plans(
    profile: dict[str, Any],
    *,
    top_k: int = DEFAULT_TOP_K
) -> list[tuple[list[str], str, float]]:
    """Build three-layer recall plans for dual-bucket profile.

    Returns:
        List of (query_terms, plan_role, source_prior):
        - primary: Use primary_recall_terms (source_prior=1.2)
        - anchor: Use anchor_recall_terms (source_prior=0.8)
        - recent: Use recent_recall_terms (source_prior=0.4)
    """
    dual_bucket_terms = profile.get("dual_bucket_layered_recall_terms") if isinstance(profile.get("dual_bucket_layered_recall_terms"), dict) else {}
    if not dual_bucket_terms:
        return []

    plans: list[tuple[list[str], str, float]] = []

    # Primary layer: intersection of anchor and recent
    primary_terms = [
        term
        for term in list(dual_bucket_terms.get("primary_recall_terms") or [])
        if term and not _is_short_acronym_query_term(term)
    ]
    if primary_terms:
        plans.append((primary_terms[:3], "primary", SOURCE_PRIOR_WEIGHTS["primary"]))

    # Anchor layer: long-term identity
    anchor_terms = [
        term
        for term in list(dual_bucket_terms.get("anchor_recall_terms") or [])
        if term and not _is_short_acronym_query_term(term)
    ]
    if anchor_terms:
        plans.append((anchor_terms[:2], "anchor", SOURCE_PRIOR_WEIGHTS["anchor"]))

    # Recent layer: new directions (limited to prevent dominance)
    recent_terms = [
        term
        for term in list(dual_bucket_terms.get("recent_recall_terms") or [])
        if term and not _is_short_acronym_query_term(term)
    ]
    if recent_terms:
        # Limit recent terms to prevent recent dominance
        plans.append((recent_terms[:2], "recent", SOURCE_PRIOR_WEIGHTS["recent"]))

    # Constraint: if no primary and no anchor, limit recent
    if not any(p[1] in ("primary", "anchor") for p in plans):
        # Filter out recent plans if there's no primary/anchor
        plans = [(terms, role, prior) for terms, role, prior in plans if role != "recent"]
        # Add limited recent if still empty
        if not plans and recent_terms:
            plans.append((recent_terms[:2], "recent", SOURCE_PRIOR_WEIGHTS["recent"] * 0.5))

    return plans


def build_arxiv_query(categories: list[str], keywords: list[str], lookback_days: int, *, max_keyword_clauses: int = 10) -> str:
    category_query = " OR ".join(f"cat:{category}" for category in categories if _clean_text(category))
    keyword_clauses = [
        f'all:"{_escape_arxiv_query(keyword)}"'
        for keyword in keywords[: max(int(max_keyword_clauses), 1)]
        if _clean_text(keyword)
    ]
    keyword_query = " OR ".join(keyword_clauses)
    submitted_after = (datetime.now(timezone.utc) - timedelta(days=max(lookback_days, 1))).strftime("%Y%m%d")
    date_query = f"submittedDate:[{submitted_after}0000 TO 300001012359]"
    if category_query and keyword_query:
        return f"({category_query}) AND ({keyword_query}) AND {date_query}"
    if category_query:
        return f"({category_query}) AND {date_query}"
    if keyword_query:
        return f"({keyword_query}) AND {date_query}"
    return date_query


def _parse_datetime(value: str) -> datetime | None:
    text = _clean_text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def fetch_arxiv_author_papers(
    author_name: str,
    *,
    lookback_days: int = 1095,
    max_results: int = 50,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search for recent papers by a specific author via arXiv API.

    Uses au: prefix for author search combined with time range filter.

    Args:
        author_name: Author name to search for
        lookback_days: How many days to look back (default 1095 = 3 years)
        max_results: Maximum number of results to return
        config: Optional configuration

    Returns:
        List of paper dictionaries from arXiv
    """
    config = config or {}
    cleaned_name = _clean_text(author_name)
    if not cleaned_name:
        return []

    # Escape special characters in author name
    escaped_name = _escape_arxiv_query(cleaned_name)

    # Build author query: au:"First Last" OR au:"Last, First"
    name_parts = escaped_name.split()
    if len(name_parts) >= 2:
        # Try both name orderings
        author_query = f'(au:"{escaped_name}" OR au:"{name_parts[-1]}, {" ".join(name_parts[:-1])}")'
    else:
        author_query = f'au:"{escaped_name}"'

    # Add time range filter
    submitted_after = (datetime.now(timezone.utc) - timedelta(days=max(lookback_days, 1))).strftime("%Y%m%d")
    date_query = f"submittedDate:[{submitted_after}0000 TO 30000101235959]"
    query = f"({author_query}) AND {date_query}"

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max(int(max_results), 1),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    url = f"{DEFAULT_ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    xml_content = ""

    # Try both HTTPS and HTTP
    for candidate_url in [url, url.replace("https://", "http://", 1)]:
        try:
            with opener.open(candidate_url, timeout=60) as response:  # nosec B310
                xml_content = response.read().decode("utf-8")
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            continue

    if not xml_content:
        return []

    papers = parse_arxiv_xml(xml_content)

    # Filter papers to ensure author name matches
    name_lower = cleaned_name.lower()
    name_tokens = set(name_lower.split())

    def author_matches(paper: dict[str, Any]) -> bool:
        for author in paper.get("authors", []):
            author_lower = author.lower()
            if name_lower in author_lower or author_lower in name_lower:
                return True
            author_tokens = set(author_lower.replace(",", " ").split())
            if name_tokens and name_tokens.issubset(author_tokens):
                return True
        return False

    filtered_papers = [paper for paper in papers if author_matches(paper)]
    return filtered_papers


def parse_arxiv_xml(xml_content: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_content)
    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ARXIV_NS):
        paper_id = _clean_text(entry.findtext("atom:id", default="", namespaces=ARXIV_NS))
        title = _clean_text(entry.findtext("atom:title", default="", namespaces=ARXIV_NS))
        summary = _clean_text(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS))
        published = _clean_text(entry.findtext("atom:published", default="", namespaces=ARXIV_NS))
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ARXIV_NS))
            for author in entry.findall("atom:author", ARXIV_NS)
            if _clean_text(author.findtext("atom:name", default="", namespaces=ARXIV_NS))
        ]
        categories = [category.attrib.get("term", "").strip() for category in entry.findall("atom:category", ARXIV_NS) if category.attrib.get("term")]
        pdf_url = ""
        for link in entry.findall("atom:link", ARXIV_NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = _clean_text(link.attrib.get("href"))
                break
        arxiv_id = paper_id.rsplit("/", 1)[-1]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": summary,
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "published": published,
                "published_date": _parse_datetime(published),
                "abs_url": paper_id,
                "pdf_url": pdf_url,
            }
        )
    return papers


def fetch_arxiv_candidates(profile: dict[str, Any], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Three-layer recall strategy:
    1. Primary layer: Use primary_recall_terms (intersection of anchor and recent) - highest priority
    2. Anchor layer: Use anchor_recall_terms (long-term identity) - medium priority
    3. Recent layer: Use recent_recall_terms (new directions) - lowest priority

    Each paper is tagged with plan_role and source_prior for ranking.
    """
    config = config or {}
    search_config = config.get("search") if isinstance(config.get("search"), dict) else {}
    lookback_days = int(search_config.get("lookback_days") or DEFAULT_ARXIV_LOOKBACK_DAYS)
    max_results = int(search_config.get("max_results") or DEFAULT_ARXIV_MAX_RESULTS)
    top_k = int(search_config.get("top_k") or profile.get("top_k") or DEFAULT_TOP_K)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error: Exception | None = None
    categories = list(profile.get("arxiv_categories") or [])
    recall_strategy = profile.get("recall_strategy") if isinstance(profile.get("recall_strategy"), dict) else {}
    queries_tried: list[str] = []
    best_query = ""
    best_papers: list[dict[str, Any]] = []
    merged_papers: dict[str, dict[str, Any]] = {}

    # Check for dual-bucket mode
    dual_bucket_terms = profile.get("dual_bucket_layered_recall_terms") if isinstance(profile.get("dual_bucket_layered_recall_terms"), dict) else {}
    use_dual_bucket = bool(dual_bucket_terms)

    if use_dual_bucket:
        # Use three-layer dual-bucket plans
        query_plans_with_role = _build_dual_bucket_query_plans(profile, top_k=top_k)
    else:
        # Use original two-layer plans
        query_plans_with_role = _build_query_plans_with_role(profile, top_k=top_k)

    minimum_scholar_candidates = max(top_k * 2, 8)
    primary_paper_count = 0
    anchor_paper_count = 0
    fallback_triggered = False

    for plan_index, plan_item in enumerate(query_plans_with_role):
        # Handle both 2-tuple (old) and 3-tuple (dual-bucket) formats
        if use_dual_bucket:
            query_terms, plan_role, source_prior = plan_item
        else:
            query_terms, plan_role = plan_item
            source_prior = 1.0 if plan_role == "primary" else 0.6

        if not query_terms:
            continue

        # Skip fallback/recent if we have enough primary candidates (scholar_path mode)
        if str(profile.get("profile_mode") or "") == "scholar_path":
            if plan_role in ("fallback", "recent") and primary_paper_count >= minimum_scholar_candidates:
                continue
            # Skip recent if we have enough anchor candidates
            if plan_role == "recent" and (primary_paper_count + anchor_paper_count) >= minimum_scholar_candidates:
                continue

        if plan_role in ("fallback", "recent"):
            fallback_triggered = True

        query = build_arxiv_query(categories, query_terms, lookback_days)
        queries_tried.append(query)
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        primary_url = f"{DEFAULT_ARXIV_API_URL}?{urllib.parse.urlencode(params)}"
        fallback_url = primary_url.replace("https://export.arxiv.org", "http://export.arxiv.org", 1)
        xml_content = ""
        for candidate_url in [primary_url, fallback_url]:
            try:
                with opener.open(candidate_url, timeout=60) as response:  # nosec B310
                    xml_content = response.read().decode("utf-8")
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                continue
        if not xml_content:
            continue
        papers = parse_arxiv_xml(xml_content)
        for paper in papers:
            identity = _paper_identity(paper)
            if identity and identity not in merged_papers:
                paper["plan_role"] = plan_role
                paper["source_prior"] = source_prior
                paper["matched_buckets"] = [plan_role]
                merged_papers[identity] = paper
                if plan_role == "primary":
                    primary_paper_count += 1
                elif plan_role == "anchor":
                    anchor_paper_count += 1
            elif identity and identity in merged_papers:
                # Paper already exists - check for dual-bucket match bonus
                existing_paper = merged_papers[identity]
                existing_buckets = existing_paper.get("matched_buckets", [existing_paper.get("plan_role")])
                if plan_role not in existing_buckets:
                    existing_paper["matched_buckets"] = existing_buckets + [plan_role]
                    # Apply dual match bonus
                    if len(existing_paper["matched_buckets"]) >= 2:
                        existing_paper["dual_bucket_match"] = True
                        existing_paper["source_prior"] = existing_paper.get("source_prior", 1.0) + DUAL_BUCKET_MATCH_BONUS

        if len(papers) > len(best_papers):
            best_papers = papers
            best_query = query

        # Scholar path: break early if we have enough primary candidates
        if str(profile.get("profile_mode") or "") == "scholar_path":
            if plan_role == "primary" and primary_paper_count >= minimum_scholar_candidates:
                break
            if plan_role == "anchor" and (primary_paper_count + anchor_paper_count) >= minimum_scholar_candidates:
                # Continue to check for recent layer but with limited priority
                pass

    merged_list = sorted(
        merged_papers.values(),
        key=lambda item: (
            # Higher source_prior first
            -float(item.get("source_prior") or 0.0),
            # Dual bucket match bonus
            -1.0 if item.get("dual_bucket_match") else 0.0,
            _days_since(item.get("published_date")),
            item.get("title", ""),
        ),
    )
    if merged_list:
        best_papers = merged_list[:max_results]
    if not best_papers:
        raise RuntimeError(f"arxiv_unreachable:{last_error}")

    # Statistics for recall layers
    primary_count = sum(1 for p in best_papers if p.get("plan_role") == "primary")
    anchor_count = sum(1 for p in best_papers if p.get("plan_role") == "anchor")
    recent_count = sum(1 for p in best_papers if p.get("plan_role") == "recent")
    fallback_count = sum(1 for p in best_papers if p.get("plan_role") == "fallback")
    dual_match_count = sum(1 for p in best_papers if p.get("dual_bucket_match"))

    return {
        "status": "success",
        "generated_at": utc_now_iso(),
        "query": best_query,
        "queries_tried": queries_tried,
        "candidate_count": len(best_papers),
        "papers": best_papers,
        "recall_role": str(recall_strategy.get("arxiv_role") or ("primary" if categories else "supplemental")),
        "recall_stats": {
            "primary_paper_count": primary_count,
            "anchor_paper_count": anchor_count if use_dual_bucket else 0,
            "recent_paper_count": recent_count if use_dual_bucket else 0,
            "fallback_paper_count": fallback_count,
            "fallback_triggered": fallback_triggered,
            "dual_bucket_match_count": dual_match_count if use_dual_bucket else 0,
        },
    }


def _days_since(published_date: datetime | None) -> int:
    if isinstance(published_date, str):
        published_date = _parse_datetime(published_date)
    if published_date is None:
        return 3650
    now = datetime.now(timezone.utc)
    normalized = published_date.astimezone(timezone.utc) if published_date.tzinfo else published_date.replace(tzinfo=timezone.utc)
    return max((now - normalized).days, 0)


def _seed_terms(profile: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for seed_paper in profile.get("seed_papers") or []:
        if not isinstance(seed_paper, dict):
            continue
        text = " ".join(
            [
                _clean_text(seed_paper.get("title")),
                " ".join(seed_paper.get("keywords") or []),
                _clean_text(seed_paper.get("abstract")),
            ]
        )
        tokens.update(_tokenize(text))
    return tokens


def build_aminer_paper_url_for_arxiv_paper(paper: dict[str, Any]) -> str:
    explicit = _clean_text(paper.get("aminer_paper_url"))
    if explicit:
        return explicit
    paper_id = _clean_text(paper.get("aminer_paper_id"))
    if paper_id:
        return build_aminer_paper_url(paper_id)
    arxiv_id = _clean_text(paper.get("arxiv_id"))
    title = _clean_text(paper.get("title"))
    query = title or arxiv_id
    return build_aminer_paper_search_url(query)


def _resolve_aminer_token(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    return _clean_text(aminer_config.get("token") or os.getenv("AMINER_TOKEN"))


def _resolve_aminer_map_url(config: dict[str, Any] | None = None) -> str:
    config = config or {}
    aminer_config = config.get("aminer") if isinstance(config.get("aminer"), dict) else {}
    return _clean_text(aminer_config.get("map_url")) or DEFAULT_AMINER_MAP_URL


def _extract_mapping_items(response: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(response, dict):
        return []
    for key in ("data", "items"):
        value = response.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def map_arxiv_ids_to_aminer_ids(arxiv_ids: list[str], *, config: dict[str, Any] | None = None) -> dict[str, str]:
    token = _resolve_aminer_token(config)
    if not token:
        return {}
    normalized_ids = [normalize_arxiv_id(arxiv_id) for arxiv_id in arxiv_ids if normalize_arxiv_id(arxiv_id)]
    if not normalized_ids:
        return {}
    body = json.dumps({"arxiv_ids": normalized_ids, "need_details": False}).encode("utf-8")
    request = urllib.request.Request(
        _resolve_aminer_map_url(config),
        data=body,
        headers={
            "Content-Type": "application/json;charset=utf-8",
            "Authorization": token,
            "User-Agent": "aminer-rec/1.0",
        },
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=20) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}
    items = _extract_mapping_items(payload)
    mapping: dict[str, str] = {}
    for index, item in enumerate(items):
        arxiv_id = normalize_arxiv_id(str(item.get("arxiv_id") or item.get("arxivId") or ""))
        if not arxiv_id and index < len(normalized_ids):
            arxiv_id = normalized_ids[index]
        aminer_id = _clean_text(item.get("id") or item.get("aminer_id"))
        if arxiv_id and aminer_id and arxiv_id not in mapping:
            mapping[arxiv_id] = aminer_id
    return mapping


def enrich_ranked_payload_with_aminer_paper_urls(
    ranked_payload: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    papers = list(ranked_payload.get("papers") or [])
    mapping = map_arxiv_ids_to_aminer_ids([str(paper.get("arxiv_id") or "") for paper in papers], config=config)
    if not mapping:
        return ranked_payload
    enriched_papers: list[dict[str, Any]] = []
    for paper in papers:
        arxiv_id = normalize_arxiv_id(str(paper.get("arxiv_id") or ""))
        aminer_paper_id = mapping.get(arxiv_id, "")
        aminer_paper_url = build_aminer_paper_url(aminer_paper_id) if aminer_paper_id else ""
        enriched_papers.append(
            {
                **paper,
                "aminer_paper_id": aminer_paper_id or _clean_text(paper.get("aminer_paper_id")),
                "aminer_paper_url": aminer_paper_url or _clean_text(paper.get("aminer_paper_url")),
            }
        )
    return {**ranked_payload, "papers": enriched_papers}


def rank_arxiv_candidates(candidate_payload: dict[str, Any], profile: dict[str, Any], *, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    papers = candidate_payload.get("papers") or []
    primary_keywords = [_clean_text(item) for item in profile.get("retrieval_keywords") or profile.get("keywords") or [] if _clean_text(item)]
    secondary_keywords = [
        _clean_text(item)
        for item in profile.get("ranking_keywords") or profile.get("retrieval_keywords") or profile.get("keywords") or []
        if _clean_text(item) and _clean_text(item).casefold() not in {_clean_text(term).casefold() for term in primary_keywords}
    ]
    keyword_weights = {
        _clean_text(key).casefold(): float(value)
        for key, value in dict(profile.get("retrieval_term_weights") or {}).items()
        if _clean_text(key)
    }
    primary_keyword_tokens = {_clean_text(keyword).casefold(): _tokenize(keyword) for keyword in primary_keywords}
    secondary_keyword_tokens = {_clean_text(keyword).casefold(): _tokenize(keyword) for keyword in secondary_keywords}
    category_set = {str(item).strip() for item in profile.get("arxiv_categories") or [] if str(item).strip()}
    preferred_authors = {str(item).casefold() for item in profile.get("preferred_authors") or [] if str(item).strip()}
    preferred_venues = {str(item).casefold() for item in profile.get("preferred_venues") or [] if str(item).strip()}
    seed_tokens = _seed_terms(profile)
    profile_text_tokens = _tokenize(
        " ".join(
            [
                *[str(item) for item in profile.get("topics") or []],
                *[str(item) for item in profile.get("keywords") or []],
                *[str(seed.get("title") or "") for seed in profile.get("seed_papers") or [] if isinstance(seed, dict)],
            ]
        )
    )
    display_blocklist = {_clean_text(item).casefold() for item in profile.get("ranking_keywords") or [] if _is_meta_keyword(_clean_text(item))}
    is_scholar_profile = str(profile.get("profile_mode") or "") == "scholar_path"

    ranked: list[dict[str, Any]] = []
    for paper in papers:
        title = _clean_text(paper.get("title"))
        abstract = _clean_text(paper.get("abstract") or paper.get("summary"))
        has_effective_abstract = _has_effective_abstract(paper)
        paper_text = f"{title}\n{abstract}"
        title_tokens = _tokenize(title)
        text_tokens = _tokenize(paper_text)
        matched_keywords: list[str] = []
        primary_match_count = 0
        high_signal_match_count = 0
        generic_primary_match_count = 0
        lexical_score = 0.0
        for keyword, tokens in primary_keyword_tokens.items():
            if not tokens:
                continue
            weight = keyword_weights.get(keyword, 1.0)
            displayable = keyword not in display_blocklist
            is_meta = _is_meta_keyword(keyword)
            is_specific = _is_scholar_specific_keyword(keyword, seed_tokens)
            is_generic = _is_generic_domain_keyword(keyword)
            exact_gain = weight
            partial_gain = max(0.2, min(weight * 0.28, 0.95))
            if is_meta:
                exact_gain = min(exact_gain, 0.18)
                partial_gain = min(partial_gain, 0.05)
            elif is_specific:
                exact_gain = weight * 1.55
                partial_gain = max(partial_gain, min(weight * 0.45, 1.4))
            elif is_generic:
                exact_gain = weight * 0.62
                partial_gain = max(0.12, min(weight * 0.18, 0.55))
            overlap_count = _term_overlap_count(text_tokens, tokens)
            phrase_match = _term_phrase_match(paper_text, keyword)
            if tokens.issubset(text_tokens) or phrase_match:
                if displayable:
                    matched_keywords.append(keyword)
                lexical_score += exact_gain
                primary_match_count += 1
                if is_generic:
                    generic_primary_match_count += 1
                if is_specific or not is_generic:
                    high_signal_match_count += 1
            elif overlap_count >= _term_min_match_count(tokens):
                if displayable:
                    matched_keywords.append(keyword)
                lexical_score += partial_gain
                primary_match_count += 1
                if is_generic:
                    generic_primary_match_count += 1
                if is_specific:
                    high_signal_match_count += 1
        for keyword, tokens in secondary_keyword_tokens.items():
            if not tokens:
                continue
            if _is_meta_keyword(keyword):
                secondary_exact_gain = 0.05
                secondary_partial_gain = 0.01
            elif _is_scholar_specific_keyword(keyword, seed_tokens):
                secondary_exact_gain = 0.55
                secondary_partial_gain = 0.2
            elif _is_generic_domain_keyword(keyword):
                secondary_exact_gain = 0.18
                secondary_partial_gain = 0.04
            else:
                secondary_exact_gain = 0.35
                secondary_partial_gain = 0.1
            overlap_count = _term_overlap_count(text_tokens, tokens)
            phrase_match = _term_phrase_match(paper_text, keyword)
            if tokens.issubset(text_tokens) or phrase_match:
                if keyword not in display_blocklist:
                    matched_keywords.append(keyword)
                lexical_score += secondary_exact_gain
            elif overlap_count >= _term_min_match_count(tokens):
                if keyword not in display_blocklist:
                    matched_keywords.append(keyword)
                lexical_score += secondary_partial_gain

        matched_categories = [category for category in paper.get("categories") or [] if category in category_set]
        lexical_score += min(len(matched_categories) * 0.8, 1.6)

        matched_authors = [
            author
            for author in paper.get("authors") or []
            if str(author).casefold() in preferred_authors
        ]
        lexical_score += len(matched_authors) * 1.2

        venue = _clean_text(paper.get("venue"))
        if venue and venue.casefold() in preferred_venues:
            lexical_score += 0.6

        seed_overlap = len(text_tokens.intersection(seed_tokens))
        if seed_overlap:
            lexical_score += min(1.2, 0.15 * seed_overlap)

        lexical_score -= _application_domain_drift_penalty(text_tokens, profile_text_tokens)
        lexical_score -= _title_domain_drift_penalty(title_tokens, profile_text_tokens)
        lexical_score -= _generic_only_scholar_penalty(
            is_scholar_profile=is_scholar_profile,
            primary_match_count=primary_match_count,
            high_signal_match_count=high_signal_match_count,
            generic_primary_match_count=generic_primary_match_count,
            text_tokens=text_tokens,
            profile_tokens=profile_text_tokens,
        )
        lexical_score = max(0.0, lexical_score)
        recency_score = _calculate_recency_score(paper.get("published_date"))
        quality_score = _calculate_quality_score(abstract)
        abstract_score = 3.0 if has_effective_abstract else 0.0
        normalized_relevance = min(lexical_score, 6.0) / 6.0 * 10.0
        normalized_recency = recency_score / 3.0 * 10.0
        normalized_quality = quality_score / 3.0 * 10.0
        normalized_abstract = abstract_score / 3.0 * 10.0
        score = (
            normalized_relevance * 0.55
            + normalized_recency * 0.20
            + normalized_quality * 0.15
            + normalized_abstract * 0.10
        )
        if high_signal_match_count > 0:
            score += min(1.2, 0.4 * high_signal_match_count)
        if primary_match_count <= 0:
            score -= 1.0
        if is_scholar_profile and primary_match_count > 0 and high_signal_match_count <= 0:
            score = min(score, 6.9)

        # 应用 source_prior 权重（primary=1.0, fallback=0.6）
        source_prior = float(paper.get("source_prior") or 1.0)
        plan_role = str(paper.get("plan_role") or "unknown")
        score = score * source_prior

        ranked.append(
            {
                **paper,
                "has_effective_abstract": has_effective_abstract,
                "plan_role": plan_role,
                "source_prior": source_prior,
                "matched_keywords": matched_keywords[:8],
                "primary_match_count": primary_match_count,
                "high_signal_match_count": high_signal_match_count,
                "matched_categories": matched_categories,
                "matched_authors": matched_authors,
                "relevance_score": round(normalized_relevance, 2),
                "recency_score": round(normalized_recency, 2),
                "quality_score": round(normalized_quality, 2),
                "abstract_score": round(normalized_abstract, 2),
                "recommendation_score": round(score, 2),
                "aminer_comment": "；".join(
                    part
                    for part in (
                        f"命中关键词: {', '.join(matched_keywords[:4])}" if matched_keywords else "",
                        f"命中分类: {', '.join(matched_categories[:3])}" if matched_categories else "",
                        f"命中作者: {', '.join(matched_authors[:3])}" if matched_authors else "",
                    )
                    if part
                ),
                "author_entries": list(paper.get("author_entries") or [{"display_name": author, "profile_url": "", "is_disambiguated": False} for author in (paper.get("authors") or [])]),
                "aminer_author_profiles": list(paper.get("aminer_author_profiles") or []),
                "aminer_paper_url": _clean_text(paper.get("aminer_paper_url")) or build_aminer_paper_url_for_arxiv_paper(paper),
                "famous_authors": list(paper.get("famous_authors") or matched_authors[:3]),
            }
        )

    ranked.sort(
        key=lambda item: (
            float(item.get("recommendation_score") or 0.0),
            float(item.get("source_prior") or 1.0),
            item.get("high_signal_match_count", 0),
            item.get("primary_match_count", 0),
            -_days_since(item.get("published_date")),
            item.get("title", ""),
        ),
        reverse=True,
    )
    selected = ranked[: max(int(top_k), 1)]
    return {
        "status": "success",
        "generated_at": utc_now_iso(),
        "candidate_count": len(ranked),
        "paper_count": len(selected),
        "profile_name": profile.get("profile_name", ""),
        "recommendation_source": profile.get("source_metadata", {}).get("source", "unknown"),
        "ranked_candidates": ranked,
        "papers": selected,
    }
