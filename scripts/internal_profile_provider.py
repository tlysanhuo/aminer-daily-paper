from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_MAX_AUTHORED_PAPERS = 20

GENERIC_DIRECTION_TERMS = {
    "computer science",
    "computer science and technology",
    "计算机科学技术",
    "机器学习",
    "machine learning",
    "深度学习",
    "deep learning",
    "人工智能",
    "artificial intelligence",
    "信息检索",
    "information retrieval",
    "计算机视觉",
    "computer vision",
    "自然语言处理",
    "natural language processing",
}

GENERIC_KEYWORD_TERMS = {
    *GENERIC_DIRECTION_TERMS,
    "generative model",
    "knowledge based systems",
    "knowledge engineering",
    "message service",
    "instant messaging",
    "computer science",
}

ENGLISH_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "via",
    "with",
}

LOW_SIGNAL_PHRASES = {
    "benchmark",
    "benchmarks",
    "evaluation",
    "evaluations",
    "method",
    "methods",
    "model",
    "models",
    "paper",
    "papers",
    "research",
    "system",
    "systems",
    "task",
    "tasks",
}

LOW_SIGNAL_HEADWORDS = {
    "analysis",
    "approach",
    "framework",
    "method",
    "methods",
    "model",
    "models",
    "study",
    "studies",
    "system",
    "systems",
}

SIGNAL_TAILWORDS = {
    "comprehension",
    "decoding",
    "disambiguation",
    "extraction",
    "grounding",
    "reasoning",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _resolve_recsys_next_dir() -> Path | None:
    configured = _clean_text(os.getenv("RECSYS_NEXT_DIR"))
    if not configured:
        return None
    return Path(configured).expanduser()


def _normalize_term(value: Any) -> str:
    text = _clean_text(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;，；、")


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _split_text_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = _normalize_term(value)
        if not text:
            return []
        return [_normalize_term(part) for part in re.split(r"[;,/|，；、]+", text) if _normalize_term(part)]
    if isinstance(value, list):
        terms: list[str] = []
        for item in value:
            terms.extend(_split_text_terms(item))
        return terms
    if isinstance(value, dict):
        terms: list[str] = []
        for item in value.values():
            terms.extend(_split_text_terms(item))
        return terms
    return []


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        text = _normalize_term(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _is_generic_direction_term(text: str) -> bool:
    return _normalize_term(text).casefold() in {term.casefold() for term in GENERIC_DIRECTION_TERMS}


def _is_generic_keyword_term(text: str) -> bool:
    return _normalize_term(text).casefold() in {term.casefold() for term in GENERIC_KEYWORD_TERMS}


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _english_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", text.lower())


def _specificity_bonus(text: str) -> float:
    normalized = _normalize_term(text)
    if not normalized:
        return 0.0
    if _contains_chinese(normalized):
        return 0.8 if len(normalized) >= 6 else 0.45
    tokens = _english_tokens(normalized)
    if not tokens:
        return 0.0
    if len(tokens) >= 3:
        return 1.1
    if len(tokens) == 2:
        return 0.8
    token = tokens[0]
    if len(token) >= 5 and token.isalpha():
        return 0.3
    if normalized.isupper() and 2 <= len(normalized) <= 8:
        return 0.6
    return 0.0


def _is_low_signal_phrase(text: str) -> bool:
    normalized = _normalize_term(text)
    if not normalized:
        return True
    lowered = normalized.casefold()
    if _is_generic_keyword_term(normalized):
        return True
    if lowered in LOW_SIGNAL_PHRASES:
        return True
    if _contains_chinese(normalized):
        return len(normalized) <= 2
    tokens = _english_tokens(normalized)
    if not tokens:
        return True
    if len(tokens) == 1:
        token = tokens[0]
        if len(token) <= 3 and normalized != normalized.upper():
            return True
        if token in LOW_SIGNAL_HEADWORDS or token in LOW_SIGNAL_PHRASES:
            return True
    if len(tokens) >= 2:
        if tokens[0] in ENGLISH_FUNCTION_WORDS or tokens[-1] in ENGLISH_FUNCTION_WORDS:
            return True
        if tokens[-1] in LOW_SIGNAL_HEADWORDS and len(tokens) < 3:
            return True
        if tokens[-1] in {"benchmark", "benchmarks", "evaluation", "evaluations"}:
            return True
        if any(token == "bench" or token.endswith("-bench") for token in tokens):
            return True
        if any(token in {"benchmark", "benchmarks", "evaluation", "evaluations"} for token in tokens):
            if not any(token in SIGNAL_TAILWORDS for token in tokens):
                return True
    return False


def _phrase_tokens(text: str) -> list[str]:
    normalized = _normalize_term(text)
    if _contains_chinese(normalized):
        return list(normalized)
    return _english_tokens(normalized)


def _is_redundant_phrase(candidate: str, selected: list[str]) -> bool:
    candidate_tokens = _phrase_tokens(candidate)
    if not candidate_tokens:
        return False
    candidate_joined = " ".join(candidate_tokens)
    for chosen in selected:
        chosen_tokens = _phrase_tokens(chosen)
        if len(chosen_tokens) <= len(candidate_tokens):
            continue
        chosen_joined = " ".join(chosen_tokens)
        if candidate_joined and candidate_joined in chosen_joined:
            return True
        if _contains_chinese(candidate) and candidate in chosen:
            return True
    return False


def _select_distinct_terms(ordered_terms: list[str], limit: int) -> list[str]:
    selected: list[str] = []
    for term in ordered_terms:
        normalized = _normalize_term(term)
        if not normalized or _is_redundant_phrase(normalized, selected):
            continue
        selected.append(normalized)
        if len(selected) >= limit:
            break
    return selected


def _extract_phrase_candidates(text: str) -> list[str]:
    normalized = _normalize_term(text)
    if not normalized:
        return []
    if _contains_chinese(normalized):
        return [] if _is_low_signal_phrase(normalized) else [normalized]

    lowered = normalized.casefold()
    tokens = _english_tokens(lowered)
    candidates: list[str] = []

    raw_chunks = re.split(r"[,;:()\[\]\n]+", normalized)
    for chunk in raw_chunks:
        compact = _normalize_term(chunk)
        chunk_tokens = _english_tokens(compact.casefold())
        if (
            compact
            and not _is_low_signal_phrase(compact)
            and (not chunk_tokens or len(chunk_tokens) <= 4)
            and (not chunk_tokens or chunk_tokens[0] not in LOW_SIGNAL_PHRASES)
        ):
            candidates.append(compact)

    for size in range(2, min(4, len(tokens)) + 1):
        for index in range(0, len(tokens) - size + 1):
            gram = tokens[index : index + size]
            if gram[0] in ENGLISH_FUNCTION_WORDS or gram[-1] in ENGLISH_FUNCTION_WORDS:
                continue
            if gram[0] in LOW_SIGNAL_PHRASES:
                continue
            phrase = " ".join(gram)
            if _is_low_signal_phrase(phrase):
                continue
            candidates.append(phrase)

    acronyms = re.findall(r"\b[A-Z][A-Z0-9-]{1,9}\b", normalized)
    candidates.extend(acronyms)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        compact = _normalize_term(item)
        key = compact.casefold()
        if not compact or key in seen or _is_low_signal_phrase(compact):
            continue
        seen.add(key)
        deduped.append(compact)
    return deduped


def _add_candidate_score(
    scores: dict[str, float],
    canonical: dict[str, str],
    support_titles: dict[str, set[str]],
    phrase_keywords: dict[str, Counter[str]],
    phrase: str,
    *,
    score: float,
    paper_title: str = "",
) -> None:
    normalized = _normalize_term(phrase)
    if not normalized or _is_low_signal_phrase(normalized):
        return
    key = normalized.casefold()
    scores[key] += score + _specificity_bonus(normalized)
    canonical.setdefault(key, normalized)
    if paper_title:
        support_titles[key].add(paper_title)
    phrase_keywords[key][normalized] += 1


def _load_mongo_modules() -> tuple[Any, Any, Any, Any]:
    recsys_next_dir = _resolve_recsys_next_dir()
    if recsys_next_dir is None:
        raise FileNotFoundError("missing_RECSYS_NEXT_DIR")
    if not recsys_next_dir.exists():
        raise FileNotFoundError(f"missing_RECSYS_NEXT_DIR_path:{recsys_next_dir}")
    if str(recsys_next_dir) not in sys.path:
        sys.path.insert(0, str(recsys_next_dir))
    from bson import ObjectId  # type: ignore
    from mongo import AMinerOnline, PubRel, RecSys  # type: ignore

    return ObjectId, AMinerOnline, RecSys, PubRel


def _extract_experts_topics(usr_doc: dict[str, Any]) -> list[str]:
    topics: list[str] = []
    for item in usr_doc.get("experts_topic") or []:
        if not isinstance(item, dict):
            continue
        topic = _normalize_term(item.get("input_name") or item.get("name") or item.get("name_zh"))
        if topic:
            topics.append(topic)
    return _dedupe_keep_order(topics)


def _collect_keyword_values(doc: dict[str, Any]) -> list[str]:
    return _dedupe_keep_order([*_split_text_terms(doc.get("keywords")), *_split_text_terms(doc.get("keywords_zh"))])


def _collect_field_values(doc: dict[str, Any]) -> list[str]:
    return _dedupe_keep_order(
        [
            *_split_text_terms(doc.get("main_field")),
            *_split_text_terms(doc.get("fields")),
            *_split_text_terms(doc.get("field")),
        ]
    )


def _collect_topic_values(doc: dict[str, Any]) -> list[str]:
    return _dedupe_keep_order([*_split_text_terms(doc.get("sub_field")), *_split_text_terms(doc.get("topic"))])


def _paper_weight(year: int, n_citation: int) -> float:
    score = 1.0
    if year >= 2024:
        score += 0.5
    elif year >= 2021:
        score += 0.25
    score += min(math.log1p(max(n_citation, 0)) / 4.0, 0.8)
    return score


def _paper_text(paper: dict[str, Any]) -> str:
    parts = [
        _normalize_term(paper.get("title")),
        _normalize_term(paper.get("abstract")),
        " ".join(_normalize_term(item) for item in list(paper.get("fields") or []) if _normalize_term(item)),
        " ".join(_normalize_term(item) for item in list(paper.get("topics") or []) if _normalize_term(item)),
        " ".join(_normalize_term(item) for item in list(paper.get("keywords") or []) if _normalize_term(item)),
    ]
    return "\n".join(part for part in parts if part).lower()


def _build_phrase_scores(
    authored_papers: list[dict[str, Any]],
    experts_topics: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, str], dict[str, set[str]], dict[str, Counter[str]]]:
    topic_scores: dict[str, float] = defaultdict(float)
    keyword_scores: dict[str, float] = defaultdict(float)
    canonical: dict[str, str] = {}
    support_titles: dict[str, set[str]] = defaultdict(set)
    phrase_keywords: dict[str, Counter[str]] = defaultdict(Counter)

    for paper in authored_papers:
        title = _normalize_term(paper.get("title"))
        abstract = _normalize_term(paper.get("abstract"))
        weight = _paper_weight(_to_int(paper.get("year")), _to_int(paper.get("n_citation")))
        for field in paper.get("fields") or []:
            _add_candidate_score(
                topic_scores,
                canonical,
                support_titles,
                phrase_keywords,
                field,
                score=2.1 * weight,
                paper_title=title,
            )
        for topic in paper.get("topics") or []:
            _add_candidate_score(
                topic_scores,
                canonical,
                support_titles,
                phrase_keywords,
                topic,
                score=2.4 * weight,
                paper_title=title,
            )
        for keyword in paper.get("keywords") or []:
            _add_candidate_score(
                topic_scores,
                canonical,
                support_titles,
                phrase_keywords,
                keyword,
                score=1.9 * weight,
                paper_title=title,
            )
            _add_candidate_score(
                keyword_scores,
                canonical,
                support_titles,
                phrase_keywords,
                keyword,
                score=2.2 * weight,
                paper_title=title,
            )
        for candidate in _extract_phrase_candidates(title):
            _add_candidate_score(
                topic_scores,
                canonical,
                support_titles,
                phrase_keywords,
                candidate,
                score=1.5 * weight,
                paper_title=title,
            )
            _add_candidate_score(
                keyword_scores,
                canonical,
                support_titles,
                phrase_keywords,
                candidate,
                score=1.1 * weight,
                paper_title=title,
            )
        for candidate in _extract_phrase_candidates(abstract):
            _add_candidate_score(
                topic_scores,
                canonical,
                support_titles,
                phrase_keywords,
                candidate,
                score=0.8 * weight,
                paper_title=title,
            )
            _add_candidate_score(
                keyword_scores,
                canonical,
                support_titles,
                phrase_keywords,
                candidate,
                score=0.9 * weight,
                paper_title=title,
            )

    for topic in experts_topics:
        _add_candidate_score(
            topic_scores,
            canonical,
            support_titles,
            phrase_keywords,
            topic,
            score=2.6,
        )
        _add_candidate_score(
            keyword_scores,
            canonical,
            support_titles,
            phrase_keywords,
            topic,
            score=1.6,
        )

    return topic_scores, keyword_scores, canonical, support_titles, phrase_keywords


def _build_core_topics(
    topic_scores: dict[str, float],
    canonical: dict[str, str],
    support_titles: dict[str, set[str]],
    phrase_keywords: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    for key, score in topic_scores.items():
        if score <= 0:
            continue
        name = canonical.get(key, "")
        if not name or _is_low_signal_phrase(name):
            continue
        supporting_papers = sorted(support_titles.get(key, set()))
        support_count = len(supporting_papers)
        final_score = score + min(support_count * 0.9, 2.7)
        ordered.append(
            {
                "name": name,
                "score": round(final_score, 3),
                "support_count": support_count,
                "keywords": [item for item, _ in phrase_keywords.get(key, Counter()).most_common(4)] or [name],
                "supporting_papers": supporting_papers[:3],
            }
        )
    ordered.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["support_count"]),
            -len(_english_tokens(str(item["name"]))),
            str(item["name"]).casefold(),
        )
    )
    selected_names: list[str] = []
    filtered: list[dict[str, Any]] = []
    for item in ordered:
        name = str(item["name"])
        if _is_redundant_phrase(name, selected_names):
            continue
        selected_names.append(name)
        filtered.append(item)
        if len(filtered) >= 6:
            break
    return filtered


def _score_terms(
    authored_papers: list[dict[str, Any]],
    experts_topics: list[str],
) -> tuple[list[str], list[str], list[str], list[str], list[dict[str, Any]]]:
    venue_scores: Counter[str] = Counter()
    author_scores: Counter[str] = Counter()
    topic_scores, keyword_scores, canonical, support_titles, phrase_keywords = _build_phrase_scores(
        authored_papers,
        experts_topics,
    )
    core_topics = _build_core_topics(topic_scores, canonical, support_titles, phrase_keywords)

    for paper in authored_papers:
        paper_weight = _paper_weight(_to_int(paper.get("year")), _to_int(paper.get("n_citation")))
        for field in paper.get("fields") or []:
            norm = _normalize_term(field)
            if norm and not _is_generic_direction_term(norm):
                topic_scores[norm.casefold()] += 1.2 * paper_weight
        for topic in paper.get("topics") or []:
            norm = _normalize_term(topic)
            if norm and not _is_generic_direction_term(norm):
                topic_scores[norm.casefold()] += 1.1 * paper_weight
        for keyword in paper.get("keywords") or []:
            norm = _normalize_term(keyword)
            if not norm or _is_generic_keyword_term(norm):
                continue
            topic_scores[norm.casefold()] += 0.7 * paper_weight
            keyword_scores[norm.casefold()] += 1.0 * paper_weight
        venue = _normalize_term(paper.get("venue"))
        if venue:
            venue_scores[venue] += 1
        for author in paper.get("coauthor_names") or []:
            author_scores[_normalize_term(author)] += 1

    for topic in experts_topics:
        norm = _normalize_term(topic)
        if not norm:
            continue
        if not _is_generic_direction_term(norm):
            topic_scores[norm.casefold()] += 1.5
        if not _is_generic_keyword_term(norm):
            keyword_scores[norm.casefold()] += 0.8

    narrow_topics = _select_distinct_terms(
        [
            canonical.get(key, "")
            for key, _ in sorted(topic_scores.items(), key=lambda item: (-item[1], item[0]))
            if canonical.get(key, "")
        ],
        8,
    )
    inferred_topics = _dedupe_keep_order([*[topic["name"] for topic in core_topics], *narrow_topics])[:8]
    narrow_keywords = _select_distinct_terms(
        [
            canonical.get(key, "")
            for key, _ in sorted(keyword_scores.items(), key=lambda item: (-item[1], item[0]))
            if canonical.get(key, "")
        ],
        16,
    )
    inferred_keywords = _dedupe_keep_order(
        [
            *[keyword for topic in core_topics for keyword in topic["keywords"]],
            *narrow_keywords,
        ]
    )[:16]
    preferred_venues = [term for term, _ in venue_scores.most_common(6) if term]
    preferred_authors = [term for term, count in author_scores.most_common(8) if term and count >= 1]
    return inferred_topics, inferred_keywords, preferred_authors, preferred_venues, core_topics


def _score_terms_for_bucket(
    papers: list[dict[str, Any]],
    bucket_name: str,
    experts_topics: list[str] | None = None,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Extract topics/keywords from a single bucket of papers.

    Reuses _score_terms logic, returning only the subset needed for bucket scoring.

    Args:
        papers: List of papers in this bucket
        bucket_name: Name of the bucket (e.g., 'recent', 'anchor')
        experts_topics: Optional list of expert topics to boost

    Returns:
        Tuple of (topics, keywords, core_topics)
    """
    # Reuse existing _score_terms logic
    topics, keywords, _, _, core_topics = _score_terms(papers, experts_topics or [])
    return topics, keywords, core_topics


def _fetch_field_map(pids: list[str], oid_to_pid: dict[Any, str], RecSys: Any, PubRel: Any) -> dict[str, dict[str, Any]]:
    field_map: dict[str, dict[str, Any]] = {}
    try:
        recsys = RecSys()
        pubrel = PubRel()
    except Exception:
        return field_map

    pid_keys: list[Any] = [*pids, *list(oid_to_pid.keys())]
    for collection in (getattr(recsys, "paper_fields_v3", None), getattr(pubrel, "paper_fields", None)):
        if collection is None:
            continue
        try:
            rows = collection.find(
                {"pid": {"$in": pid_keys}},
                {"pid": 1, "main_field": 1, "sub_field": 1, "fields": 1, "topic": 1, "field": 1},
            )
        except Exception:
            continue
        for row in rows:
            pid = _normalize_term(row.get("pid")) or oid_to_pid.get(row.get("pid"), "")
            if not pid:
                continue
            field_map.setdefault(pid, {}).update(
                {
                    "main_field": row.get("main_field"),
                    "sub_field": row.get("sub_field"),
                    "fields": row.get("fields"),
                    "topic": row.get("topic"),
                    "field": row.get("field"),
                }
            )
    return field_map


def load_internal_uid_profile(uid: str, *, max_authored_papers: int = DEFAULT_MAX_AUTHORED_PAPERS) -> dict[str, Any]:
    cleaned_uid = _normalize_term(uid)
    if not cleaned_uid:
        return {"status": "degraded", "source_metadata": {"reason": "missing_user_id"}}

    try:
        ObjectId, AMinerOnline, RecSys, PubRel = _load_mongo_modules()
    except Exception as exc:
        return {"status": "degraded", "source_metadata": {"reason": f"mongo_import_failed:{exc}"}}

    try:
        online = AMinerOnline()
    except Exception as exc:
        return {"status": "degraded", "source_metadata": {"reason": f"mongo_connect_failed:{exc}"}}

    usr_col = online.usr
    pub_col = online.publication_dupl

    try:
        usr_doc = usr_col.find_one(
            {"_id": ObjectId(cleaned_uid)} if ObjectId.is_valid(cleaned_uid) else {"_id": cleaned_uid},
            {"_id": 1, "name": 1, "bind": 1, "experts_topic": 1},
        )
    except Exception as exc:
        return {"status": "degraded", "source_metadata": {"reason": f"usr_lookup_failed:{exc}"}}

    if not isinstance(usr_doc, dict):
        return {"status": "degraded", "source_metadata": {"reason": "uid_not_found"}}

    bind_id = _normalize_term(usr_doc.get("bind"))
    experts_topics = _extract_experts_topics(usr_doc)
    user_name = _normalize_term(usr_doc.get("name"))

    authored_rows: list[dict[str, Any]] = []
    if bind_id:
        author_query_values: list[Any] = [bind_id]
        if ObjectId.is_valid(bind_id):
            author_query_values.insert(0, ObjectId(bind_id))
        try:
            authored_rows = list(
                pub_col.find(
                    {"authors._id": {"$in": author_query_values}},
                    {
                        "_id": 1,
                        "title": 1,
                        "title_zh": 1,
                        "abstract": 1,
                        "abstract_zh": 1,
                        "summary": 1,
                        "summary_zh": 1,
                        "keywords": 1,
                        "keywords_zh": 1,
                        "venue": 1,
                        "year": 1,
                        "n_citation": 1,
                        "authors": 1,
                    },
                )
                .sort([("year", -1), ("n_citation", -1)])
                .limit(max(int(max_authored_papers), 1))
            )
        except Exception:
            authored_rows = []

    oid_to_pid: dict[Any, str] = {}
    paper_ids: list[str] = []
    for row in authored_rows:
        pid = _normalize_term(row.get("_id"))
        if not pid:
            continue
        paper_ids.append(pid)
        if ObjectId.is_valid(pid):
            oid_to_pid[ObjectId(pid)] = pid
    field_map = _fetch_field_map(paper_ids, oid_to_pid, RecSys, PubRel) if paper_ids else {}

    authored_papers: list[dict[str, Any]] = []
    for row in authored_rows:
        pid = _normalize_term(row.get("_id"))
        if not pid:
            continue
        metadata = field_map.get(pid, {})
        title = _normalize_term(row.get("title") or row.get("title_zh"))
        abstract = _normalize_term(
            row.get("abstract") or row.get("abstract_zh") or row.get("summary") or row.get("summary_zh")
        )
        keywords = _collect_keyword_values({**row, **metadata})
        fields = _collect_field_values(metadata)
        topics = _collect_topic_values(metadata)
        venue = _normalize_term(row.get("venue"))
        coauthor_names: list[str] = []
        for author in row.get("authors") or []:
            if not isinstance(author, dict):
                continue
            author_id = _normalize_term(author.get("_id"))
            author_name = _normalize_term(author.get("name") or author.get("name_zh"))
            if not author_name:
                continue
            if bind_id and author_id == bind_id:
                continue
            if user_name and author_name == user_name:
                continue
            coauthor_names.append(author_name)
        authored_papers.append(
            {
                "paper_id": pid,
                "title": title,
                "abstract": abstract,
                "keywords": keywords,
                "fields": fields,
                "topics": topics,
                "venue": venue,
                "year": _to_int(row.get("year")),
                "n_citation": _to_int(row.get("n_citation")),
                "coauthor_names": _dedupe_keep_order(coauthor_names),
            }
        )

    inferred_topics, inferred_keywords, preferred_authors, preferred_venues, core_topics = _score_terms(authored_papers, experts_topics)

    has_bind_signal = bool(bind_id and authored_papers)
    has_topic_signal = bool(experts_topics)
    if not has_bind_signal and not has_topic_signal:
        return {
            "status": "degraded",
            "user_name": user_name,
            "bind_scholar_ids": [bind_id] if bind_id else [],
            "seed_papers": [],
            "source_metadata": {"reason": "no_bind_papers_or_experts_topic"},
        }

    source = "authored_papers_bind_profile" if has_bind_signal else "experts_topic_profile"
    return {
        "status": "success",
        "user_name": user_name,
        "bind_scholar_ids": [bind_id] if bind_id else [],
        "topics": inferred_topics,
        "keywords": inferred_keywords,
        "preferred_authors": preferred_authors,
        "preferred_venues": preferred_venues,
        "experts_topics": experts_topics,
        "seed_papers": authored_papers[:8],
        "source_metadata": {
            "source": source,
            "bind_scholar_id": bind_id,
            "authored_paper_count": len(authored_papers),
            "experts_topic_count": len(experts_topics),
            "core_topics": core_topics,
        },
    }
