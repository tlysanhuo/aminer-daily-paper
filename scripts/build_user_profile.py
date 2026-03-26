from __future__ import annotations

import re
from typing import Any

from scripts.datacenter_client import call_segmentation_pro
from scripts.internal_profile_provider import load_internal_uid_profile
from scripts.llm_client import ProfileTopicGenerationError, SummaryGenerationError, llm_profile_topics


TOPIC_CATEGORY_HINTS = {
    "多模态": ["cs.CV", "cs.CL", "cs.MM", "cs.AI"],
    "多模态学习": ["cs.CV", "cs.CL", "cs.MM", "cs.AI"],
    "视觉语言": ["cs.CV", "cs.CL", "cs.AI"],
    "视觉语言模型": ["cs.CV", "cs.CL", "cs.AI"],
    "机器学习": ["cs.LG", "cs.AI"],
    "machine learning": ["cs.LG", "cs.AI"],
    "深度学习": ["cs.LG", "cs.AI"],
    "deep learning": ["cs.LG", "cs.AI"],
    "自然语言处理": ["cs.CL", "cs.AI"],
    "natural language processing": ["cs.CL", "cs.AI"],
    "计算机视觉": ["cs.CV"],
    "computer vision": ["cs.CV"],
    "信息检索": ["cs.IR"],
    "information retrieval": ["cs.IR"],
    "知识图谱": ["cs.AI", "cs.IR"],
    "knowledge graph": ["cs.AI", "cs.IR"],
    "智能体": ["cs.AI", "cs.MA"],
    "智能代理": ["cs.AI", "cs.MA"],
    "多智能体": ["cs.AI", "cs.MA"],
    "具身智能": ["cs.RO", "cs.AI"],
    "机器人": ["cs.RO"],
    "vision": ["cs.CV", "cs.MM"],
    "visual": ["cs.CV", "cs.MM"],
    "image": ["cs.CV"],
    "video": ["cs.CV", "cs.MM"],
    "language": ["cs.CL", "cs.AI"],
    "llm": ["cs.CL", "cs.AI", "cs.LG"],
    "pre-training": ["cs.LG", "cs.AI", "cs.CL"],
    "fine-tuning": ["cs.LG", "cs.AI", "cs.CL"],
    "rlhf": ["cs.AI", "cs.LG"],
    "reasoning": ["cs.AI", "cs.CL"],
    "agent": ["cs.AI", "cs.MA"],
    "multi-agent": ["cs.AI", "cs.MA"],
    "robot": ["cs.RO"],
    "action": ["cs.RO", "cs.CV"],
    "multimodal": ["cs.CV", "cs.CL", "cs.MM"],
}

TOPIC_CANONICAL_ALIASES = {
    "多模态智能体": {
        "多模态智能体",
        "多模态代理",
        "multimodal agents",
        "multimodal agent",
    },
    "tool use": {
        "tool use",
        "tool-use",
        "工具使用",
    },
}

GENERIC_TOPIC_STOPWORDS = {
    "推荐",
    "看看",
    "看下",
    "最新",
    "最近",
    "论文",
    "最近论文",
    "最新论文",
    "paper",
    "papers",
    "research",
    "推荐下",
    "给我推荐",
    "帮我",
    "给我",
    "帮忙",
    "请推荐",
    "相关",
}

GENERIC_SEGMENTED_STOPWORDS = {
    *GENERIC_TOPIC_STOPWORDS,
    "recommend",
    "recommended",
    "recommendation",
    "latest",
}

LOW_SIGNAL_TOPIC_EXACT = {
    "we propose",
    "we present",
    "we introduce",
    "we study",
    "we show",
    "propose a novel",
    "show that our",
    "this paper",
    "our method",
    "our approach",
    "to address",
    "in this work",
}

LOW_SIGNAL_TOPIC_PREFIXES = (
    "we ",
    "our ",
    "this ",
    "these ",
    "those ",
    "to ",
    "in this ",
)

LOW_SIGNAL_TOPIC_HEADWORDS = {
    "we",
    "our",
    "this",
    "these",
    "those",
    "show",
    "showing",
    "propose",
    "proposing",
    "present",
    "presenting",
    "introduce",
    "introducing",
    "demonstrate",
    "demonstrating",
    "study",
    "studying",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _canonicalize_topic(topic: str) -> str:
    compact = _clean_text(topic)
    lowered = compact.casefold()
    if not compact:
        return ""
    for canonical, aliases in TOPIC_CANONICAL_ALIASES.items():
        if lowered == canonical.casefold():
            return canonical
        for alias in aliases:
            alias_lower = alias.casefold()
            if lowered == alias_lower:
                return canonical
    matched_canonicals = [
        canonical
        for canonical, aliases in TOPIC_CANONICAL_ALIASES.items()
        if any(alias.casefold() in lowered for alias in aliases)
    ]
    if len(set(matched_canonicals)) == 1:
        return matched_canonicals[0]
    if len(set(matched_canonicals)) > 1:
        return ""
    return compact


def _flatten_keyword_items(value: Any) -> list[str]:
    results: list[str] = []
    if isinstance(value, str):
        compact = _clean_text(value)
        if compact:
            results.append(compact)
        return results
    if isinstance(value, list):
        for item in value:
            results.extend(_flatten_keyword_items(item))
        return results
    if isinstance(value, dict):
        for item in value.values():
            results.extend(_flatten_keyword_items(item))
    return results


def _dedupe_texts(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        compact = _clean_text(item)
        if not compact:
            continue
        key = compact.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(compact)
    return deduped


def _is_generic_stopword(text: str, *, segmented: bool = False) -> bool:
    compact = _clean_text(text).casefold()
    if not compact:
        return True
    stopwords = GENERIC_SEGMENTED_STOPWORDS if segmented else GENERIC_TOPIC_STOPWORDS
    return compact in stopwords


def _looks_like_sentence_fragment_topic(text: str) -> bool:
    compact = _clean_text(text)
    lowered = compact.casefold()
    if not compact:
        return True
    if lowered in LOW_SIGNAL_TOPIC_EXACT:
        return True
    if any(lowered.startswith(prefix) for prefix in LOW_SIGNAL_TOPIC_PREFIXES):
        return True
    english_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", lowered)
    if not english_tokens:
        return False
    if english_tokens[:2] in (
        ["we", "propose"],
        ["we", "present"],
        ["we", "introduce"],
        ["we", "study"],
        ["we", "show"],
        ["show", "that"],
        ["to", "address"],
        ["in", "this"],
        ["our", "method"],
        ["our", "approach"],
        ["this", "paper"],
    ):
        return True
    if english_tokens[0] in LOW_SIGNAL_TOPIC_HEADWORDS:
        return True
    return False


def _normalize_topics(topics: list[str]) -> list[str]:
    normalized = [_canonicalize_topic(_clean_text(topic)) for topic in topics]
    return [
        topic
        for topic in _dedupe_texts(normalized)
        if topic and not _is_generic_stopword(topic) and not _looks_like_sentence_fragment_topic(topic)
    ]


def _keyword_priority(keyword: str) -> tuple[int, int, str]:
    compact = _clean_text(keyword)
    if not compact:
        return (3, 1, "")
    lowered = compact.casefold()
    has_ascii = bool(re.search(r"[a-z0-9]", lowered))
    is_phrase = 0 if " " in compact or "-" in compact else 1
    if _is_generic_stopword(compact, segmented=True):
        bucket = 4
    elif has_ascii:
        bucket = 0
    elif re.search(r"[\u4e00-\u9fff]", compact):
        bucket = 1
    else:
        bucket = 2
    return (bucket, is_phrase, lowered)


def _prioritize_keywords(keywords: list[str]) -> list[str]:
    deduped = _dedupe_texts(
        [
            keyword
            for keyword in keywords
            if not _is_generic_stopword(keyword, segmented=True) and not _looks_like_sentence_fragment_topic(keyword)
        ]
    )
    return sorted(deduped, key=_keyword_priority)


def _resolve_llm_config(config: dict[str, Any] | None = None) -> dict[str, str]:
    config = config or {}
    llm_config = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    return {
        "api_key": _clean_text(llm_config.get("api_key")),
        "base_url": _clean_text(llm_config.get("base_url")),
        "model": _clean_text(llm_config.get("model")) or "gpt-5-mini",
        "timeout_seconds": str(llm_config.get("timeout_seconds") or 30),
    }


def _resolve_llm_candidates(config: dict[str, Any] | None = None) -> list[dict[str, str]]:
    config = config or {}
    llm_config = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    candidates = [_resolve_llm_config(config)]
    fallback = llm_config.get("fallback") if isinstance(llm_config.get("fallback"), dict) else {}
    if fallback:
        candidates.append(
            {
                "api_key": _clean_text(fallback.get("api_key")),
                "base_url": _clean_text(fallback.get("base_url")),
                "model": _clean_text(fallback.get("model")) or candidates[0]["model"],
                "timeout_seconds": str(fallback.get("timeout_seconds") or candidates[0]["timeout_seconds"] or 30),
            }
        )
    return candidates


def _maybe_apply_llm_topics(
    internal_profile: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    allow_without_seed_papers: bool = False,
) -> tuple[dict[str, Any], str]:
    llm_candidates = _resolve_llm_candidates(config)
    if not any(candidate["api_key"] for candidate in llm_candidates):
        return internal_profile, "missing_api_key"
    seed_papers = list(internal_profile.get("seed_papers") or [])
    if len(seed_papers) < 3 and not allow_without_seed_papers:
        return internal_profile, "insufficient_seed_papers"
    llm_topics: list[dict[str, Any]] = []
    raw_output = ""
    failure_reasons: list[str] = []
    for index, llm_config in enumerate(llm_candidates):
        if not llm_config["api_key"]:
            continue
        label = "primary" if index == 0 else f"fallback_{index}"
        try:
            llm_topics, raw_output = llm_profile_topics(
                internal_profile,
                api_key=llm_config["api_key"],
                base_url=llm_config["base_url"],
                model=llm_config["model"],
                timeout_seconds=int(llm_config["timeout_seconds"]),
            )
            break
        except (ProfileTopicGenerationError, SummaryGenerationError) as exc:
            failure_reasons.append(f"{label}:{exc}")
        except Exception as exc:
            failure_reasons.append(f"{label}:llm_topic_unavailable:{exc.__class__.__name__}")
    if not llm_topics:
        return internal_profile, "; ".join(failure_reasons) if failure_reasons else "llm_topic_unavailable"
    llm_topic_names = _dedupe_texts([str(item.get("name") or "") for item in llm_topics if isinstance(item, dict)])
    llm_keywords = _prioritize_keywords(
        [
            keyword
            for item in llm_topics
            if isinstance(item, dict)
            for keyword in list(item.get("keywords") or [])
        ]
    )
    if not llm_topic_names:
        return internal_profile, "invalid_llm_topics"
    merged = dict(internal_profile)
    merged["topics"] = llm_topic_names[:5]
    merged["keywords"] = _prioritize_keywords([*llm_keywords, *list(internal_profile.get("keywords") or []), *llm_topic_names])
    metadata = dict(merged.get("source_metadata") or {})
    metadata["llm_topics"] = llm_topics
    metadata["llm_topic_raw_output"] = raw_output
    merged["source_metadata"] = metadata
    return merged, ""


def _flatten_param_values(value: Any) -> list[str]:
    flattened: list[str] = []
    if isinstance(value, str):
        compact = _clean_text(value)
        if compact:
            flattened.append(compact)
        return flattened
    if isinstance(value, list):
        for item in value:
            flattened.extend(_flatten_param_values(item))
        return flattened
    if isinstance(value, dict):
        for item in value.values():
            flattened.extend(_flatten_param_values(item))
    return flattened


def _categories_from_topics(topics: list[str], extra_keywords: list[str]) -> list[str]:
    categories: list[str] = []
    for text in [*topics, *extra_keywords]:
        lowered = text.casefold()
        for hint, mapped_categories in TOPIC_CATEGORY_HINTS.items():
            if hint in lowered:
                for category in mapped_categories:
                    if category not in categories:
                        categories.append(category)
    return categories


def _is_cs_user(categories: list[str]) -> bool:
    return any(str(category).strip().startswith("cs.") for category in categories if str(category).strip())


def _build_recall_strategy(categories: list[str]) -> dict[str, Any]:
    is_cs = _is_cs_user(categories)
    return {
        "is_cs_user": is_cs,
        "primary_recall_source": "arxiv" if is_cs else "aminer",
        "secondary_recall_source": "aminer" if is_cs else "arxiv",
        "arxiv_role": "primary" if is_cs else "supplemental",
    }


def _query_from_topics(topics: list[str]) -> str:
    cleaned_topics = [_clean_text(topic) for topic in topics if _clean_text(topic)]
    return "，".join(cleaned_topics)


def build_user_profile(uid: str, topics: list[str], *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cleaned_uid = _clean_text(uid)
    explicit_topics = _normalize_topics(topics)
    if not cleaned_uid:
        return {
            "status": "degraded",
            "enabled": False,
            "user_id": "",
            "source_metadata": {"reason": "missing_user_id"},
        }

    internal_profile = load_internal_uid_profile(cleaned_uid)
    llm_topic_reason = ""
    if internal_profile.get("status") == "success":
        internal_profile, llm_topic_reason = _maybe_apply_llm_topics(internal_profile, config=config)
    internal_topics = _normalize_topics(list(internal_profile.get("topics") or internal_profile.get("experts_topics") or []))
    internal_keywords = _prioritize_keywords(list(internal_profile.get("keywords") or []))
    merged_topics = _dedupe_texts([*internal_topics, *explicit_topics])
    query_inputs = merged_topics or internal_keywords[:6]
    if not query_inputs:
        return {
            "status": "degraded",
            "enabled": False,
            "user_id": cleaned_uid,
            "source_metadata": {
                "reason": str((internal_profile.get("source_metadata") or {}).get("reason") or "missing_topics"),
            },
        }

    query = _query_from_topics(query_inputs)
    segmented_keywords: list[str] = []
    params: dict[str, Any] = {}
    segmentation_reason = ""
    if query:
        try:
            segmentation_data = call_segmentation_pro(query=query, user_id=cleaned_uid, config=config)
            segmented_keywords = _prioritize_keywords(_flatten_keyword_items(segmentation_data.get("keywords")))
            params = segmentation_data.get("params") if isinstance(segmentation_data.get("params"), dict) else {}
        except Exception as exc:
            segmentation_reason = f"segmentation_unavailable:{exc}"

    keywords = _prioritize_keywords([*segmented_keywords, *internal_keywords, *merged_topics])
    categories = _categories_from_topics(merged_topics or internal_topics, keywords)
    recall_strategy = _build_recall_strategy(categories)
    preferred_authors = _dedupe_texts(
        [
            *list(internal_profile.get("preferred_authors") or []),
            *_flatten_param_values(params.get("person")),
        ]
    )
    preferred_venues = _dedupe_texts(
        [
            *list(internal_profile.get("preferred_venues") or []),
            *_flatten_param_values(params.get("venue")),
            *_flatten_param_values(params.get("conference")),
            *_flatten_param_values(params.get("journal")),
        ]
    )

    return {
        "status": "success",
        "enabled": True,
        "user_id": cleaned_uid,
        "profile_name": _clean_text(internal_profile.get("user_name")) or f"aminer_user_{cleaned_uid[-6:]}",
        "bind_scholar_ids": list(internal_profile.get("bind_scholar_ids") or []),
        "topics": merged_topics,
        "keywords": keywords[:18],
        "arxiv_categories": categories,
        "is_cs_user": recall_strategy["is_cs_user"],
        "recall_primary_source": recall_strategy["primary_recall_source"],
        "recall_secondary_source": recall_strategy["secondary_recall_source"],
        "recall_strategy": recall_strategy,
        "preferred_authors": preferred_authors[:8],
        "preferred_venues": preferred_venues[:6],
        "seed_papers": list(internal_profile.get("seed_papers") or [])[:8],
        "source_metadata": {
            "source": str((internal_profile.get("source_metadata") or {}).get("source") or "datacenter_segmentation_profile"),
            "query": query,
            "segmentation_params": params,
            "segmented_keyword_count": len(segmented_keywords),
            "internal_profile": internal_profile.get("source_metadata") or {},
            "degraded_reason": segmentation_reason,
            "llm_topic_reason": llm_topic_reason,
        },
    }


def build_topics_profile(
    topics: list[str],
    *,
    config: dict[str, Any] | None = None,
    enable_llm_topics: bool = True,
) -> dict[str, Any]:
    normalized_topics = _normalize_topics(topics)
    segmented_keywords: list[str] = []
    query = _query_from_topics(normalized_topics)
    if query:
        try:
            segmentation_data = call_segmentation_pro(query=query, user_id="", config=config)
        except Exception:
            segmentation_data = {}
        segmented_keywords = _prioritize_keywords(_flatten_keyword_items(segmentation_data.get("keywords")))
    keywords = _prioritize_keywords([*segmented_keywords, *normalized_topics])
    categories = _categories_from_topics(normalized_topics, keywords)
    recall_strategy = _build_recall_strategy(categories)
    profile = {
        "status": "success",
        "enabled": True,
        "user_id": "",
        "profile_name": "topics_fallback",
        "bind_scholar_ids": [],
        "topics": normalized_topics,
        "keywords": keywords,
        "arxiv_categories": categories,
        "is_cs_user": recall_strategy["is_cs_user"],
        "recall_primary_source": recall_strategy["primary_recall_source"],
        "recall_secondary_source": recall_strategy["secondary_recall_source"],
        "recall_strategy": recall_strategy,
        "preferred_authors": [],
        "preferred_venues": [],
        "seed_papers": [],
        "source_metadata": {
            "source": "topics_fallback",
            "query": query,
            "segmented_keyword_count": len(segmented_keywords),
        },
    }
    metadata = dict(profile.get("source_metadata") or {})
    if enable_llm_topics:
        profile, llm_topic_reason = _maybe_apply_llm_topics(profile, config=config, allow_without_seed_papers=True)
        metadata = dict(profile.get("source_metadata") or {})
        metadata["llm_topic_reason"] = llm_topic_reason
    else:
        metadata["llm_topic_reason"] = "skipped_disabled"
        profile["source_metadata"] = metadata
        return profile
    profile["source_metadata"] = metadata
    return profile
