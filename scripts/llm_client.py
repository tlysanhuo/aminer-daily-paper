from __future__ import annotations

import json
import re
from typing import Any

DEFAULT_LLM_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = """假如你是推荐机器人，根据给定的论文信息，包括论文title,authors,abstract,comments，对这篇论文生成一份总结，确保包含以下字段：keywords, structured_summary, famous_authors。

其中：
- keywords: 2 到 4 个中文短关键词
- structured_summary: JSON 对象，包含 research_problem, research_challenge, research_method，以及可选的 experimental_results
- structured_summary 的每个字段都应是简洁中文短句，突出论文 insight 和贡献，不要编造摘要里没有的信息
- 如果论文不包含明确实验结果，可省略 experimental_results
- 如果 abstract 缺失、过短、或与 title 基本相同，禁止根据 title 臆测论文内容；这时应明确说明信息不足
- famous_authors部分是作者简单介绍，作者之间要换行，选择最有名的，h-index高于30的两个学者，如果不足两个则输出一个或者不输出，简介头衔，荣誉称号，研究方向和h-index等，中文
- famous_authors字段由输入的famous authors传入的内容来生成，如果没有传入则生成空

输出内容格式要求：
- 以 JSON 对象输出，只返回 keywords、structured_summary、famous_authors
- 每个部分的内容都简短一点，但是要全面
- 所有英文名称、术语必须保持原始拼写，禁止翻译或音译
- 不要输出额外解释。"""

PROFILE_TOPIC_SYSTEM_PROMPT = """你是学术画像归纳助手。请根据给定用户的已发表论文、已有显式 topics 和关键词，归纳 3 到 5 个稳定研究方向。

要求：
- 输出 JSON 对象，只返回 topics
- topics 是数组，每个元素包含：
  - name: 一个简洁研究方向名，优先中文；已有英文术语如 RAG、LLM、NER、OAG 保持原样
  - keywords: 2 到 5 个该方向的检索关键词，可中英混合，但要适合论文检索
  - rationale: 一句简短中文说明，说明为什么这个方向成立
- 不要编造输入中不存在的方向
- 优先保留高频、连续、可检索的方向，而不是过细的问题
- 如果方向之间高度重合，要合并
- 不要输出额外解释。"""

NON_CS_RERANK_SYSTEM_PROMPT = """你是学术推荐精排助手。你会看到用户研究方向和一组候选论文。任务是严格区分“表面关键词命中但质量低/不够相关”的论文。

要求：
- 只返回 JSON 对象，格式为 {"results":[...]}
- results 是数组，每个元素包含：
  - index: 候选论文序号
  - relevance: 0 到 100 的整数，表示和用户研究方向的真实相关性
  - quality: 0 到 100 的整数，表示论文本身作为推荐对象的质量与值得阅读程度
  - reason: 一句简短中文理由
- 重点惩罚只靠宽泛词命中的论文，例如只碰到 generic terms 但主题并不一致
- 优先保留与用户 topics 在问题域、方法域、数据域上都更一致的论文
- 不要输出任何 JSON 之外的解释。"""

PROFILE_INPUT_PARSE_SYSTEM_PROMPT = """你是学术推荐技能的输入解析助手。请把用户的自然语言输入解析成结构化画像线索。

输出 JSON 对象，字段固定为：
- intent: "scholar" | "topic" | "mixed" | "unknown"
- scholar_name: 字符串，没有则输出空字符串
- scholar_org: 字符串，没有则输出空字符串
- topics: 字符串数组，没有则输出空数组
- free_text: 去掉已抽取 scholar/topic 后剩余的补充描述；如果没有则输出空字符串

规则：
- 如果用户主要是在介绍“我是某某，某机构”，应识别为 scholar 或 mixed
- scholar_name 要尽量是人名，不要把机构、职位、请求词塞进去
- scholar_org 要尽量是机构名
- topics 只保留真正研究方向，不要把人名、机构名当作 topics
- 如果无法可靠判断，就返回 unknown，并尽量保留 free_text
- 不要输出任何 JSON 之外的解释。"""

SCHOLAR_TERM_LABEL_SYSTEM_PROMPT = """你是 scholar profile 词项标注助手。请根据学者兴趣、核心主题、代表论文与候选词，判断每个词在推荐系统中应该扮演的角色。

输出 JSON 对象，字段固定为：
- labels: 数组

每个 label 元素包含：
- term: 原始词项
- role: 只能是 "scholar_specific" | "core_domain" | "broad_superordinate" | "method" | "auxiliary" | "noise"
- weight: 0 到 1.5 的浮点数，表示该词作为排序信号的建议强度。scholar_specific 通常最高，noise 最低。
- rationale: 一句简短中文说明

判定规则：
- scholar_specific: 明显指向该学者长期场景、实体、数据资产或专属问题，如 OAG、author disambiguation、academic graph mining
- core_domain: 稳定主方向，但不是该学者独有术语
- broad_superordinate: 上位词、覆盖面很广的任务词，如 Named Entity Recognition、Information Extraction
- method: 方法词、训练范式、评测词，如 Benchmark、Contrastive Learning、Self-supervised learning
- auxiliary: 历史旁支/辅助线索，可弱参考但不应主导排序
- noise: 明显句式碎片、无检索意义的词

要求：
- 不要编造输入中不存在的词
- 优先把泛词和方法词与核心方向区分开
- 只返回 JSON，不要输出额外解释。"""

MAX_PROMPT_AUTHOR_PROFILES = 2
PROMPT_AUTHOR_HINDEX_THRESHOLD = 20
PROMPT_AUTHOR_CITATIONS_THRESHOLD = 1000
USER_PROMPT_TEMPLATE = """
title: {title}
authors: {authors}
abstract: {abstract}
famous authors:
{known_authors}
comment: {comment}
"""
PROFILE_TOPIC_PROMPT_TEMPLATE = """
user_name: {user_name}
explicit_topics:
{explicit_topics}

seed_papers:
{seed_papers}
"""
KNOWN_AUTHOR_PROMPT = """{author_name}:
citation: {citation}
h-index: {h_index}
affiliation: {affiliation}
profile: {profile}
"""
SUMMARY_SECTION_LABELS = {
    "research_problem": "研究问题",
    "research_challenge": "研究挑战",
    "research_method": "研究方法",
    "experimental_results": "实验效果",
}
SUMMARY_REQUIRED_KEYS = ("research_problem", "research_challenge", "research_method")
SUMMARY_OPTIONAL_KEYS = ("experimental_results",)
SUMMARY_ALL_KEYS = SUMMARY_REQUIRED_KEYS + SUMMARY_OPTIONAL_KEYS


class SummaryGenerationError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class ProfileTopicGenerationError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class RerankGenerationError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class ProfileInputParseError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


class ScholarTermLabelError(RuntimeError):
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


def _normalize_client_error(exc: Exception) -> str:
    text = f"{exc.__class__.__name__}:{exc}".lower()
    if "timeout" in text or "timed out" in text:
        return "llm_timeout"
    return f"llm_client_error:{exc.__class__.__name__}"


def parse_model_json(raw_text: str, required_keys: set[str] | None = None) -> dict[str, Any]:
    raw_text = raw_text.strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", raw_text):
        try:
            parsed, _ = decoder.raw_decode(raw_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (required_keys is None or required_keys.issubset(parsed.keys())):
            return parsed
    raise ValueError("no JSON object found in model response")


def _create_openai_client(api_key: str, base_url: str, timeout_seconds: int):
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise SummaryGenerationError("openai package is not installed") from exc
    return OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=timeout_seconds,
        max_retries=0,
    )


def _call_model_raw_output(
    *,
    system_prompt: str,
    user_prompt: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
) -> str:
    client = _create_openai_client(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    raw_output = ""
    if not raw_output:
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            raw_output = response.choices[0].message.content or ""
        except Exception as exc:
            if hasattr(client, "responses"):
                try:
                    response = client.responses.create(model=model, input=messages)
                    raw_output = response.output_text
                except Exception as responses_exc:
                    raise SummaryGenerationError(_normalize_client_error(responses_exc), raw_output=raw_output) from responses_exc
            else:
                raise SummaryGenerationError(_normalize_client_error(exc), raw_output=raw_output) from exc
    return raw_output


def _require_string_list(value: Any, field_name: str, raw_output: str) -> list[str]:
    if not isinstance(value, list):
        raise SummaryGenerationError(f"model returned invalid {field_name}", raw_output=raw_output)
    items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise SummaryGenerationError(f"model returned invalid {field_name}", raw_output=raw_output)
        text = item.strip()
        if text:
            items.append(text)
    return items


def _require_string_list_or_scalar_string(value: Any, field_name: str, raw_output: str) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if field_name == "famous_authors" and isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                title = str(item.get("title") or "").strip()
                affiliation = str(item.get("affiliation") or "").strip()
                h_index = str(item.get("h_index") or item.get("hindex") or "").strip()
                research_area = str(item.get("research_area") or item.get("research") or "").strip()
                parts = [part for part in (title, affiliation, f"h-index {h_index}" if h_index else "", research_area) if part]
                if name:
                    normalized.append(f"{name}: {'，'.join(parts)}" if parts else name)
                    continue
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip())
        return normalized
    return _require_string_list(value, field_name, raw_output)


def _normalize_keywords(parsed_keywords: list[str]) -> list[str]:
    deduped: list[str] = []
    for keyword in parsed_keywords:
        text = keyword.strip()
        if text and text not in deduped:
            deduped.append(text)
        if len(deduped) == 4:
            break
    if len(deduped) < 2:
        raise SummaryGenerationError("model returned invalid keywords")
    return deduped[:4]


def _clean_summary_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def format_structured_summary(structured_summary: dict[str, str]) -> str:
    sections: list[str] = []
    for key in SUMMARY_ALL_KEYS:
        text = _clean_summary_text(structured_summary.get(key, ""))
        if not text:
            continue
        sections.append(f"{SUMMARY_SECTION_LABELS[key]}：{text}")
    return "\n".join(sections)


def normalize_structured_summary(value: Any, raw_output: str = "") -> dict[str, str]:
    if not isinstance(value, dict):
        raise SummaryGenerationError("model returned invalid structured_summary", raw_output=raw_output)
    normalized: dict[str, str] = {}
    for key in SUMMARY_ALL_KEYS:
        text = _clean_summary_text(value.get(key, ""))
        if text:
            normalized[key] = text
    missing_required = [key for key in SUMMARY_REQUIRED_KEYS if not normalized.get(key)]
    if missing_required:
        raise SummaryGenerationError("model returned invalid structured_summary", raw_output=raw_output)
    return normalized


def _prompt_author_profiles(paper: dict[str, Any]) -> list[dict[str, Any]]:
    profiles = [profile for profile in (paper.get("aminer_author_profiles") or []) if profile.get("name")]
    prompt_profiles = sorted(
        profiles,
        key=lambda profile: (
            -(int(profile.get("hindex") or 0)),
            -(int(profile.get("citations") or 0)),
            str(profile.get("name") or ""),
        ),
    )[:MAX_PROMPT_AUTHOR_PROFILES]
    return [
        profile
        for profile in prompt_profiles
        if int(profile.get("hindex") or 0) > PROMPT_AUTHOR_HINDEX_THRESHOLD
        or int(profile.get("citations") or 0) > PROMPT_AUTHOR_CITATIONS_THRESHOLD
    ]


def _normalize_famous_authors(parsed_authors: list[str]) -> list[str]:
    normalized: list[str] = []
    for text in parsed_authors:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            continue
        split_candidates = re.split(
            r"(?:\s+(?=(?:[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){1,5}|[\u4e00-\u9fff·]{2,20})[:：]))",
            compact,
        )
        for candidate in split_candidates:
            item = candidate.strip()
            if item:
                normalized.append(item)
            if len(normalized) == MAX_PROMPT_AUTHOR_PROFILES:
                break
        if len(normalized) == MAX_PROMPT_AUTHOR_PROFILES:
            break
    return normalized


def _prompt_honor_awards(profile: dict[str, Any]) -> list[str]:
    raw_honors = profile.get("honor_raw") or []
    if not isinstance(raw_honors, list):
        return []
    awards: list[str] = []
    for item in raw_honors:
        if not isinstance(item, dict):
            continue
        award = str(item.get("award") or "").strip()
        if award:
            awards.append(award)
    return awards


def _format_author_profile_text(profile: dict[str, Any]) -> str:
    awards = ", ".join(_prompt_honor_awards(profile))
    bio = str(profile.get("bio", "")).strip()
    profile_text = ", ".join(part for part in (awards, bio) if part)
    name = str(profile.get("name", "")).strip()
    return KNOWN_AUTHOR_PROMPT.format(
        author_name=name,
        citation=profile.get("citations", 0) or 0,
        h_index=profile.get("hindex", 0) or 0,
        affiliation=str(profile.get("affiliation", "")).strip(),
        profile=profile_text,
    ).strip()


def build_summary_prompt(paper: dict[str, Any]) -> str:
    known_authors = [
        _format_author_profile_text(profile)
        for profile in _prompt_author_profiles(paper)
        if profile.get("name")
    ]
    authors = [str(author).strip() for author in list(paper.get("authors") or []) if str(author).strip()]
    return USER_PROMPT_TEMPLATE.format(
        title=paper.get("title", ""),
        authors=", ".join(authors),
        abstract=paper.get("abstract", ""),
        comment=paper.get("aminer_comment", ""),
        known_authors="\n".join(known_authors),
    ).strip()


def build_profile_topic_prompt(profile: dict[str, Any]) -> str:
    explicit_topics = [str(item).strip() for item in list(profile.get("experts_topics") or profile.get("topics") or []) if str(item).strip()]
    seed_lines: list[str] = []
    for index, paper in enumerate(list(profile.get("seed_papers") or [])[:10], start=1):
        if not isinstance(paper, dict):
            continue
        title = str(paper.get("title", "")).strip()
        fields = ", ".join(str(item).strip() for item in list(paper.get("fields") or []) if str(item).strip())
        topics = ", ".join(str(item).strip() for item in list(paper.get("topics") or []) if str(item).strip())
        keywords = ", ".join(str(item).strip() for item in list(paper.get("keywords") or []) if str(item).strip())
        abstract = re.sub(r"\s+", " ", str(paper.get("abstract", "")).strip())[:500]
        seed_lines.append(
            f"{index}. title: {title}\nfields: {fields}\ntopics: {topics}\nkeywords: {keywords}\nabstract: {abstract}"
        )
    return PROFILE_TOPIC_PROMPT_TEMPLATE.format(
        user_name=str(profile.get("user_name") or profile.get("profile_name") or "").strip(),
        explicit_topics="\n".join(f"- {topic}" for topic in explicit_topics) or "- None",
        seed_papers="\n\n".join(seed_lines) or "None",
    ).strip()


def build_profile_input_parse_prompt(text: str) -> str:
    return f"user_input:\n{text.strip()}"


def build_non_cs_rerank_prompt(profile: dict[str, Any], papers: list[dict[str, Any]]) -> str:
    topics = [str(item).strip() for item in list(profile.get("topics") or []) if str(item).strip()]
    keywords = [str(item).strip() for item in list(profile.get("keywords") or []) if str(item).strip()]
    paper_blocks: list[str] = []
    for index, paper in enumerate(papers):
        authors = ", ".join(str(author).strip() for author in list(paper.get("authors") or []) if str(author).strip())
        matched_keywords = ", ".join(str(item).strip() for item in list(paper.get("matched_keywords") or []) if str(item).strip())
        matched_categories = ", ".join(str(item).strip() for item in list(paper.get("matched_categories") or []) if str(item).strip())
        paper_blocks.append(
            "\n".join(
                [
                    f"index: {index}",
                    f"title: {str(paper.get('title') or '').strip()}",
                    f"authors: {authors}",
                    f"abstract: {str(paper.get('abstract') or paper.get('summary') or '').strip()[:1200]}",
                    f"matched_keywords: {matched_keywords}",
                    f"matched_categories: {matched_categories}",
                    f"rule_score: {paper.get('recommendation_score', 0)}",
                ]
            )
        )
    return (
        "user_topics:\n"
        + ("\n".join(f"- {topic}" for topic in topics) or "- None")
        + "\n\nuser_keywords:\n"
        + ("\n".join(f"- {keyword}" for keyword in keywords[:12]) or "- None")
        + "\n\ncandidate_papers:\n"
        + "\n\n".join(paper_blocks)
    ).strip()


def llm_profile_topics(
    profile: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], str]:
    raw_output = _call_model_raw_output(
        system_prompt=PROFILE_TOPIC_SYSTEM_PROMPT,
        user_prompt=build_profile_topic_prompt(profile),
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    try:
        parsed = parse_model_json(raw_output, required_keys={"topics"})
    except Exception as exc:
        raise ProfileTopicGenerationError("model returned invalid JSON", raw_output=raw_output) from exc
    raw_topics = parsed.get("topics")
    if not isinstance(raw_topics, list):
        raise ProfileTopicGenerationError("model returned invalid topics", raw_output=raw_output)
    normalized: list[dict[str, Any]] = []
    for item in raw_topics:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        rationale = re.sub(r"\s+", " ", str(item.get("rationale") or "").strip())
        raw_keywords = item.get("keywords")
        if isinstance(raw_keywords, str):
            keywords = [raw_keywords.strip()] if raw_keywords.strip() else []
        elif isinstance(raw_keywords, list):
            keywords = [str(keyword).strip() for keyword in raw_keywords if str(keyword).strip()]
        else:
            keywords = []
        if not name:
            continue
        normalized.append({"name": name, "keywords": keywords[:5], "rationale": rationale})
        if len(normalized) == 5:
            break
    if not normalized:
        raise ProfileTopicGenerationError("model returned invalid topics", raw_output=raw_output)
    return normalized, raw_output


def llm_parse_profile_input(
    text: str,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], str]:
    raw_output = _call_model_raw_output(
        system_prompt=PROFILE_INPUT_PARSE_SYSTEM_PROMPT,
        user_prompt=build_profile_input_parse_prompt(text),
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    try:
        parsed = parse_model_json(raw_output, required_keys={"intent"})
    except Exception as exc:
        raise ProfileInputParseError("model returned invalid JSON", raw_output=raw_output) from exc

    intent = str(parsed.get("intent") or "").strip().lower()
    if intent not in {"scholar", "topic", "mixed", "unknown"}:
        raise ProfileInputParseError("model returned invalid intent", raw_output=raw_output)

    scholar_name = re.sub(r"\s+", " ", str(parsed.get("scholar_name") or "")).strip()
    scholar_org = re.sub(r"\s+", " ", str(parsed.get("scholar_org") or "")).strip()
    raw_topics = parsed.get("topics") or []
    if isinstance(raw_topics, str):
        topics = [raw_topics.strip()] if raw_topics.strip() else []
    elif isinstance(raw_topics, list):
        topics = [str(item).strip() for item in raw_topics if str(item).strip()]
    else:
        topics = []
    free_text = re.sub(r"\s+", " ", str(parsed.get("free_text") or "")).strip()

    return (
        {
            "intent": intent,
            "scholar_name": scholar_name,
            "scholar_org": scholar_org,
            "topics": topics,
            "free_text": free_text,
        },
        raw_output,
    )


def build_scholar_term_label_prompt(payload: dict[str, Any]) -> str:
    candidate_terms = [str(item).strip() for item in list(payload.get("candidate_terms") or []) if str(item).strip()]
    resolved_interests = [str(item).strip() for item in list(payload.get("resolved_interests") or []) if str(item).strip()]
    core_topics = [item for item in list(payload.get("core_topics") or []) if isinstance(item, dict)]
    seed_papers = [item for item in list(payload.get("seed_papers") or []) if isinstance(item, dict)]
    core_blocks: list[str] = []
    for topic in core_topics[:6]:
        name = str(topic.get("name") or "").strip()
        keywords = [str(item).strip() for item in list(topic.get("keywords") or []) if str(item).strip()]
        if not name:
            continue
        core_blocks.append(f"- {name}" + (f" | keywords: {', '.join(keywords[:4])}" if keywords else ""))
    seed_blocks: list[str] = []
    for paper in seed_papers[:5]:
        title = str(paper.get("title") or "").strip()
        keywords = [str(item).strip() for item in list(paper.get("keywords") or []) if str(item).strip()]
        if not title:
            continue
        seed_blocks.append(f"- {title}" + (f" | keywords: {', '.join(keywords[:5])}" if keywords else ""))
    return (
        f"scholar_name: {str(payload.get('scholar_name') or '').strip()}\n"
        f"scholar_org: {str(payload.get('scholar_org') or '').strip()}\n\n"
        "resolved_interests:\n"
        + ("\n".join(f"- {item}" for item in resolved_interests) or "- None")
        + "\n\ncore_topics:\n"
        + ("\n".join(core_blocks) or "- None")
        + "\n\nseed_papers:\n"
        + ("\n".join(seed_blocks) or "- None")
        + "\n\ncandidate_terms:\n"
        + ("\n".join(f"- {item}" for item in candidate_terms) or "- None")
    ).strip()


def llm_label_scholar_terms(
    payload: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], str]:
    try:
        raw_output = _call_model_raw_output(
            system_prompt=SCHOLAR_TERM_LABEL_SYSTEM_PROMPT,
            user_prompt=build_scholar_term_label_prompt(payload),
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
    except ScholarTermLabelError:
        raise
    except Exception as exc:
        raw_output = getattr(exc, "raw_output", "") if exc is not None else ""
        raise ScholarTermLabelError(str(exc), raw_output=raw_output) from exc
    try:
        parsed = parse_model_json(raw_output, required_keys={"labels"})
    except Exception as exc:
        raise ScholarTermLabelError("model returned invalid JSON", raw_output=raw_output) from exc
    labels = parsed.get("labels")
    if not isinstance(labels, list):
        raise ScholarTermLabelError("model returned invalid labels", raw_output=raw_output)
    allowed_roles = {"scholar_specific", "core_domain", "broad_superordinate", "method", "auxiliary", "noise"}
    normalized: list[dict[str, Any]] = []
    for item in labels:
        if not isinstance(item, dict):
            continue
        term = re.sub(r"\s+", " ", str(item.get("term") or "").strip())
        role = str(item.get("role") or "").strip()
        rationale = re.sub(r"\s+", " ", str(item.get("rationale") or "").strip())
        if not term or role not in allowed_roles:
            continue
        try:
            weight = float(item.get("weight"))
        except (TypeError, ValueError):
            weight = 0.0
        normalized.append(
            {
                "term": term,
                "role": role,
                "weight": max(0.0, min(weight, 1.5)),
                "rationale": rationale,
            }
        )
    if not normalized:
        raise ScholarTermLabelError("model returned invalid labels", raw_output=raw_output)
    return normalized, raw_output


def llm_rerank_non_cs(
    profile: dict[str, Any],
    papers: list[dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> tuple[list[dict[str, Any]], str]:
    raw_output = _call_model_raw_output(
        system_prompt=NON_CS_RERANK_SYSTEM_PROMPT,
        user_prompt=build_non_cs_rerank_prompt(profile, papers),
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    try:
        parsed = parse_model_json(raw_output, required_keys={"results"})
    except Exception as exc:
        raise RerankGenerationError("model returned invalid JSON", raw_output=raw_output) from exc
    results = parsed.get("results")
    if not isinstance(results, list):
        raise RerankGenerationError("model returned invalid results", raw_output=raw_output)
    normalized: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
            relevance = max(0, min(100, int(item.get("relevance"))))
            quality = max(0, min(100, int(item.get("quality"))))
        except (TypeError, ValueError):
            continue
        reason = re.sub(r"\s+", " ", str(item.get("reason") or "").strip())
        normalized.append(
            {
                "index": index,
                "relevance": relevance,
                "quality": quality,
                "reason": reason,
            }
        )
    if not normalized:
        raise RerankGenerationError("model returned invalid results", raw_output=raw_output)
    return normalized, raw_output


def llm_summary(
    paper: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
) -> tuple[list[str], str, list[str], dict[str, str]]:
    raw_output = _call_model_raw_output(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_summary_prompt(paper),
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    try:
        parsed = parse_model_json(raw_output, required_keys={"keywords", "structured_summary", "famous_authors"})
    except Exception:
        try:
            parsed = parse_model_json(raw_output, required_keys={"keywords", "summary", "famous_authors"})
        except Exception as exc:
            try:
                parsed = parse_model_json(raw_output)
            except Exception as inner_exc:
                raise SummaryGenerationError("model returned invalid JSON", raw_output=raw_output) from inner_exc
    keywords = _normalize_keywords(_require_string_list(parsed.get("keywords"), "keywords", raw_output))
    famous_authors = _normalize_famous_authors(
        _require_string_list_or_scalar_string(parsed.get("famous_authors"), "famous_authors", raw_output)
    )
    structured_summary: dict[str, str] = {}
    if "structured_summary" in parsed:
        raw_structured_summary = parsed.get("structured_summary")
        if not isinstance(raw_structured_summary, dict):
            raise SummaryGenerationError("model returned invalid structured_summary", raw_output=raw_output)
        structured_summary = {
            key: _clean_summary_text(value)
            for key, value in raw_structured_summary.items()
            if key in SUMMARY_ALL_KEYS and _clean_summary_text(value)
        }
        has_required_keys = all(structured_summary.get(key) for key in SUMMARY_REQUIRED_KEYS)
        if has_required_keys:
            summary = format_structured_summary(structured_summary)
        else:
            summary_value = parsed.get("summary")
            summary = summary_value.strip() if isinstance(summary_value, str) and summary_value.strip() else ""
    else:
        summary_value = parsed.get("summary")
        if not isinstance(summary_value, str) or not summary_value.strip():
            raise SummaryGenerationError("model returned invalid summary", raw_output=raw_output)
        summary = summary_value.strip()
        if not summary.startswith("本文"):
            summary = f"本文{summary}"
    return keywords, summary, famous_authors, structured_summary
