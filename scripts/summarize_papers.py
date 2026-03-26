#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import first_nonempty, merge_payload_status, read_json, summarize_statuses, utc_now_iso, write_json
from scripts.llm_client import SummaryGenerationError, format_structured_summary, llm_summary, parse_model_json

DEFAULT_LLM_MODEL = "gpt-5-mini"
DEFAULT_LLM_TIMEOUT_SECONDS = 30
DEFAULT_LLM_MAX_CONCURRENT_REQUESTS = 10
DEFAULT_LLM_RETRY_ATTEMPTS = 0

SUMMARY_REQUIRED_KEYS = ("research_problem", "research_challenge", "research_method")
RESULT_HINT_PATTERN = re.compile(
    r"\b(experiment|experiments|experimental|benchmark|evaluation|results?|outperform|improv(?:e|es|ed)|accuracy|f1|bleu)\b",
    re.IGNORECASE,
)


def fallback_keywords(paper: dict[str, Any]) -> list[str]:
    subjects = str(paper.get("subjects", ""))
    fragments = [fragment.strip() for fragment in re.split(r"[,;/]", subjects) if fragment.strip()]
    if fragments:
        return fragments[:3]
    title_words = [word for word in re.findall(r"[A-Za-z]{4,}", str(paper.get("title", "")))][:3]
    return title_words or ["arXiv"]


def _split_sentences(text: str) -> list[str]:
    return [fragment.strip() for fragment in re.split(r"(?<=[。！？.!?])\s+|[。！？]", text) if fragment.strip()]


def _clip_text(text: str, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit].strip()


def _normalize_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _paper_has_usable_abstract(paper: dict[str, Any]) -> bool:
    title = _normalize_for_compare(str(paper.get("title", "")))
    abstract = re.sub(r"\s+", " ", str(paper.get("abstract", "")).strip())
    if not abstract:
        return False
    normalized_abstract = abstract.casefold()
    if title and normalized_abstract == title:
        return False
    if len(abstract) < 80:
        return False
    token_count = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", abstract))
    return token_count >= 12


def _missing_abstract_structured_summary(paper: dict[str, Any], keywords: list[str]) -> dict[str, str]:
    topic_hint = keywords[0] if keywords else "当前论文"
    return {
        "research_problem": f"摘要缺失，暂无法准确概括 {topic_hint} 的具体研究问题",
        "research_challenge": "当前记录未提供有效 abstract，现有文本不足以支撑可靠总结",
        "research_method": "建议先补齐 arXiv 或 AMiner 摘要，再生成结构化论文小结",
    }


def _build_fallback_structured_summary(paper: dict[str, Any], keywords: list[str], summary_text: str = "") -> dict[str, str]:
    title = str(paper.get("title", "")).strip()
    if not _paper_has_usable_abstract(paper):
        return _missing_abstract_structured_summary(paper, keywords)
    abstract = first_nonempty(str(paper.get("abstract", "")).strip(), summary_text.strip(), title)
    sentences = _split_sentences(abstract)
    problem_source = title or (sentences[0] if sentences else abstract)
    challenge_source = sentences[1] if len(sentences) > 1 else ""
    method_source = sentences[0] if sentences else abstract
    structured_summary = {
        "research_problem": _clip_text(problem_source or f"围绕{keywords[0]}相关问题展开"),
        "research_challenge": _clip_text(challenge_source or "摘要未明确说明"),
        "research_method": _clip_text(method_source or summary_text or abstract),
    }
    experimental_source = next((sentence for sentence in sentences if RESULT_HINT_PATTERN.search(sentence)), "")
    if experimental_source:
        structured_summary["experimental_results"] = _clip_text(experimental_source)
    return structured_summary


def _coerce_summary_result(
    result: tuple[Any, ...],
    paper: dict[str, Any],
    *,
    fallback_on_invalid: bool = True,
) -> tuple[list[str], str, list[str], dict[str, str]]:
    if len(result) == 4:
        keywords, summary, famous_authors, structured_summary = result
    elif len(result) == 3:
        keywords, summary, famous_authors = result
        structured_summary = {}
    else:
        raise SummaryGenerationError("model returned invalid summary payload")

    keywords_list = [str(item).strip() for item in list(keywords or []) if str(item).strip()]
    summary_text = str(summary or "").strip()
    famous_authors_list = [str(item).strip() for item in list(famous_authors or []) if str(item).strip()]

    normalized_structured: dict[str, str] = {}
    used_fallback_structured = False
    if isinstance(structured_summary, dict):
        normalized_structured = {key: str(value).strip() for key, value in structured_summary.items() if str(value).strip()}
    missing_required = [key for key in SUMMARY_REQUIRED_KEYS if not normalized_structured.get(key)]
    if missing_required:
        if not fallback_on_invalid:
            raise SummaryGenerationError("model returned invalid structured_summary")
        normalized_structured = _build_fallback_structured_summary(paper, keywords_list or fallback_keywords(paper), summary_text)
        used_fallback_structured = True
    if used_fallback_structured or not summary_text:
        summary_text = format_structured_summary(normalized_structured)
    return keywords_list, summary_text, famous_authors_list, normalized_structured


def fallback_summary(paper: dict[str, Any]) -> tuple[list[str], str, list[str], dict[str, str]]:
    keywords = fallback_keywords(paper)
    abstract = first_nonempty(str(paper.get("abstract", "")), "")
    structured_summary = _build_fallback_structured_summary(paper, keywords[:4], abstract[:120].strip())
    summary = format_structured_summary(structured_summary)
    return keywords[:4], summary, [], structured_summary


def _is_retryable_summary_error(exc: SummaryGenerationError) -> bool:
    reason = str(exc).strip()
    if not reason:
        return False
    if reason in {"missing_api_key", "openai package is not installed"}:
        return False
    if reason.startswith("llm_"):
        return True
    return reason.startswith("model returned invalid ")


def _format_summary_reason(reason: str, retry_count: int) -> str:
    if retry_count <= 0:
        return reason
    return f"{reason}（已重试{retry_count}次）"


def _summarize_one(
    paper: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
    retry_attempts: int,
) -> dict[str, Any]:
    reason = ""
    raw_output = ""
    retry_count = 0
    retry_reasons: list[str] = []
    existing_famous_authors = [str(item).strip() for item in list(paper.get("famous_authors") or []) if str(item).strip()]
    if not _paper_has_usable_abstract(paper):
        keywords, summary, famous_authors, structured_summary = fallback_summary(paper)
        if not famous_authors:
            famous_authors = existing_famous_authors
        return {
            **paper,
            "summary_status": "degraded",
            "summary_reason": "missing_effective_abstract",
            "summary_retry_count": 0,
            "summary_retry_reasons": [],
            "keywords": keywords,
            "structured_summary": structured_summary,
            "summary": summary,
            "famous_authors": famous_authors,
            "summary_raw_output": "",
        }
    try:
        if api_key.strip():
            while True:
                try:
                    keywords, summary, famous_authors, structured_summary = _coerce_summary_result(
                        llm_summary(
                            paper,
                            api_key=api_key,
                            base_url=base_url,
                            model=model,
                            timeout_seconds=timeout_seconds,
                        ),
                        paper,
                        fallback_on_invalid=True,
                    )
                    break
                except SummaryGenerationError as exc:
                    if retry_count < retry_attempts and _is_retryable_summary_error(exc):
                        retry_count += 1
                        retry_reasons.append(str(exc))
                        raw_output = exc.raw_output
                        continue
                    raise
            status = "success"
        else:
            raise SummaryGenerationError("missing_api_key")
    except SummaryGenerationError as exc:
        keywords, summary, famous_authors, structured_summary = fallback_summary(paper)
        status = "degraded"
        reason = _format_summary_reason(str(exc), retry_count)
        raw_output = exc.raw_output or raw_output
    except Exception as exc:
        keywords, summary, famous_authors, structured_summary = fallback_summary(paper)
        status = "degraded"
        reason = f"unexpected_error:{exc.__class__.__name__}"
    if not famous_authors:
        famous_authors = existing_famous_authors
    return {
        **paper,
        "summary_status": status,
        "summary_reason": reason,
        "summary_retry_count": retry_count,
        "summary_retry_reasons": retry_reasons,
        "keywords": keywords,
        "structured_summary": structured_summary,
        "summary": summary,
        "famous_authors": famous_authors,
        "summary_raw_output": raw_output,
    }


def _build_llm_candidates(
    *,
    api_key: str,
    base_url: str,
    model: str,
    timeout_seconds: int,
    fallback_api_key: str = "",
    fallback_base_url: str = "",
    fallback_model: str = "",
    fallback_timeout_seconds: int | None = None,
) -> list[dict[str, Any]]:
    candidates = [
        {
            "label": "primary",
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "timeout_seconds": timeout_seconds,
        }
    ]
    if str(fallback_api_key or "").strip():
        candidates.append(
            {
                "label": "fallback",
                "api_key": fallback_api_key,
                "base_url": fallback_base_url,
                "model": fallback_model or model,
                "timeout_seconds": int(fallback_timeout_seconds or timeout_seconds),
            }
        )
    return candidates


def _build_summary_payload(
    enriched_payload: dict[str, Any],
    *,
    summarized: list[dict[str, Any]],
    statuses: list[str],
    debug_outputs: list[dict[str, str]],
) -> dict[str, Any]:
    return merge_payload_status(
        {
            "status": summarize_statuses(statuses or ["degraded"]),
            "generated_at": utc_now_iso(),
            "paper_count": len(summarized),
            "debug": {"failed_outputs": debug_outputs},
            "profile_topics": list(enriched_payload.get("profile_topics") or []),
            "profile_name": str(enriched_payload.get("profile_name") or ""),
            "profile_source": str(enriched_payload.get("profile_source") or ""),
            "papers": summarized,
        },
        enriched_payload,
    )


def _summarize_one_with_fallback(
    paper: dict[str, Any],
    *,
    candidates: list[dict[str, Any]],
    retry_attempts: int,
) -> dict[str, Any]:
    last_result: dict[str, Any] | None = None
    for candidate in candidates:
        result = _summarize_one(
            paper,
            api_key=str(candidate.get("api_key") or ""),
            base_url=str(candidate.get("base_url") or ""),
            model=str(candidate.get("model") or DEFAULT_LLM_MODEL),
            timeout_seconds=int(candidate.get("timeout_seconds") or DEFAULT_LLM_TIMEOUT_SECONDS),
            retry_attempts=retry_attempts,
        )
        result["summary_provider"] = str(candidate.get("label") or "")
        if str(result.get("summary_status")) == "success":
            return result
        last_result = result
    return last_result or _summarize_one(
        paper,
        api_key="",
        base_url="",
        model=DEFAULT_LLM_MODEL,
        timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
        retry_attempts=retry_attempts,
    )


def summarize_papers(
    enriched_payload: dict[str, Any],
    api_key: str = "",
    base_url: str = "",
    model: str = DEFAULT_LLM_MODEL,
    timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS,
    fallback_api_key: str = "",
    fallback_base_url: str = "",
    fallback_model: str = DEFAULT_LLM_MODEL,
    fallback_timeout_seconds: int | None = None,
    max_concurrent_requests: int = DEFAULT_LLM_MAX_CONCURRENT_REQUESTS,
    retry_attempts: int = DEFAULT_LLM_RETRY_ATTEMPTS,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    papers = enriched_payload.get("papers", [])
    summarized_results: list[dict[str, Any] | None] = [None] * len(papers)
    statuses: list[str] = []
    debug_outputs: list[dict[str, str]] = []
    batch_size = max(1, max_concurrent_requests)
    llm_candidates = _build_llm_candidates(
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        fallback_api_key=fallback_api_key,
        fallback_base_url=fallback_base_url,
        fallback_model=fallback_model,
        fallback_timeout_seconds=fallback_timeout_seconds,
    )

    def _ordered_results() -> list[dict[str, Any]]:
        return [result for result in summarized_results if result is not None]

    def _checkpoint() -> None:
        if checkpoint_path is None:
            return
        write_json(
            checkpoint_path,
            _build_summary_payload(
                enriched_payload,
                summarized=_ordered_results(),
                statuses=statuses,
                debug_outputs=debug_outputs,
            ),
        )

    def _store_result(index: int, result: dict[str, Any]) -> None:
        statuses.append(str(result["summary_status"]))
        raw_output = str(result.get("summary_raw_output", ""))
        if raw_output:
            debug_outputs.append({"arxiv_id": str(result.get("arxiv_id", "")), "raw_output": raw_output})
        summarized_results[index] = result
        _checkpoint()

    for start in range(0, len(papers), batch_size):
        batch = papers[start : start + batch_size]
        if not any(str(candidate.get("api_key") or "").strip() for candidate in llm_candidates):
            for offset, paper in enumerate(batch):
                result = _summarize_one_with_fallback(
                    paper,
                    candidates=llm_candidates,
                    retry_attempts=retry_attempts,
                )
                _store_result(start + offset, result)
        elif batch_size == 1:
            result = _summarize_one_with_fallback(
                    batch[0],
                    candidates=llm_candidates,
                    retry_attempts=retry_attempts,
                )
            _store_result(start, result)
        else:
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_index = {
                    executor.submit(
                        _summarize_one_with_fallback,
                        paper,
                        candidates=llm_candidates,
                        retry_attempts=retry_attempts,
                    ): start + index
                    for index, paper in enumerate(batch)
                }
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    _store_result(index, future.result())
    return _build_summary_payload(
        enriched_payload,
        summarized=_ordered_results(),
        statuses=statuses,
        debug_outputs=debug_outputs,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize papers with optional LLM support.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_LLM_TIMEOUT_SECONDS)
    parser.add_argument("--fallback-api-key", default="")
    parser.add_argument("--fallback-base-url", default="")
    parser.add_argument("--fallback-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--fallback-timeout-seconds", type=int, default=DEFAULT_LLM_TIMEOUT_SECONDS)
    parser.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_LLM_MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--retry-attempts", type=int, default=DEFAULT_LLM_RETRY_ATTEMPTS)
    args = parser.parse_args()

    payload = summarize_papers(
        read_json(args.input),
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
        fallback_api_key=args.fallback_api_key,
        fallback_base_url=args.fallback_base_url,
        fallback_model=args.fallback_model,
        fallback_timeout_seconds=args.fallback_timeout_seconds,
        max_concurrent_requests=args.max_concurrent_requests,
        retry_attempts=args.retry_attempts,
        checkpoint_path=args.output,
    )
    write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
