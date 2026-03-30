#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.dispatch_feishu_messages import send_text_via_route
from scripts.llm_client import llm_parse_profile_input


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _split_topics(text: str) -> list[str]:
    pieces = re.split(r"[,，;/；、\n]+", text)
    topics: list[str] = []
    for piece in pieces:
        topic = _clean_text(piece)
        if topic and topic not in topics:
            topics.append(topic)
    return topics


def _split_papers(text: str) -> list[str]:
    pieces = re.split(r"[|\n;；]+", text)
    papers: list[str] = []
    for piece in pieces:
        paper = _clean_text(piece)
        if paper and paper not in papers:
            papers.append(paper)
    return papers


def _extract_command_text(raw_text: str) -> str:
    lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.search(r"(/(?:skill\s+)?aminer[-_]rec5\b.*)$", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return str(raw_text or "")


FIELD_LABELS = {
    "aminer_user_id": ["aminer_user_id"],
    "topics": ["topics", "topic", "方向", "研究方向"],
    "scholar_name": ["scholar", "name", "author", "学者", "作者"],
    "scholar_org": ["org", "organization", "affiliation", "机构", "单位"],
    "paper_titles": ["paper", "papers", "代表作", "论文"],
    "papers_file": ["papers_file", "source_file", "profile_file", "文件", "路径"],
}


GENERIC_REQUEST_PATTERNS = [
    r"帮我推荐(?:一下)?论文",
    r"推荐(?:一下)?论文",
    r"推荐一些论文",
    r"给我推荐(?:一下)?论文",
    r"推荐最近论文",
    r"想看论文",
]

ORG_HINT_PATTERNS = (
    r"大学",
    r"学院",
    r"研究院",
    r"研究所",
    r"实验室",
    r"中心",
    r"University",
    r"College",
    r"Institute",
    r"Laboratory",
    r"Lab\b",
    r"School",
    r"Department",
)


MAX_TOPICS = 8
MAX_TOPIC_LENGTH = 80
MAX_PAPER_TITLES = 8
MAX_PAPER_TITLE_LENGTH = 300
MAX_SCHOLAR_NAME_LENGTH = 80
MAX_SCHOLAR_ORG_LENGTH = 160
MAX_FREE_TEXT_LENGTH = 600
MAX_TARGET_LENGTH = 160
MAX_ACCOUNT_ID_LENGTH = 64
ALLOWED_PAPERS_FILE_SUFFIXES = {".json"}


def _capture_field(command_body: str, field_name: str) -> str:
    labels = FIELD_LABELS[field_name]
    all_labels = [re.escape(label) for values in FIELD_LABELS.values() for label in values]
    pattern = rf"(?:{'|'.join(re.escape(label) for label in labels)})\s*[:：]\s*(.+?)(?=\s*(?:{'|'.join(all_labels)})\s*[:：]|$)"
    match = re.search(pattern, command_body, flags=re.IGNORECASE | re.S)
    return _clean_text(match.group(1)) if match else ""


def _truncate_text(value: Any, max_length: int) -> str:
    cleaned = _clean_text(value)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].strip()


def _normalize_topics_for_interface(values: list[Any]) -> list[str]:
    topics: list[str] = []
    for value in list(values or []):
        topic = _truncate_text(value, MAX_TOPIC_LENGTH)
        if topic and topic not in topics:
            topics.append(topic)
        if len(topics) >= MAX_TOPICS:
            break
    return topics


def _normalize_paper_titles_for_interface(values: list[Any]) -> list[str]:
    paper_titles: list[str] = []
    for value in list(values or []):
        paper_title = _truncate_text(value, MAX_PAPER_TITLE_LENGTH)
        if paper_title and paper_title not in paper_titles:
            paper_titles.append(paper_title)
        if len(paper_titles) >= MAX_PAPER_TITLES:
            break
    return paper_titles


def _resolve_interface_papers_file(base_dir: Path, path_text: str) -> str:
    cleaned = _clean_text(path_text)
    if not cleaned:
        return ""

    candidate = Path(cleaned).expanduser()
    resolved_base_dir = base_dir.resolve()
    resolved_candidate = (resolved_base_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_base_dir)
    except ValueError as exc:
        raise ValueError("papers_file_outside_base_dir") from exc

    if resolved_candidate.suffix.lower() not in ALLOWED_PAPERS_FILE_SUFFIXES:
        raise ValueError("unsupported_papers_file")
    return str(resolved_candidate)


def _normalize_interface_payload(parsed: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    normalized = dict(parsed)
    raw_uid = _clean_text(parsed.get("raw_aminer_user_id"))
    if raw_uid and not re.fullmatch(r"[0-9a-fA-F]{24}", raw_uid):
        raise ValueError("invalid_aminer_user_id")

    normalized["aminer_user_id"] = _clean_text(parsed.get("aminer_user_id"))
    normalized["topics"] = _normalize_topics_for_interface(list(parsed.get("topics") or []))
    normalized["scholar_name"] = _truncate_text(parsed.get("scholar_name"), MAX_SCHOLAR_NAME_LENGTH)
    normalized["scholar_org"] = _truncate_text(parsed.get("scholar_org"), MAX_SCHOLAR_ORG_LENGTH)
    normalized["paper_titles"] = _normalize_paper_titles_for_interface(list(parsed.get("paper_titles") or []))
    normalized["papers_file"] = _resolve_interface_papers_file(base_dir, str(parsed.get("papers_file") or ""))
    normalized["free_text"] = _truncate_text(parsed.get("free_text"), MAX_FREE_TEXT_LENGTH)
    return normalized


def _strip_explicit_fields(command_body: str) -> str:
    all_labels = [re.escape(label) for values in FIELD_LABELS.values() for label in values]
    pattern = rf"(?:{'|'.join(all_labels)})\s*[:：]\s*.+?(?=\s*(?:{'|'.join(all_labels)})\s*[:：]|$)"
    cleaned = re.sub(pattern, " ", command_body, flags=re.IGNORECASE | re.S)
    return _clean_text(cleaned)


def _remove_generic_request_phrases(text: str) -> str:
    cleaned = str(text or "")
    for pattern in GENERIC_REQUEST_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[。！？!?.，,；;、\s]+", " ", cleaned)
    return _clean_text(cleaned)


def _infer_scholar_from_free_text(text: str) -> tuple[str, str, str]:
    normalized = _clean_text(text)
    if not normalized:
        return "", "", ""

    patterns = [
        r"^我(?:是|叫)\s*(?P<name>[^，,。；;、\s]{2,20})\s*[，,、]\s*(?P<org>[^。；;，,]{2,60})",
        r"^本人(?:是)?\s*(?P<name>[^，,。；;、\s]{2,20})\s*[，,、]\s*(?P<org>[^。；;，,]{2,60})",
        r"^我是\s*(?P<org>[^，,。；;]{2,60})\s*的\s*(?P<name>[^，,。；;、\s]{2,20})",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        scholar_name = _clean_text(match.groupdict().get("name"))
        scholar_org = _clean_text(match.groupdict().get("org"))
        if not scholar_name:
            continue
        residual = _clean_text(normalized[match.end() :])
        residual = _remove_generic_request_phrases(residual)
        scholar_org = re.split(r"[。！？!?.；;]", scholar_org, maxsplit=1)[0].strip()
        scholar_org = _remove_generic_request_phrases(scholar_org)
        return scholar_name, scholar_org, residual

    bare_match = re.search(
        r"^(?P<name>[A-Za-z][A-Za-z .'-]{1,60}|[\u4e00-\u9fff·]{2,20})\s*[，,、]\s*(?P<org>[^。；;，,\n]{2,80})",
        normalized,
        flags=re.IGNORECASE,
    )
    if bare_match:
        scholar_name = _clean_text(bare_match.group("name"))
        scholar_org = _clean_text(bare_match.group("org"))
        org_hint_pattern = "|".join(ORG_HINT_PATTERNS)
        if scholar_name and scholar_org and re.search(org_hint_pattern, scholar_org, flags=re.IGNORECASE):
            residual = _clean_text(normalized[bare_match.end() :])
            residual = _remove_generic_request_phrases(residual)
            scholar_org = re.split(r"[。！？!?.；;]", scholar_org, maxsplit=1)[0].strip()
            scholar_org = _remove_generic_request_phrases(scholar_org)
            return scholar_name, scholar_org, residual
    return "", "", normalized


def parse_trigger_text(text: str) -> dict[str, Any]:
    raw_text = str(text or "")
    command_text = _extract_command_text(raw_text)
    normalized = _clean_text(command_text)
    body = re.sub(r"^/(skill\s+)?aminer[-_]rec5\b", "", command_text, flags=re.IGNORECASE).strip()
    uid_match = re.search(r"aminer_user_id\s*[:：]\s*([0-9a-fA-F]{24})", body, flags=re.IGNORECASE)
    uid = uid_match.group(1) if uid_match else ""
    scholar_name = _capture_field(body, "scholar_name")
    scholar_org = _capture_field(body, "scholar_org")
    free_text = _strip_explicit_fields(re.sub(r"aminer_user_id\s*[:：]\s*[0-9a-fA-F]{24}", " ", body, flags=re.IGNORECASE))
    if not scholar_name:
        inferred_name, inferred_org, residual = _infer_scholar_from_free_text(free_text)
        if inferred_name:
            scholar_name = inferred_name
            if inferred_org and not scholar_org:
                scholar_org = inferred_org
            free_text = residual

    return {
        "raw_text": raw_text,
        "command_text": command_text,
        "raw_aminer_user_id": _capture_field(body, "aminer_user_id"),
        "aminer_user_id": uid,
        "topics": _split_topics(_capture_field(body, "topics")),
        "scholar_name": scholar_name,
        "scholar_org": scholar_org,
        "paper_titles": _split_papers(_capture_field(body, "paper_titles")),
        "papers_file": _capture_field(body, "papers_file"),
        "free_text": free_text,
        "is_trigger": bool(re.search(r"^/(skill\s+)?aminer[-_]rec5\b", normalized, flags=re.IGNORECASE)),
    }


def _load_config(base_dir: Path, config_path: Path | None) -> dict[str, Any]:
    if config_path and config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return _merge_llm_config(config)
    default_path = base_dir / "config.example.yaml"
    if default_path.exists():
        config = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
        return _merge_llm_config(config)
    return _merge_llm_config({})


def _load_openclaw_runtime_config() -> dict[str, Any]:
    explicit = _clean_text(os.getenv("OPENCLAW_CONFIG_PATH"))
    config_path = Path(explicit).expanduser() if explicit else _resolve_openclaw_home() / "openclaw.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    providers = payload.get("models", {}).get("providers", {})
    primary_model = _clean_text(payload.get("agents", {}).get("defaults", {}).get("model", {}).get("primary"))
    provider_name = primary_model.split("/", 1)[0] if "/" in primary_model else ""
    model_name = primary_model.split("/", 1)[1] if "/" in primary_model else primary_model
    provider_config = providers.get(provider_name) if provider_name else {}
    if not isinstance(provider_config, dict):
        provider_config = {}
    api_key = _clean_text(provider_config.get("apiKey"))
    base_url = _clean_text(provider_config.get("baseUrl"))
    if not model_name:
        models = provider_config.get("models") or []
        if isinstance(models, list) and models:
            first_model = models[0]
            if isinstance(first_model, dict):
                model_name = _clean_text(first_model.get("id") or first_model.get("name"))
            else:
                model_name = _clean_text(first_model)
    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model_name,
    }


def _resolve_openclaw_home() -> Path:
    explicit = _clean_text(os.getenv("OPENCLAW_HOME"))
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".openclaw"


def _merge_llm_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(config or {})
    llm_config = dict(merged.get("llm") or {})
    discovered = _load_openclaw_runtime_config()
    placeholder_models = {"", "gpt-5-mini", "gpt-5", "gpt-4.1-mini"}
    if not _clean_text(llm_config.get("api_key")) and _clean_text(discovered.get("api_key")):
        llm_config["api_key"] = discovered["api_key"]
    if not _clean_text(llm_config.get("base_url")) and _clean_text(discovered.get("base_url")):
        llm_config["base_url"] = discovered["base_url"]
    current_model = _clean_text(llm_config.get("model"))
    if current_model in placeholder_models and _clean_text(discovered.get("model")):
        llm_config["model"] = discovered["model"]
    if llm_config:
        merged["llm"] = llm_config
    return merged


def _llm_config_candidates(config: dict[str, Any]) -> list[dict[str, str]]:
    llm_config = dict(config.get("llm") or {})
    parse_timeout = min(int(llm_config.get("input_parse_timeout_seconds") or llm_config.get("timeout_seconds") or 30), 8)
    candidates = [
        {
            "api_key": _clean_text(llm_config.get("api_key")),
            "base_url": _clean_text(llm_config.get("base_url")),
            "model": _clean_text(llm_config.get("model")),
            "timeout_seconds": str(max(parse_timeout, 3)),
        }
    ]
    fallback = llm_config.get("fallback") if isinstance(llm_config.get("fallback"), dict) else {}
    if fallback:
        candidates.append(
            {
                "api_key": _clean_text(fallback.get("api_key")),
                "base_url": _clean_text(fallback.get("base_url")),
                "model": _clean_text(fallback.get("model")) or _clean_text(llm_config.get("model")),
                "timeout_seconds": str(
                    max(min(int(fallback.get("input_parse_timeout_seconds") or fallback.get("timeout_seconds") or parse_timeout), 8), 3)
                ),
            }
        )
    return candidates


def _maybe_enrich_free_text_with_llm(parsed: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if parsed.get("scholar_name") or parsed.get("topics") or parsed.get("aminer_user_id"):
        return parsed
    free_text = _clean_text(parsed.get("free_text"))
    if not free_text:
        return parsed

    for candidate in _llm_config_candidates(config):
        if not candidate.get("api_key"):
            continue
        try:
            resolved, _ = llm_parse_profile_input(
                free_text,
                api_key=str(candidate.get("api_key") or ""),
                base_url=str(candidate.get("base_url") or ""),
                model=str(candidate.get("model") or "gpt-5-mini"),
                timeout_seconds=int(candidate.get("timeout_seconds") or 30),
            )
        except Exception:
            continue

        enriched = dict(parsed)
        intent = _clean_text(resolved.get("intent")).lower()
        scholar_name = _clean_text(resolved.get("scholar_name"))
        scholar_org = _clean_text(resolved.get("scholar_org"))
        topics = _split_topics("，".join(str(item) for item in list(resolved.get("topics") or [])))
        residual = _clean_text(resolved.get("free_text"))

        if intent in {"scholar", "mixed"} and scholar_name:
            enriched["scholar_name"] = scholar_name
            enriched["scholar_org"] = scholar_org or _clean_text(parsed.get("scholar_org"))
        if intent in {"topic", "mixed"} and topics:
            enriched["topics"] = topics
        if scholar_name or topics:
            enriched["free_text"] = residual
            return enriched
    return parsed


def _infer_route_from_sessions_store() -> dict[str, str]:
    explicit = _clean_text(os.getenv("OPENCLAW_SESSIONS_PATH"))
    sessions_path = Path(explicit).expanduser() if explicit else _resolve_openclaw_home() / "agents/main/sessions/sessions.json"
    if not sessions_path.exists():
        return {"target": "", "account_id": "default"}
    try:
        payload = json.loads(sessions_path.read_text(encoding="utf-8"))
    except Exception:
        return {"target": "", "account_id": "default"}

    latest_key = ""
    latest_updated_at = -1
    for key, value in payload.items():
        if not str(key).startswith("agent:main:feishu:direct:"):
            continue
        if not isinstance(value, dict):
            continue
        updated_at = int(value.get("updatedAt") or -1)
        if updated_at > latest_updated_at:
            latest_updated_at = updated_at
            latest_key = str(key)

    if not latest_key:
        return {"target": "", "account_id": "default"}

    sender_id = latest_key.rsplit(":", 1)[-1].strip()
    target = f"user:{sender_id}" if sender_id else ""
    return {"target": target, "account_id": "default"}


def infer_delivery_route(text: str) -> dict[str, str]:
    raw_text = str(text or "")
    sender_match = re.search(r'"sender_id"\s*:\s*"([^"]+)"', raw_text)
    if not sender_match:
        sender_match = re.search(r"sender_id\s*[:：]\s*([A-Za-z0-9_-]+)", raw_text, flags=re.IGNORECASE)
    sender_id = _clean_text(sender_match.group(1)) if sender_match else ""
    target = f"user:{sender_id}" if sender_id and ":" not in sender_id else sender_id

    account_match = re.search(r'"accountId"\s*:\s*"([^"]+)"', raw_text)
    if not account_match:
        account_match = re.search(r"account[_ ]id\s*[:：]\s*([A-Za-z0-9_-]+)", raw_text, flags=re.IGNORECASE)
    account_id = _clean_text(account_match.group(1)) if account_match else "default"

    if target:
        return {"target": target, "account_id": account_id or "default"}

    return _infer_route_from_sessions_store()


def _run_pipeline(
    *,
    base_dir: Path,
    output_dir: Path,
    config_path: Path | None,
    aminer_user_id: str,
    topics: list[str],
    scholar_name: str,
    scholar_org: str,
    paper_titles: list[str],
    papers_file: str,
    free_text: str,
    target: str,
    account_id: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(base_dir / "scripts" / "run_pipeline.py"),
        "--base-dir",
        str(base_dir),
        "--output-dir",
        str(output_dir),
        "--account",
        account_id,
    ]
    if config_path is not None:
        command.extend(["--config", str(config_path)])
    if target.strip():
        command.extend(["--target", target.strip()])
    if aminer_user_id.strip():
        command.extend(["--aminer-user-id", aminer_user_id.strip()])
    if topics:
        command.extend(["--topics", *topics])
    if scholar_name.strip():
        command.extend(["--scholar-name", scholar_name.strip()])
    if scholar_org.strip():
        command.extend(["--scholar-org", scholar_org.strip()])
    for paper_title in paper_titles:
        if paper_title.strip():
            command.extend(["--paper-title", paper_title.strip()])
    if papers_file.strip():
        command.extend(["--papers-file", papers_file.strip()])
    if free_text.strip():
        command.extend(["--free-text", free_text.strip()])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "run_pipeline failed"
        raise RuntimeError(detail)
    return json.loads(completed.stdout)


def _compact_pipeline_error(detail: str) -> str:
    text = _clean_text(detail)
    if not text:
        return "unknown_error"
    if "Traceback" not in text:
        return text
    lines = [line.strip() for line in str(detail or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("RuntimeError:"):
            return _clean_text(line.split("RuntimeError:", 1)[1])
    return _clean_text(lines[-1]) if lines else text


def _build_acknowledgement_message(parsed: dict[str, Any]) -> str:
    scholar_name = _clean_text(parsed.get("scholar_name"))
    scholar_org = _clean_text(parsed.get("scholar_org"))
    topics = [topic for topic in list(parsed.get("topics") or []) if _clean_text(topic)]
    free_text = _clean_text(parsed.get("free_text"))
    aminer_user_id = _clean_text(parsed.get("aminer_user_id"))

    if scholar_name:
        if scholar_org:
            return f"已识别学者 {scholar_name}（{scholar_org}），正在根据已发论文归纳研究方向并生成推荐，请稍候。"
        return f"已识别学者 {scholar_name}，正在根据已发论文归纳研究方向并生成推荐，请稍候。"
    if aminer_user_id:
        return "已识别 AMiner 学者线索，正在归纳研究方向并生成推荐，请稍候。"
    if topics:
        return f"已收到推荐请求，正在围绕 {' / '.join(topics[:5])} 检索并整理论文，请稍候。"
    if free_text:
        return "已收到推荐请求，正在解析研究方向并检索相关论文，请稍候。"
    return "已收到推荐请求，正在生成论文推荐，请稍候。"


def _maybe_send_acknowledgement(
    *,
    output_dir: Path,
    parsed: dict[str, Any],
    target: str,
    account_id: str,
) -> None:
    if not _clean_text(target):
        return
    message_text = _build_acknowledgement_message(parsed)
    try:
        send_text_via_route(
            output_dir,
            message_text,
            target=target,
            account_id=account_id,
            dry_run=False,
        )
    except Exception:
        # Acknowledgement is best-effort; do not block the pipeline if this fails.
        return


def handle_trigger(
    *,
    base_dir: Path,
    text: str,
    target: str = "",
    account_id: str = "default",
    config_path: Path | None = None,
) -> dict[str, Any]:
    parsed = parse_trigger_text(text)
    try:
        parsed = _normalize_interface_payload(parsed, base_dir=base_dir)
    except ValueError as exc:
        detail = _clean_text(str(exc))
        if detail == "invalid_aminer_user_id":
            reply_text = "输入里的 `aminer_user_id` 不合法。请提供 24 位十六进制字符串，例如：`/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, 智能体`。"
        elif detail == "papers_file_outside_base_dir":
            reply_text = "出于安全限制，`papers_file` 只能指向当前 skill 目录内的 JSON 文件，不能引用目录外路径。"
        elif detail == "unsupported_papers_file":
            reply_text = "`papers_file` 目前只支持 `.json` 文件。"
        else:
            reply_text = f"输入不符合接口约束：{detail}"
        return {
            "status": "success",
            "mode": "invalid_input",
            "final_response": "TEXT",
            "reply_text": reply_text,
        }
    inferred_route = infer_delivery_route(text)
    resolved_target = _truncate_text(_clean_text(target) or inferred_route["target"], MAX_TARGET_LENGTH)
    resolved_account_id = _truncate_text(_clean_text(account_id) or inferred_route["account_id"] or "default", MAX_ACCOUNT_ID_LENGTH) or "default"
    has_profile_input = bool(
        parsed["aminer_user_id"]
        or parsed["topics"]
        or parsed["scholar_name"]
        or parsed["scholar_org"]
        or parsed["paper_titles"]
        or parsed["papers_file"]
        or parsed["free_text"]
    )
    if not parsed["is_trigger"] and not has_profile_input:
        return {
            "status": "success",
            "mode": "help",
            "final_response": "TEXT",
            "reply_text": "请发送 `/aminer-rec5 topics: 多模态, 智能体`，或 `/aminer-rec5 scholar: Jie Tang papers: Paper A | Paper B`，也可以直接用自然语言描述你的研究方向。",
        }

    if not has_profile_input:
        return {
            "status": "success",
            "mode": "onboarding_prompt",
            "final_response": "TEXT",
            "reply_text": "请提供 `aminer_user_id`、论文标题、论文文件路径，或直接描述研究方向，例如：`/aminer-rec5 我做多模态智能体和 tool-use`。",
        }

    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded_config = _load_config(base_dir, config_path)
    parsed = _maybe_enrich_free_text_with_llm(parsed, loaded_config)
    runtime_config_path = output_dir / "runtime_config.yaml"
    runtime_config_path.write_text(yaml.safe_dump(loaded_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _maybe_send_acknowledgement(
        output_dir=output_dir,
        parsed=parsed,
        target=resolved_target,
        account_id=resolved_account_id,
    )

    try:
        pipeline_result = _run_pipeline(
            base_dir=base_dir,
            output_dir=output_dir,
            config_path=runtime_config_path,
            aminer_user_id=parsed["aminer_user_id"],
            topics=parsed["topics"],
            scholar_name=parsed["scholar_name"],
            scholar_org=parsed["scholar_org"],
            paper_titles=parsed["paper_titles"],
            papers_file=parsed["papers_file"],
            free_text=parsed["free_text"],
            target=resolved_target,
            account_id=resolved_account_id,
        )
    except Exception as exc:
        detail = _compact_pipeline_error(str(exc).strip())
        if parsed["aminer_user_id"] and not parsed["topics"] and ("profile_unavailable" in detail or "missing_topics" in detail or "no_bind_papers_or_experts_topic" in detail):
            return {
                "status": "success",
                "mode": "onboarding_prompt",
                "final_response": "TEXT",
                "reply_text": "我还没能从这个 `aminer_user_id` 归纳出稳定研究方向，请补充研究方向或代表论文，例如：`/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, 智能体`。",
            }
        if parsed["scholar_name"] and ("profile_unavailable" in detail or "missing_topics" in detail):
            return {
                "status": "success",
                "mode": "onboarding_prompt",
                "final_response": "TEXT",
                "reply_text": "我查到了学者线索，但还没成功归纳出稳定研究方向。请补充 `topics`、代表论文标题，或直接描述方向，例如：`/aminer-rec5 我是张帆进，清华大学，做多模态智能体和 tool-use`。",
            }
        if has_profile_input or detail:
            return {
                "status": "success",
                "mode": "error",
                "final_response": "TEXT",
                "reply_text": f"推荐流程执行失败，出错阶段：{detail}",
            }

    return {
        "status": "success",
        "mode": pipeline_result.get("mode", "success"),
        "parsed_input": parsed,
        "artifacts": {
            "runtime_config": str(runtime_config_path),
            "output_dir": str(output_dir),
        },
        "delivery_route": {
            "target": resolved_target,
            "account_id": resolved_account_id,
        },
        "pipeline": pipeline_result,
        "final_response": pipeline_result.get("final_response", "NO_REPLY"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Handle Feishu trigger text for aminer-rec5.")
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--text", required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--account", default="default")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    result = handle_trigger(
        base_dir=args.base_dir.resolve(),
        text=args.text,
        target=args.target,
        account_id=args.account,
        config_path=args.config.resolve() if args.config else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
