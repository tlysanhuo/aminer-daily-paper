#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib import error, request

import yaml


FIELD_LABELS = {
    "aminer_user_id": ["aminer_user_id"],
    "topics": ["topics", "topic", "方向", "研究方向"],
    "scholar_name": ["scholar", "name", "author", "学者", "作者"],
    "scholar_org": ["org", "organization", "affiliation", "机构", "单位"],
    "paper_titles": ["paper", "papers", "代表作", "论文"],
    "papers_file": ["papers_file", "source_file", "profile_file", "文件", "路径"],
}


MAX_TOPICS = 8
MAX_TOPIC_LENGTH = 80
MAX_PAPER_TITLES = 8
MAX_PAPER_TITLE_LENGTH = 300
MAX_SCHOLAR_NAME_LENGTH = 80
MAX_SCHOLAR_ORG_LENGTH = 160
MAX_FREE_TEXT_LENGTH = 600
MAX_TARGET_LENGTH = 160
MAX_ACCOUNT_ID_LENGTH = 64
MAX_COMMAND_TEXT_LENGTH = 2000
MAX_LANGUAGE_LENGTH = 16
MAX_BACKEND_MODE_LENGTH = 64
MAX_BACKEND_REQUEST_ID_LENGTH = 128
MAX_REPLY_TEXT_LENGTH = 4000
MAX_ERROR_CODE_LENGTH = 128
MAX_ERROR_MESSAGE_LENGTH = 500
MAX_SEED_PAPERS = 20
MAX_SEED_PAPER_TITLE_LENGTH = 300
MAX_SEED_PAPER_ABSTRACT_LENGTH = 4000
MAX_SEED_PAPER_VENUE_LENGTH = 200
MAX_SEED_PAPER_URL_LENGTH = 500
MAX_SEED_PAPER_ID_LENGTH = 128
MAX_SEED_PAPER_AUTHORS = 20
MAX_SEED_PAPER_AUTHOR_LENGTH = 80
ALLOWED_PAPERS_FILE_SUFFIXES = {".json"}
DEFAULT_BACKEND_RECOMMEND_PATH = "/v1/recommend-and-dispatch"
DEFAULT_BACKEND_TIMEOUT_SECONDS = 30
MAX_BACKEND_TIMEOUT_SECONDS = 120


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _truncate_text(value: Any, max_length: int) -> str:
    cleaned = _clean_text(value)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[:max_length].strip()


def _parse_int(value: Any, default: int, *, minimum: int, maximum: int) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception:
        return default
    return max(min(parsed, maximum), minimum)


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


def _capture_field(command_body: str, field_name: str) -> str:
    labels = FIELD_LABELS[field_name]
    all_labels = [re.escape(label) for values in FIELD_LABELS.values() for label in values]
    pattern = rf"(?:{'|'.join(re.escape(label) for label in labels)})\s*[:：]\s*(.+?)(?=\s*(?:{'|'.join(all_labels)})\s*[:：]|$)"
    match = re.search(pattern, command_body, flags=re.IGNORECASE | re.S)
    return _clean_text(match.group(1)) if match else ""


def _strip_explicit_fields(command_body: str) -> str:
    all_labels = [re.escape(label) for values in FIELD_LABELS.values() for label in values]
    pattern = rf"(?:{'|'.join(all_labels)})\s*[:：]\s*.+?(?=\s*(?:{'|'.join(all_labels)})\s*[:：]|$)"
    cleaned = re.sub(pattern, " ", command_body, flags=re.IGNORECASE | re.S)
    return _clean_text(cleaned)


def parse_trigger_text(text: str) -> dict[str, Any]:
    raw_text = str(text or "")
    command_text = _extract_command_text(raw_text)
    normalized = _clean_text(command_text)
    body = re.sub(r"^/(skill\s+)?aminer[-_]rec5\b", "", command_text, flags=re.IGNORECASE).strip()
    uid_match = re.search(r"aminer_user_id\s*[:：]\s*([0-9a-fA-F]{24})", body, flags=re.IGNORECASE)
    uid = uid_match.group(1) if uid_match else ""

    return {
        "raw_text": raw_text,
        "command_text": _truncate_text(command_text, MAX_COMMAND_TEXT_LENGTH),
        "raw_aminer_user_id": _capture_field(body, "aminer_user_id"),
        "aminer_user_id": uid,
        "topics": _split_topics(_capture_field(body, "topics")),
        "scholar_name": _capture_field(body, "scholar_name"),
        "scholar_org": _capture_field(body, "scholar_org"),
        "paper_titles": _split_papers(_capture_field(body, "paper_titles")),
        "papers_file": _capture_field(body, "papers_file"),
        "free_text": _strip_explicit_fields(re.sub(r"aminer_user_id\s*[:：]\s*[0-9a-fA-F]{24}", " ", body, flags=re.IGNORECASE)),
        "is_trigger": bool(re.search(r"^/(skill\s+)?aminer[-_]rec5\b", normalized, flags=re.IGNORECASE)),
    }


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


def _normalize_seed_paper(record: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    title = _truncate_text(record.get("title"), MAX_SEED_PAPER_TITLE_LENGTH)
    abstract = _truncate_text(record.get("abstract"), MAX_SEED_PAPER_ABSTRACT_LENGTH)
    venue = _truncate_text(record.get("venue"), MAX_SEED_PAPER_VENUE_LENGTH)
    url_value = _truncate_text(record.get("url"), MAX_SEED_PAPER_URL_LENGTH)
    paper_id = _truncate_text(record.get("paper_id") or record.get("id"), MAX_SEED_PAPER_ID_LENGTH)
    year_raw = record.get("year")
    year: int | None = None
    try:
        if str(year_raw).strip():
            parsed_year = int(str(year_raw).strip())
            if 1900 <= parsed_year <= 2100:
                year = parsed_year
    except Exception:
        year = None

    authors: list[str] = []
    for item in list(record.get("authors") or []):
        author = _truncate_text(item, MAX_SEED_PAPER_AUTHOR_LENGTH)
        if author and author not in authors:
            authors.append(author)
        if len(authors) >= MAX_SEED_PAPER_AUTHORS:
            break

    if title:
        normalized["title"] = title
    if abstract:
        normalized["abstract"] = abstract
    if venue:
        normalized["venue"] = venue
    if url_value:
        normalized["url"] = url_value
    if paper_id:
        normalized["paper_id"] = paper_id
    if year is not None:
        normalized["year"] = year
    if authors:
        normalized["authors"] = authors
    return normalized


def _load_seed_papers_from_file(path_text: str) -> list[dict[str, Any]]:
    path = Path(path_text).expanduser()
    if not path.exists():
        raise ValueError("papers_file_not_found")
    if path.suffix.lower() not in ALLOWED_PAPERS_FILE_SUFFIXES:
        raise ValueError("unsupported_papers_file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError("invalid_papers_file_json") from exc

    papers = _extract_papers_from_payload(payload)
    normalized: list[dict[str, Any]] = []
    for record in papers:
        compact = _normalize_seed_paper(record)
        if compact and ("title" in compact or "abstract" in compact):
            normalized.append(compact)
        if len(normalized) >= MAX_SEED_PAPERS:
            break
    if not normalized:
        raise ValueError("empty_papers_file_payload")
    return normalized


def _resolve_openclaw_home() -> Path:
    explicit = _clean_text(os.getenv("OPENCLAW_HOME"))
    if explicit:
        return Path(explicit).expanduser()
    return Path.home() / ".openclaw"


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


def _load_config(base_dir: Path, config_path: Path | None) -> dict[str, Any]:
    candidate = config_path if config_path and config_path.exists() else base_dir / "config.example.yaml"
    if not candidate.exists():
        return {}
    return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}


def _resolve_backend_config(config: dict[str, Any]) -> dict[str, Any]:
    backend = dict(config.get("backend") or {})
    base_url = _clean_text(os.getenv("AMINER_REC_BACKEND_BASE_URL") or backend.get("base_url"))
    recommend_path = _clean_text(os.getenv("AMINER_REC_BACKEND_PATH") or backend.get("recommend_path") or DEFAULT_BACKEND_RECOMMEND_PATH)
    api_key = _clean_text(os.getenv("AMINER_REC_BACKEND_API_KEY") or backend.get("api_key"))
    timeout_seconds = _parse_int(
        os.getenv("AMINER_REC_BACKEND_TIMEOUT_SECONDS") or backend.get("timeout_seconds") or DEFAULT_BACKEND_TIMEOUT_SECONDS,
        DEFAULT_BACKEND_TIMEOUT_SECONDS,
        minimum=3,
        maximum=MAX_BACKEND_TIMEOUT_SECONDS,
    )
    language = _truncate_text(os.getenv("AMINER_REC_LANGUAGE") or config.get("language") or "zh", MAX_LANGUAGE_LENGTH) or "zh"

    if not base_url:
        raise ValueError("missing_backend_base_url")
    return {
        "base_url": base_url,
        "recommend_path": recommend_path,
        "api_key": api_key,
        "timeout_seconds": timeout_seconds,
        "language": language,
    }


def _build_backend_url(base_url: str, path_text: str) -> str:
    if re.match(r"^https?://", path_text, flags=re.IGNORECASE):
        return path_text
    return f"{base_url.rstrip('/')}/{path_text.lstrip('/')}"


def _build_backend_request_payload(
    *,
    parsed: dict[str, Any],
    target: str,
    account_id: str,
    language: str,
) -> dict[str, Any]:
    seed_papers = _load_seed_papers_from_file(parsed["papers_file"]) if parsed.get("papers_file") else []
    return {
        "request_context": {
            "skill_name": "aminer-rec5",
            "channel": "openclaw",
            "language": language,
            "command_text": _truncate_text(parsed.get("command_text"), MAX_COMMAND_TEXT_LENGTH),
        },
        "route": {
            "target": _truncate_text(target, MAX_TARGET_LENGTH),
            "account_id": _truncate_text(account_id, MAX_ACCOUNT_ID_LENGTH) or "default",
        },
        "profile_input": {
            "aminer_user_id": _clean_text(parsed.get("aminer_user_id")),
            "topics": _normalize_topics_for_interface(list(parsed.get("topics") or [])),
            "scholar_name": _truncate_text(parsed.get("scholar_name"), MAX_SCHOLAR_NAME_LENGTH),
            "scholar_org": _truncate_text(parsed.get("scholar_org"), MAX_SCHOLAR_ORG_LENGTH),
            "paper_titles": _normalize_paper_titles_for_interface(list(parsed.get("paper_titles") or [])),
            "seed_papers": seed_papers,
            "free_text": _truncate_text(parsed.get("free_text"), MAX_FREE_TEXT_LENGTH),
        },
    }


def _normalize_backend_response(payload: dict[str, Any]) -> dict[str, Any]:
    status = _clean_text(payload.get("status") or "success") or "success"
    mode = _truncate_text(payload.get("mode") or "backend_proxy", MAX_BACKEND_MODE_LENGTH) or "backend_proxy"
    final_response = _clean_text(payload.get("final_response") or "")
    reply_text = _truncate_text(payload.get("reply_text"), MAX_REPLY_TEXT_LENGTH)
    backend_request_id = _truncate_text(payload.get("backend_request_id"), MAX_BACKEND_REQUEST_ID_LENGTH)
    error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else {}
    result_payload = payload.get("result") if isinstance(payload.get("result"), dict) else {}

    if not final_response:
        final_response = "TEXT" if reply_text else "NO_REPLY"
    if status not in {"success", "error"}:
        raise ValueError("invalid_backend_status")
    if final_response not in {"TEXT", "NO_REPLY"}:
        raise ValueError("invalid_backend_final_response")

    normalized_error = {}
    error_code = _truncate_text(error_payload.get("code"), MAX_ERROR_CODE_LENGTH)
    error_message = _truncate_text(error_payload.get("message"), MAX_ERROR_MESSAGE_LENGTH)
    if error_code:
        normalized_error["code"] = error_code
    if error_message:
        normalized_error["message"] = error_message
    if isinstance(error_payload.get("retryable"), bool):
        normalized_error["retryable"] = error_payload["retryable"]

    if status == "error" and not reply_text:
        reply_text = error_message or "推荐后端返回了错误。"
        final_response = "TEXT"

    if final_response == "TEXT" and not reply_text:
        raise ValueError("missing_backend_reply_text")

    normalized = {
        "status": status,
        "mode": mode,
        "final_response": final_response,
    }
    if reply_text:
        normalized["reply_text"] = reply_text
    if backend_request_id:
        normalized["backend_request_id"] = backend_request_id
    if normalized_error:
        normalized["error"] = normalized_error
    if result_payload:
        normalized["result"] = result_payload
    return normalized


def _extract_backend_error_message(status_code: int, body_text: str) -> str:
    cleaned = _clean_text(body_text)
    if not cleaned:
        return f"backend_http_{status_code}"
    try:
        payload = json.loads(body_text)
    except Exception:
        return f"backend_http_{status_code}: {cleaned}"
    if isinstance(payload, dict):
        error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        message = _clean_text(error_payload.get("message") or payload.get("message"))
        code = _clean_text(error_payload.get("code"))
        if code and message:
            return f"{code}: {message}"
        if message:
            return message
    return f"backend_http_{status_code}: {cleaned}"


def _call_backend_api(payload: dict[str, Any], backend_config: dict[str, Any]) -> dict[str, Any]:
    url = _build_backend_url(str(backend_config["base_url"]), str(backend_config["recommend_path"]))
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
    }
    if _clean_text(backend_config.get("api_key")):
        headers["Authorization"] = f"Bearer {_clean_text(backend_config['api_key'])}"

    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=int(backend_config["timeout_seconds"])) as response:
            response_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(_extract_backend_error_message(exc.code, response_body)) from exc
    except error.URLError as exc:
        reason = _clean_text(getattr(exc, "reason", "")) or "unknown_network_error"
        raise RuntimeError(f"backend_unreachable: {reason}") from exc

    try:
        decoded = json.loads(response_body)
    except Exception as exc:
        raise RuntimeError("invalid_backend_json_response") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("invalid_backend_response_shape")
    try:
        return _normalize_backend_response(decoded)
    except ValueError as exc:
        raise RuntimeError(_clean_text(str(exc)) or "invalid_backend_response") from exc


def _invalid_input_response(reply_text: str) -> dict[str, Any]:
    return {
        "status": "success",
        "mode": "invalid_input",
        "final_response": "TEXT",
        "reply_text": reply_text,
    }


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
            return _invalid_input_response(
                "输入里的 `aminer_user_id` 不合法。请提供 24 位十六进制字符串，例如：`/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, 智能体`。"
            )
        if detail == "papers_file_outside_base_dir":
            return _invalid_input_response("出于安全限制，`papers_file` 只能指向当前 skill 目录内的 JSON 文件，不能引用目录外路径。")
        if detail == "unsupported_papers_file":
            return _invalid_input_response("`papers_file` 目前只支持 `.json` 文件。")
        return _invalid_input_response(f"输入不符合接口约束：{detail}")

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

    try:
        backend_config = _resolve_backend_config(_load_config(base_dir, config_path))
    except ValueError:
        return {
            "status": "success",
            "mode": "config_error",
            "final_response": "TEXT",
            "reply_text": "还没有配置推荐后端。请在 `config.yaml` 的 `backend.base_url` 或环境变量 `AMINER_REC_BACKEND_BASE_URL` 里提供后端地址。",
        }

    try:
        request_payload = _build_backend_request_payload(
            parsed=parsed,
            target=resolved_target,
            account_id=resolved_account_id,
            language=str(backend_config["language"]),
        )
    except ValueError as exc:
        detail = _clean_text(str(exc))
        if detail == "papers_file_not_found":
            return _invalid_input_response("`papers_file` 指向的 JSON 文件不存在。")
        if detail in {"invalid_papers_file_json", "empty_papers_file_payload"}:
            return _invalid_input_response("`papers_file` 不是合法 JSON，或里面没有可提取的 paper 数据。")
        if detail == "unsupported_papers_file":
            return _invalid_input_response("`papers_file` 目前只支持 `.json` 文件。")
        return _invalid_input_response(f"`papers_file` 读取失败：{detail}")

    try:
        backend_response = _call_backend_api(request_payload, backend_config)
    except RuntimeError as exc:
        detail = _clean_text(str(exc)) or "unknown_backend_error"
        return {
            "status": "success",
            "mode": "backend_error",
            "final_response": "TEXT",
            "reply_text": f"推荐后端调用失败：{detail}",
        }

    return {
        "status": "success",
        "mode": backend_response.get("mode", "backend_proxy"),
        "parsed_input": parsed,
        "delivery_route": {
            "target": resolved_target,
            "account_id": resolved_account_id,
        },
        "backend_request": request_payload,
        "backend_response": backend_response,
        "final_response": backend_response.get("final_response", "NO_REPLY"),
        **({"reply_text": backend_response["reply_text"]} if backend_response.get("reply_text") else {}),
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
