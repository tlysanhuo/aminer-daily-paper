#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from glob import glob
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.common import read_json, utc_now_iso, write_json
from scripts.openclaw_stub import resolve_delivery_route

PROGRESS_FILE_NAME = "pipeline_progress.json"
DISPATCH_STAGE_MESSAGE = "开始发送论文卡片"
OPENCLAW_TEXT_SEND_TIMEOUT_SECONDS = 20
OPENCLAW_CARD_SEND_TIMEOUT_SECONDS = 12


def _candidate_node_bin_dirs() -> list[str]:
    home = Path.home()
    candidates = [
        home / ".local/node-v22/bin",
        home / ".local/node-v24/bin",
    ]
    candidates.extend(Path(path) for path in sorted(glob(str(home / ".nvm/versions/node/v22.*/bin"))))
    candidates.extend(Path(path) for path in sorted(glob(str(home / ".nvm/versions/node/v24.*/bin"))))
    return [str(path) for path in candidates if (path / "node").exists()]


def build_openclaw_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    for candidate in reversed(_candidate_node_bin_dirs()):
        if candidate not in path_entries:
            path_entries.insert(0, candidate)
    if path_entries:
        env["PATH"] = os.pathsep.join(path_entries)
    return env


def build_openclaw_command(card_json: str, target: str, account: str, dry_run: bool) -> list[str]:
    command = [
        "openclaw",
        "message",
        "send",
        "--channel",
        "feishu",
        "--account",
        account,
        "--target",
        target,
        "--card",
        card_json,
        "--json",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _extract_json_objects(output: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def _parse_openclaw_send_output(output: str, dry_run: bool) -> Any:
    stripped = output.strip()
    if not stripped:
        raise RuntimeError("openclaw message send returned empty output")

    parsed_payload: Any = None
    try:
        parsed_payload = json.loads(stripped)
    except json.JSONDecodeError:
        objects = _extract_json_objects(stripped)
        if objects:
            parsed_payload = objects[-1]
        else:
            parsed_payload = stripped

    if dry_run:
        return parsed_payload

    if isinstance(parsed_payload, dict):
        payload = parsed_payload.get("payload")
        if isinstance(payload, dict):
            result = payload.get("result")
            if isinstance(result, dict) and str(result.get("messageId", "")).strip():
                return parsed_payload
        if str(parsed_payload.get("messageId", "")).strip():
            return parsed_payload

    raise RuntimeError("openclaw message send did not return a delivery receipt with messageId")


def _build_openclaw_text_command(message_text: str, *, target: str, account: str, dry_run: bool) -> list[str]:
    command = [
        "openclaw",
        "message",
        "send",
        "--channel",
        "feishu",
        "--account",
        account,
        "--target",
        target,
        "--message",
        message_text,
        "--json",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def _send_text_message(
    message_text: str,
    *,
    target: str,
    account: str,
    dry_run: bool,
    openclaw_bin: str,
) -> dict[str, Any]:
    command = _build_openclaw_text_command(
        message_text,
        target=target,
        account=account,
        dry_run=dry_run,
    )
    command[0] = openclaw_bin
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=build_openclaw_env(),
            timeout=OPENCLAW_TEXT_SEND_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("openclaw message send timeout") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "openclaw message send failed"
        raise RuntimeError(detail)
    return _parse_openclaw_send_output(completed.stdout.strip(), dry_run=dry_run)


def send_text_via_route(
    output_dir: Path,
    message_text: str,
    *,
    target: str = "",
    account_id: str = "main",
    dry_run: bool = False,
    openclaw_bin: str = "openclaw",
) -> dict[str, Any]:
    route = resolve_delivery_route(
        output_dir,
        fallback_target=target,
        fallback_account_id=account_id,
    )
    return _send_text_message(
        message_text,
        target=route["target"],
        account=route["accountId"],
        dry_run=dry_run,
        openclaw_bin=openclaw_bin,
    )


def _load_progress_events(output_dir: Path) -> list[dict[str, Any]]:
    progress_path = output_dir / PROGRESS_FILE_NAME
    if not progress_path.exists():
        return []
    payload = read_json(progress_path)
    events = payload.get("events", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _write_progress_events(output_dir: Path, events: list[dict[str, Any]]) -> None:
    write_json(
        output_dir / PROGRESS_FILE_NAME,
        {
            "status": "success",
            "generated_at": utc_now_iso(),
            "event_count": len(events),
            "events": events,
        },
    )


def _load_progress_messages(output_dir: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    events = _load_progress_events(output_dir)
    terminal_dispatched_stages = {
        str(event.get("stage", "")).strip()
        for event in events
        if str(event.get("stage", "")).strip()
        and str(event.get("status", "")).strip() != "in_progress"
        and bool(event.get("dispatched"))
    }
    for index, event in enumerate(events):
        stage = str(event.get("stage", "")).strip()
        status = str(event.get("status", "")).strip()
        message = str(event.get("message", "")).strip()
        dispatched = bool(event.get("dispatched"))
        if dispatched or not message:
            continue
        if status == "in_progress" and stage in terminal_dispatched_stages:
            continue
        if status == "in_progress" and stage == "render":
            continue
        if status == "in_progress":
            messages.append({"status": status, "message": message, "event_index": index})
    return messages


def _mark_progress_event_dispatched(output_dir: Path, event_index: int) -> None:
    events = _load_progress_events(output_dir)
    if not (0 <= event_index < len(events)):
        return
    event = events[event_index]
    if str(event.get("status", "")).strip() != "in_progress":
        return
    if not str(event.get("message", "")).strip():
        return
    event["dispatched"] = True
    event.pop("dispatch_error", None)
    _write_progress_events(output_dir, events)


def _mark_failure_event_dispatched(output_dir: Path, failure_message: str) -> None:
    events = _load_progress_events(output_dir)
    for event in events:
        status = str(event.get("status", "")).strip()
        message = str(event.get("message", "")).strip()
        if status != "failure" or message != failure_message:
            continue
        event["dispatched"] = True
        event.pop("dispatch_error", None)
        _write_progress_events(output_dir, events)
        return


def _load_failure_event(output_dir: Path) -> dict[str, Any] | None:
    for event in _load_progress_events(output_dir):
        status = str(event.get("status", "")).strip()
        message = str(event.get("message", "")).strip()
        if status == "failure" and message:
            return {
                "status": status,
                "message": message,
                "dispatched": bool(event.get("dispatched")),
            }
    return None


def _build_dispatch_stage_message(payload: dict[str, Any]) -> str:
    profile_topics = [str(item).strip() for item in list(payload.get("profile_topics") or []) if str(item).strip()]
    if not profile_topics:
        return DISPATCH_STAGE_MESSAGE
    return f"{DISPATCH_STAGE_MESSAGE}\n研究方向：{' / '.join(profile_topics[:5])}"


def _render_card_markdown_text(card: dict[str, Any]) -> str:
    sections: list[str] = []

    header = card.get("header")
    if isinstance(header, dict):
        title = header.get("title")
        if isinstance(title, dict):
            content = str(title.get("content", "")).strip()
            if content:
                sections.append(content)

    for element in card.get("elements", []):
        if not isinstance(element, dict):
            continue
        text = element.get("text")
        if not isinstance(text, dict):
            continue
        content = str(text.get("content", "")).strip()
        if content:
            sections.append(content)

    rendered = "\n\n".join(sections).strip()
    if not rendered:
        raise ValueError("card_json does not contain any renderable text blocks for markdown fallback")
    return rendered


def extract_card_json(message: dict[str, Any], index: int) -> str:
    if "card_json" in message:
        card_json = str(message["card_json"])
    elif "card" in message:
        card_json = json.dumps(message["card"], ensure_ascii=False, separators=(",", ":"))
    else:
        raise ValueError(f"message {index} missing card_json")
    parsed = json.loads(card_json)
    if not isinstance(parsed, dict):
        raise ValueError(f"message {index} card_json must decode to a JSON object")
    return card_json


def dispatch_messages(
    messages_path: Path,
    target: str = "",
    account_id: str = "main",
    dry_run: bool = False,
    openclaw_bin: str = "openclaw",
) -> dict[str, Any]:
    route = resolve_delivery_route(
        messages_path.parent,
        fallback_target=target,
        fallback_account_id=account_id,
    )

    results: list[dict[str, Any]] = []
    failure_event = _load_failure_event(messages_path.parent)
    if failure_event is not None:
        failure_result = None
        if not failure_event.get("dispatched"):
            failure_result = _send_text_message(
                failure_event["message"],
                target=route["target"],
                account=route["accountId"],
                dry_run=dry_run,
                openclaw_bin=openclaw_bin,
            )
            if not dry_run:
                _mark_failure_event_dispatched(messages_path.parent, failure_event["message"])
        results.append(
            {
                "index": 1,
                "title": failure_event["message"],
                "delivery_mode": "failure_text",
                "result": failure_result,
            }
        )
        return {
            "status": "failure",
            "message_count": len(results),
            "progress_message_count": len(results),
            "target": route["target"],
            "accountId": route["accountId"],
            "results": results,
            "final_response": "NO_REPLY",
        }

    payload = read_json(messages_path)
    progress_messages = _load_progress_messages(messages_path.parent)
    if payload.get("messages"):
        progress_messages.append({"status": "in_progress", "message": _build_dispatch_stage_message(payload)})
    for index, progress_message in enumerate(progress_messages, start=1):
        progress_result = _send_text_message(
            progress_message["message"],
            target=route["target"],
            account=route["accountId"],
            dry_run=dry_run,
            openclaw_bin=openclaw_bin,
        )
        if not dry_run and "event_index" in progress_message:
            _mark_progress_event_dispatched(messages_path.parent, int(progress_message["event_index"]))
        results.append(
            {
                "index": index,
                "title": progress_message["message"],
                "delivery_mode": "progress_text",
                "result": progress_result,
            }
        )

    card_delivery_supported = True
    for index, message in enumerate(payload.get("messages", []), start=1):
        card_json = extract_card_json(message, index)
        card_payload = json.loads(card_json)
        if not isinstance(card_payload, dict):
            raise ValueError(f"message {index} card_json must decode to a JSON object")
        delivery_mode = "card"
        message_result: Any
        if card_delivery_supported:
            command = build_openclaw_command(
                card_json=card_json,
                target=route["target"],
                account=route["accountId"],
                dry_run=dry_run,
            )
            command[0] = openclaw_bin
            card_send_timed_out = False
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=build_openclaw_env(),
                    timeout=OPENCLAW_CARD_SEND_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
                card_send_timed_out = True
                completed = None
            if completed is not None and completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip() or "openclaw message send failed"
                raise RuntimeError(detail)
            try:
                if card_send_timed_out or completed is None:
                    raise RuntimeError("openclaw card send timeout")
                message_result = _parse_openclaw_send_output(completed.stdout.strip(), dry_run=dry_run)
            except RuntimeError:
                card_delivery_supported = False
                message_result = _send_text_message(
                    _render_card_markdown_text(card_payload),
                    target=route["target"],
                    account=route["accountId"],
                    dry_run=dry_run,
                    openclaw_bin=openclaw_bin,
                )
                delivery_mode = "markdown_fallback"
        else:
            message_result = _send_text_message(
                _render_card_markdown_text(card_payload),
                target=route["target"],
                account=route["accountId"],
                dry_run=dry_run,
                openclaw_bin=openclaw_bin,
            )
            delivery_mode = "markdown_fallback"
        results.append(
            {
                "index": index,
                "title": str(message.get("title", "")),
                "delivery_mode": delivery_mode,
                "result": message_result,
            }
        )
    return {
        "status": "success",
        "message_count": len(results),
        "progress_message_count": len(progress_messages),
        "target": route["target"],
        "accountId": route["accountId"],
        "results": results,
        "final_response": "NO_REPLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dispatch Feishu messages through OpenClaw CLI.")
    parser.add_argument("--messages", type=Path, required=True)
    parser.add_argument("--target", default="")
    parser.add_argument("--account-id", default="main")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--openclaw-bin", default="openclaw")
    args = parser.parse_args()

    result = dispatch_messages(
        args.messages,
        target=args.target,
        account_id=args.account_id,
        dry_run=args.dry_run,
        openclaw_bin=args.openclaw_bin,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
