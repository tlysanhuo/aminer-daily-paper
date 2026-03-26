from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from scripts.common import read_json


def _read_nonempty_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def resolve_delivery_route(
    output_dir: Path,
    fallback_target: str = "",
    fallback_account_id: str = "main",
) -> dict[str, str]:
    route_path = output_dir / "manual_reply_route.json"
    route_raw = _read_nonempty_text(route_path)
    if route_raw:
        route_payload = json.loads(route_raw)
        target = str(route_payload.get("target", "")).strip()
        account_id = str(route_payload.get("accountId", "")).strip()
        if not target:
            raise ValueError(f"manual reply route missing target: {route_path}")
        if not account_id:
            raise ValueError(f"manual reply route missing accountId: {route_path}")
        return {"target": target, "accountId": account_id}

    target = _read_nonempty_text(output_dir / "manual_reply_target.txt") or fallback_target.strip()
    account_id = _read_nonempty_text(output_dir / "manual_reply_account_id.txt") or fallback_account_id.strip() or "main"
    if not target:
        raise ValueError("Feishu target is required for OpenClaw message dispatch")
    if not account_id:
        raise ValueError("Feishu accountId is required for OpenClaw message dispatch")
    return {"target": target, "accountId": account_id}


def dispatch_message_actions(
    messages_payload: dict[str, Any],
    send_message: Callable[[dict[str, Any]], None],
    target: str,
    account_id: str = "main",
) -> dict[str, Any]:
    resolved_target = target.strip()
    resolved_account_id = account_id.strip()
    if not resolved_target:
        raise ValueError("Feishu target is required for OpenClaw message dispatch")
    if not resolved_account_id:
        raise ValueError("Feishu accountId is required for OpenClaw message dispatch")
    actions: list[dict[str, Any]] = []
    for message in messages_payload.get("messages", []):
        action: dict[str, Any] = {
            "action": "send",
            "channel": "feishu",
            "accountId": resolved_account_id,
            "target": resolved_target,
        }
        if "card_json" in message:
            action["card"] = str(message["card_json"])
        elif "card" in message:
            action["card"] = message["card"]
        else:
            action["message"] = str(message.get("text", ""))
        send_message(action)
        actions.append(action)
    return {"actions": actions, "final_response": "NO_REPLY"}


def load_and_dispatch_messages(
    messages_path: Path,
    send_message: Callable[[dict[str, Any]], None],
    target: str = "",
    account_id: str = "main",
) -> dict[str, Any]:
    payload = read_json(messages_path)
    route = resolve_delivery_route(
        messages_path.parent,
        fallback_target=target,
        fallback_account_id=account_id,
    )
    return dispatch_message_actions(
        payload,
        send_message=send_message,
        target=route["target"],
        account_id=route["accountId"],
    )
