from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.dispatch_feishu_messages import _build_dispatch_stage_message, dispatch_messages


class DispatchFeishuMessagesTests(unittest.TestCase):
    def test_build_dispatch_stage_message_appends_profile_topics(self) -> None:
        message = _build_dispatch_stage_message({"profile_topics": ["多模态智能体", "tool-use", "视觉推理"]})
        self.assertIn("开始发送论文卡片", message)
        self.assertIn("研究方向：多模态智能体 / tool-use / 视觉推理", message)

    def test_build_dispatch_stage_message_keeps_default_when_no_topics(self) -> None:
        self.assertEqual(_build_dispatch_stage_message({}), "开始发送论文卡片")

    def test_dispatch_messages_falls_back_to_markdown_when_card_send_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            messages_path = output_dir / "feishu_messages.json"
            route_path = output_dir / "manual_reply_route.json"
            route_path.write_text(
                json.dumps({"target": "user:demo", "accountId": "default"}, ensure_ascii=False),
                encoding="utf-8",
            )
            messages_path.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "paper_count": 1,
                        "messages": [
                            {
                                "title": "Paper A",
                                "card_json": json.dumps(
                                    {
                                        "header": {"title": {"content": "1. Paper A"}},
                                        "elements": [{"text": {"content": "**关键词**\nKG"}}],
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            calls: list[list[str]] = []

            def fake_run(command, capture_output, text, check, env, timeout):  # type: ignore[no-untyped-def]
                calls.append(list(command))
                if "--card" in command:
                    raise subprocess.TimeoutExpired(command, timeout)

                class Result:
                    returncode = 0
                    stdout = json.dumps({"payload": {"result": {"messageId": "om_demo"}}}, ensure_ascii=False)
                    stderr = ""

                return Result()

            with patch("scripts.dispatch_feishu_messages.subprocess.run", side_effect=fake_run):
                result = dispatch_messages(messages_path, openclaw_bin="openclaw")

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(calls), 3)
        self.assertIn("--message", calls[0])
        self.assertIn("--card", calls[1])
        self.assertIn("--message", calls[2])


if __name__ == "__main__":
    unittest.main()
