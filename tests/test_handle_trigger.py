from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from handle_trigger import (
    _build_acknowledgement_message,
    _merge_llm_config,
    handle_trigger,
    infer_delivery_route,
    parse_trigger_text,
)


class HandleTriggerTests(unittest.TestCase):
    def test_parse_trigger_text_extracts_uid_and_topics(self) -> None:
        parsed = parse_trigger_text("/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, 智能体")
        self.assertTrue(parsed["is_trigger"])
        self.assertEqual(parsed["aminer_user_id"], "696259801cb939bc391d3a37")
        self.assertEqual(parsed["topics"], ["多模态", "智能体"])

    def test_parse_trigger_text_extracts_scholar_and_papers(self) -> None:
        parsed = parse_trigger_text("/aminer-rec5 scholar: Jie Tang org: Tsinghua papers: Paper A | Paper B")
        self.assertEqual(parsed["scholar_name"], "Jie Tang")
        self.assertEqual(parsed["scholar_org"], "Tsinghua")
        self.assertEqual(parsed["paper_titles"], ["Paper A", "Paper B"])

    def test_parse_trigger_text_handles_wrapped_feishu_message(self) -> None:
        raw_text = """
System: metadata
{"sender_id":"ou_123"}
[message_id: om_xxx]
ou_123: /aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, 智能体
"""
        parsed = parse_trigger_text(raw_text)
        self.assertTrue(parsed["is_trigger"])
        self.assertEqual(parsed["command_text"], "/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, 智能体")
        self.assertEqual(parsed["topics"], ["多模态", "智能体"])

    def test_parse_trigger_text_keeps_free_text(self) -> None:
        parsed = parse_trigger_text("/aminer-rec5 我做多模态智能体和 tool-use")
        self.assertEqual(parsed["free_text"], "我做多模态智能体和 tool-use")

    def test_parse_trigger_text_infers_scholar_from_natural_language_intro(self) -> None:
        parsed = parse_trigger_text("/aminer-rec5 我是张帆进，清华大学。推荐论文")
        self.assertEqual(parsed["scholar_name"], "张帆进")
        self.assertEqual(parsed["scholar_org"], "清华大学")
        self.assertEqual(parsed["free_text"], "")

    def test_parse_trigger_text_infers_bare_name_and_org_when_org_looks_like_affiliation(self) -> None:
        parsed = parse_trigger_text("/aminer-rec5 李涓子，清华大学。推荐论文")
        self.assertEqual(parsed["scholar_name"], "李涓子")
        self.assertEqual(parsed["scholar_org"], "清华大学")
        self.assertEqual(parsed["free_text"], "")

    def test_build_acknowledgement_message_for_scholar_identity(self) -> None:
        message = _build_acknowledgement_message(
            {
                "scholar_name": "李涓子",
                "scholar_org": "清华大学",
                "topics": [],
                "free_text": "",
                "aminer_user_id": "",
            }
        )
        self.assertIn("李涓子", message)
        self.assertIn("清华大学", message)
        self.assertIn("已发论文", message)

    def test_infer_delivery_route_from_wrapped_feishu_message(self) -> None:
        route = infer_delivery_route('{"sender_id":"ou_9f3999920efe9b7f69fd0c6f322e7d7c"}')
        self.assertEqual(route["target"], "user:ou_9f3999920efe9b7f69fd0c6f322e7d7c")
        self.assertEqual(route["account_id"], "default")

    def test_handle_trigger_returns_help_without_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = handle_trigger(base_dir=Path(temp_dir), text="")
        self.assertEqual(result["mode"], "help")
        self.assertEqual(result["final_response"], "TEXT")

    def test_handle_trigger_runs_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("handle_trigger.send_text_via_route") as send_mock, patch(
                "handle_trigger._run_pipeline",
                return_value={"status": "success", "final_response": "NO_REPLY", "mode": "scholar_path"},
            ) as pipeline_mock:
                result = handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, 智能体",
                    target="user:demo",
                    account_id="main",
                )
        self.assertEqual(result["final_response"], "NO_REPLY")
        self.assertEqual(result["mode"], "scholar_path")
        send_mock.assert_called_once()
        pipeline_mock.assert_called_once()

    def test_handle_trigger_runs_pipeline_for_uid_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch(
                "handle_trigger._run_pipeline",
                return_value={"status": "success", "final_response": "NO_REPLY", "mode": "scholar_path"},
            ) as pipeline_mock:
                result = handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37",
                    target="user:demo",
                    account_id="main",
                )
        self.assertEqual(result["mode"], "scholar_path")
        self.assertEqual(result["final_response"], "NO_REPLY")
        pipeline_mock.assert_called_once()

    def test_handle_trigger_runs_pipeline_for_free_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch(
                "handle_trigger.llm_parse_profile_input",
                return_value=(
                    {
                        "intent": "topic",
                        "scholar_name": "",
                        "scholar_org": "",
                        "topics": ["多模态智能体", "tool-use"],
                        "free_text": "",
                    },
                    "{}",
                ),
            ), patch(
                "handle_trigger._run_pipeline",
                return_value={"status": "success", "final_response": "NO_REPLY", "mode": "topic_path"},
            ) as pipeline_mock:
                result = handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 我做多模态智能体和 tool-use",
                    target="user:demo",
                    account_id="main",
                )
        self.assertEqual(result["mode"], "topic_path")
        _, kwargs = pipeline_mock.call_args
        self.assertEqual(kwargs["topics"], ["多模态智能体", "tool-use"])
        self.assertEqual(kwargs["free_text"], "")

    def test_handle_trigger_runs_pipeline_for_scholar_identity_without_topics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch(
                "handle_trigger._run_pipeline",
                return_value={"status": "success", "final_response": "NO_REPLY", "mode": "scholar_path"},
            ) as pipeline_mock:
                result = handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 我是张帆进，清华大学。推荐论文",
                    target="user:demo",
                    account_id="main",
                )
        self.assertEqual(result["mode"], "scholar_path")
        _, kwargs = pipeline_mock.call_args
        self.assertEqual(kwargs["scholar_name"], "张帆进")
        self.assertEqual(kwargs["scholar_org"], "清华大学")
        self.assertEqual(kwargs["free_text"], "")

    def test_handle_trigger_runs_pipeline_for_bare_name_and_org_identity_without_llm_parse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("handle_trigger.llm_parse_profile_input") as parse_mock, patch(
                "handle_trigger._run_pipeline",
                return_value={"status": "success", "final_response": "NO_REPLY", "mode": "scholar_path"},
            ) as pipeline_mock:
                result = handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 李涓子，清华大学。推荐论文",
                    target="user:demo",
                    account_id="main",
                )
        self.assertEqual(result["mode"], "scholar_path")
        _, kwargs = pipeline_mock.call_args
        self.assertEqual(kwargs["scholar_name"], "李涓子")
        self.assertEqual(kwargs["scholar_org"], "清华大学")
        self.assertEqual(kwargs["free_text"], "")
        parse_mock.assert_not_called()

    def test_handle_trigger_uses_inferred_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch(
                "handle_trigger._run_pipeline",
                return_value={"status": "success", "final_response": "NO_REPLY", "mode": "scholar_path"},
            ) as pipeline_mock:
                handle_trigger(
                    base_dir=base_dir,
                    text='{"sender_id":"ou_123"}\n/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态',
                )
        _, kwargs = pipeline_mock.call_args
        self.assertEqual(kwargs["target"], "user:ou_123")
        self.assertEqual(kwargs["account_id"], "default")

    def test_handle_trigger_uses_recent_feishu_session_route_when_wrapper_missing(self) -> None:
        fake_sessions = {
            "agent:main:main": {"updatedAt": 1},
            "agent:main:feishu:direct:ou_recent": {"updatedAt": 10},
            "agent:main:feishu:direct:ou_old": {"updatedAt": 5},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("handle_trigger.Path.exists", return_value=True), patch(
                "handle_trigger.Path.read_text",
                return_value=__import__("json").dumps(fake_sessions, ensure_ascii=False),
            ), patch(
                "handle_trigger._run_pipeline",
                return_value={"status": "success", "final_response": "NO_REPLY", "mode": "scholar_path"},
            ) as pipeline_mock:
                handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态",
                )
        _, kwargs = pipeline_mock.call_args
        self.assertEqual(kwargs["target"], "user:ou_recent")
        self.assertEqual(kwargs["account_id"], "default")

    def test_merge_llm_config_uses_openclaw_provider_defaults(self) -> None:
        with patch(
            "handle_trigger._load_openclaw_runtime_config",
            return_value={
                "api_key": "demo-key",
                "base_url": "https://api.example.com/v1",
                "model": "demo-model",
            },
        ):
            merged = _merge_llm_config({"llm": {"api_key": "", "base_url": "", "model": ""}})
        self.assertEqual(merged["llm"]["api_key"], "demo-key")
        self.assertEqual(merged["llm"]["base_url"], "https://api.example.com/v1")
        self.assertEqual(merged["llm"]["model"], "demo-model")

    def test_merge_llm_config_replaces_placeholder_model(self) -> None:
        with patch(
            "handle_trigger._load_openclaw_runtime_config",
            return_value={
                "api_key": "demo-key",
                "base_url": "https://api.example.com/v1",
                "model": "demo-model",
            },
        ):
            merged = _merge_llm_config({"llm": {"api_key": "", "base_url": "", "model": "gpt-5-mini"}})
        self.assertEqual(merged["llm"]["model"], "demo-model")

    def test_merge_llm_config_keeps_explicit_values(self) -> None:
        with patch(
            "handle_trigger._load_openclaw_runtime_config",
            return_value={
                "api_key": "demo-key",
                "base_url": "https://api.example.com/v1",
                "model": "demo-model",
            },
        ):
            merged = _merge_llm_config(
                {"llm": {"api_key": "explicit-key", "base_url": "https://custom.example/v1", "model": "custom-model"}}
            )
        self.assertEqual(merged["llm"]["api_key"], "explicit-key")
        self.assertEqual(merged["llm"]["base_url"], "https://custom.example/v1")
        self.assertEqual(merged["llm"]["model"], "custom-model")

    def test_handle_trigger_prompts_topics_when_uid_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("handle_trigger._run_pipeline", side_effect=RuntimeError("profile_unavailable")):
                result = handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态",
                    target="user:demo",
                    account_id="main",
                )
        self.assertEqual(result["mode"], "error")
        self.assertEqual(result["final_response"], "TEXT")
        self.assertIn("profile_unavailable", result["reply_text"])

    def test_handle_trigger_uid_only_reports_downstream_error_instead_of_topics_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("handle_trigger._run_pipeline", side_effect=RuntimeError("arxiv_unreachable:ssl")):
                result = handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 aminer_user_id: 578f672b9ed5db014c418caf",
                    target="user:demo",
                    account_id="main",
                )
        self.assertEqual(result["mode"], "error")
        self.assertIn("arxiv_unreachable", result["reply_text"])

    def test_handle_trigger_prompts_when_scholar_resolution_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("handle_trigger._run_pipeline", side_effect=RuntimeError("profile_unavailable")):
                result = handle_trigger(
                    base_dir=base_dir,
                    text="/aminer-rec5 我是张帆进，清华大学。推荐论文",
                    target="user:demo",
                    account_id="main",
                )
        self.assertEqual(result["mode"], "onboarding_prompt")
        self.assertIn("研究方向", result["reply_text"])


if __name__ == "__main__":
    unittest.main()
