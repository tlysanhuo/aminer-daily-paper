from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.build_user_profile import build_topics_profile, build_user_profile


class BuildUserProfileTests(unittest.TestCase):
    def test_build_topics_profile_generates_categories(self) -> None:
        profile = build_topics_profile(["multimodal", "multi-agent"], config={})
        self.assertEqual(profile["status"], "success")
        self.assertIn("cs.CV", profile["arxiv_categories"])
        self.assertIn("cs.MA", profile["arxiv_categories"])
        self.assertTrue(profile["is_cs_user"])
        self.assertEqual(profile["recall_primary_source"], "arxiv")
        self.assertEqual(profile["recall_secondary_source"], "aminer")

    def test_build_topics_profile_marks_non_cs_users_without_forcing_arxiv_categories(self) -> None:
        with patch(
            "scripts.build_user_profile.call_segmentation_pro",
            return_value={"keywords": {"high": [["protein folding"], ["drug discovery"]]}, "params": {}},
        ):
            profile = build_topics_profile(["protein folding", "drug discovery"], config={})

        self.assertEqual(profile["status"], "success")
        self.assertEqual(profile["arxiv_categories"], [])
        self.assertFalse(profile["is_cs_user"])
        self.assertEqual(profile["recall_primary_source"], "aminer")
        self.assertEqual(profile["recall_secondary_source"], "arxiv")

    def test_build_topics_profile_uses_segmentation_and_filters_generic_intent_words(self) -> None:
        with patch(
            "scripts.build_user_profile.call_segmentation_pro",
            return_value={
                "keywords": {
                    "high": [["多模态", "multimodal"], ["推荐", "recommendation"]],
                    "middle": [["多模态学习", "multimodal learning"]],
                },
                "params": {},
            },
        ):
            profile = build_topics_profile(["多模态", "推荐"], config={})

        self.assertEqual(profile["topics"], ["多模态"])
        self.assertIn("multimodal", profile["keywords"])
        self.assertIn("multimodal learning", profile["keywords"])
        self.assertNotIn("推荐", profile["keywords"])
        self.assertNotIn("recommendation", profile["keywords"])
        self.assertEqual(profile["source_metadata"]["segmented_keyword_count"], 4)

    def test_build_topics_profile_filters_sentence_fragment_topics(self) -> None:
        with patch(
            "scripts.build_user_profile.call_segmentation_pro",
            return_value={"keywords": {"high": [["we propose"], ["Entity linking system"], ["show that our"]]}, "params": {}},
        ):
            profile = build_topics_profile(["we propose", "Entity linking system", "show that our"], config={})

        self.assertEqual(profile["topics"], ["Entity linking system"])
        self.assertIn("Entity linking system", profile["keywords"])
        self.assertNotIn("we propose", profile["keywords"])
        self.assertNotIn("show that our", profile["keywords"])

    def test_build_user_profile_uses_segmentationpro_signals(self) -> None:
        with patch(
            "scripts.build_user_profile.load_internal_uid_profile",
            return_value={
                "status": "success",
                "user_name": "测试用户",
                "bind_scholar_ids": ["abc123"],
                "topics": ["多模态"],
                "keywords": ["vision-language model"],
                "preferred_authors": ["Alice"],
                "preferred_venues": ["NeurIPS"],
                "seed_papers": [
                    {"title": "Seed Paper A", "keywords": ["multimodal"], "abstract": "seed"},
                    {"title": "Seed Paper B", "keywords": ["vision"], "abstract": "seed"},
                    {"title": "Seed Paper C", "keywords": ["language"], "abstract": "seed"},
                ],
                "source_metadata": {"source": "authored_papers_bind_profile"},
            },
        ), patch(
            "scripts.build_user_profile.llm_profile_topics",
            return_value=(
                [
                    {"name": "多模态推理", "keywords": ["multimodal reasoning", "vision-language reasoning"], "rationale": "来自多篇论文"},
                    {"name": "视觉语言模型", "keywords": ["vision-language model"], "rationale": "方向稳定"},
                ],
                '{"topics":[{"name":"多模态推理"}]}',
            ),
        ), patch(
            "scripts.build_user_profile.call_segmentation_pro",
            return_value={
                "keywords": {
                    "high": [["多模态", "multimodal"], ["智能体", "agent"]],
                    "middle": [["预训练", "pre-training"]],
                },
                "params": {
                    "person": [["Yuqing Wang"]],
                    "conference": "CVPR",
                },
            },
        ):
            profile = build_user_profile(
                "696259801cb939bc391d3a37",
                ["多模态", "智能体"],
                config={"llm": {"api_key": "demo-key", "model": "demo-model", "base_url": "", "timeout_seconds": 30}},
            )

        self.assertEqual(profile["status"], "success")
        self.assertEqual(profile["profile_name"], "测试用户")
        self.assertIn("多模态", profile["keywords"])
        self.assertIn("multimodal", profile["keywords"])
        self.assertIn("vision-language model", profile["keywords"])
        self.assertIn("多模态推理", profile["topics"])
        self.assertIn(profile["keywords"][0], {"pre-training", "agent", "multimodal", "multimodal reasoning"})
        self.assertEqual(profile["bind_scholar_ids"], ["abc123"])
        self.assertEqual(profile["seed_papers"][0]["title"], "Seed Paper A")
        self.assertIn("Alice", profile["preferred_authors"])
        self.assertIn("Yuqing Wang", profile["preferred_authors"])
        self.assertIn("NeurIPS", profile["preferred_venues"])
        self.assertIn("CVPR", profile["preferred_venues"])
        self.assertEqual(profile["source_metadata"]["source"], "authored_papers_bind_profile")
        self.assertEqual(profile["source_metadata"]["llm_topic_reason"], "")
        self.assertEqual(profile["source_metadata"]["internal_profile"]["llm_topics"][0]["name"], "多模态推理")
        self.assertTrue(profile["is_cs_user"])
        self.assertEqual(profile["recall_primary_source"], "arxiv")
        self.assertEqual(profile["recall_secondary_source"], "aminer")

    def test_build_user_profile_can_use_internal_uid_profile_without_explicit_topics(self) -> None:
        with patch(
            "scripts.build_user_profile.load_internal_uid_profile",
            return_value={
                "status": "success",
                "user_name": "测试用户",
                "bind_scholar_ids": ["abc123"],
                "topics": ["RLHF", "Multi-agent"],
                "keywords": ["multi-agent", "rlhf"],
                "preferred_authors": [],
                "preferred_venues": [],
                "seed_papers": [],
                "source_metadata": {"source": "experts_topic_profile"},
            },
        ), patch(
            "scripts.build_user_profile.call_segmentation_pro",
            return_value={"keywords": {"high": [["multi-agent"]]}, "params": {}},
        ):
            profile = build_user_profile(
                "696259801cb939bc391d3a37",
                [],
                config={"llm": {"api_key": "demo-key", "model": "demo-model", "base_url": "", "timeout_seconds": 30}},
            )
        self.assertEqual(profile["status"], "success")
        self.assertIn("RLHF", profile["topics"])
        self.assertIn("Multi-agent", profile["topics"])
        self.assertEqual(profile["source_metadata"]["llm_topic_reason"], "insufficient_seed_papers")
        self.assertTrue(profile["is_cs_user"])

    def test_build_user_profile_degrades_when_segmentation_is_unavailable(self) -> None:
        with patch(
            "scripts.build_user_profile.load_internal_uid_profile",
            return_value={
                "status": "success",
                "user_name": "测试用户",
                "bind_scholar_ids": [],
                "topics": ["多模态"],
                "keywords": ["multimodal"],
                "preferred_authors": [],
                "preferred_venues": [],
                "seed_papers": [],
                "source_metadata": {"source": "experts_topic_profile"},
            },
        ), patch("scripts.build_user_profile.call_segmentation_pro", side_effect=RuntimeError("timeout")):
            profile = build_user_profile("696259801cb939bc391d3a37", ["多模态"])
        self.assertEqual(profile["status"], "success")
        self.assertIn("segmentation_unavailable", profile["source_metadata"]["degraded_reason"])

    def test_build_user_profile_falls_back_when_llm_topic_inference_fails(self) -> None:
        with patch(
            "scripts.build_user_profile.load_internal_uid_profile",
            return_value={
                "status": "success",
                "user_name": "测试用户",
                "bind_scholar_ids": ["abc123"],
                "topics": ["自然语言处理"],
                "keywords": ["named entity recognition"],
                "preferred_authors": [],
                "preferred_venues": [],
                "seed_papers": [
                    {"title": "A", "keywords": ["ner"], "abstract": "seed"},
                    {"title": "B", "keywords": ["ie"], "abstract": "seed"},
                    {"title": "C", "keywords": ["information extraction"], "abstract": "seed"},
                ],
                "source_metadata": {"source": "authored_papers_bind_profile"},
            },
        ), patch(
            "scripts.build_user_profile.llm_profile_topics",
            side_effect=__import__("scripts.llm_client", fromlist=["ProfileTopicGenerationError"]).ProfileTopicGenerationError("llm_timeout"),
        ), patch(
            "scripts.build_user_profile.call_segmentation_pro",
            return_value={"keywords": {"high": [["named entity recognition"]]}, "params": {}},
        ):
            profile = build_user_profile(
                "696259801cb939bc391d3a37",
                [],
                config={"llm": {"api_key": "demo-key", "model": "demo-model", "base_url": "", "timeout_seconds": 30}},
            )
        self.assertEqual(profile["status"], "success")
        self.assertIn("自然语言处理", profile["topics"])
        self.assertEqual(profile["source_metadata"]["llm_topic_reason"], "primary:llm_timeout")

    def test_build_user_profile_degrades_when_uid_has_no_internal_signal_and_no_topics(self) -> None:
        with patch(
            "scripts.build_user_profile.load_internal_uid_profile",
            return_value={"status": "degraded", "source_metadata": {"reason": "no_bind_papers_or_experts_topic"}},
        ):
            profile = build_user_profile("696259801cb939bc391d3a37", [])
        self.assertEqual(profile["status"], "degraded")
        self.assertEqual(profile["source_metadata"]["reason"], "no_bind_papers_or_experts_topic")


if __name__ == "__main__":
    unittest.main()
