from __future__ import annotations

import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.feishu_cards import build_paper_card, render_summary_blocks
from scripts.render_feishu_messages import render_feishu_messages


class FeishuCardTests(unittest.TestCase):
    def test_render_summary_blocks_merges_structured_summary_into_single_paragraph(self) -> None:
        paper = {
            "structured_summary": {
                "research_problem": "离散扩散模型的解码方法研究不足。",
                "research_challenge": "标准 beam search 不适用于迭代去噪框架。",
                "research_method": "提出 D5P4，将选择步骤建模为 DPP 的 MAP 推理。",
                "experimental_results": "在自由形式生成和问答任务上显著提升批量多样性。",
            }
        }
        blocks = render_summary_blocks(paper)
        self.assertEqual(len(blocks), 1)
        content = blocks[0]["text"]["content"]
        self.assertTrue(content.startswith("**小结**"))
        self.assertNotIn("研究问题", content)
        self.assertNotIn("研究挑战", content)
        self.assertNotIn("研究方法", content)
        self.assertNotIn("实验效果", content)
        self.assertIn("离散扩散模型的解码方法研究不足", content)
        self.assertIn("标准 beam search 不适用于迭代去噪框架", content)
        self.assertIn("提出 D5P4", content)
        self.assertIn("显著提升批量多样性", content)

    def test_build_paper_card_uses_existing_famous_author_text(self) -> None:
        card = build_paper_card(
            1,
            {
                "title": "Demo",
                "year": 2025,
                "keywords": ["multimodal"],
                "summary": "研究问题：A\n研究方法：B",
                "authors": ["Xiang Bai"],
                "author_entries": [
                    {
                        "display_name": "Xiang Bai",
                        "profile_url": "https://www.aminer.cn/profile/53f45a3ddabfaee02ad67536",
                    }
                ],
                "aminer_author_profiles": [],
                "famous_authors": ["白翔，来自华中科技大学，h-index 为 105。"],
                "aminer_paper_url": "https://www.aminer.cn/pub/demo-paper",
                "abs_url": "https://arxiv.org/abs/2501.00001",
            },
            [],
        )
        elements = card["elements"]
        self.assertTrue(elements[0]["text"]["content"].startswith("**关键词**"))
        year_block = next(item for item in elements if item["text"]["content"].startswith("**年份**"))
        self.assertIn("2025", year_block["text"]["content"])
        summary_block = next(item for item in elements if item["text"]["content"].startswith("**小结**"))
        self.assertIn("A", summary_block["text"]["content"])
        famous_block = next(item for item in elements if item["text"]["content"].startswith("**大牛作者**"))
        self.assertIn("白翔", famous_block["text"]["content"])
        links_block = next(item for item in elements if item["text"]["content"].startswith("**AMiner 论文链接**"))
        self.assertIn("[查看论文](https://www.aminer.cn/pub/demo-paper)", links_block["text"]["content"])

    def test_build_paper_card_links_famous_author_when_aminer_profile_exists(self) -> None:
        card = build_paper_card(
            1,
            {
                "title": "Demo",
                "keywords": ["recsys"],
                "summary": "研究问题：A\n研究方法：B",
                "authors": ["Meng Wang", "Zhiyong Cheng"],
                "author_entries": [
                    {"display_name": "Meng Wang", "profile_url": "https://www.aminer.cn/profile/meng-wang"},
                    {"display_name": "Zhiyong Cheng", "profile_url": "https://www.aminer.cn/profile/zhiyong-cheng"},
                ],
                "aminer_author_profiles": [
                    {"name": "Meng Wang", "profile_url": "https://www.aminer.cn/profile/meng-wang"},
                    {"name": "Zhiyong Cheng", "profile_url": "https://www.aminer.cn/profile/zhiyong-cheng"},
                ],
                "famous_authors": [
                    "Meng Wang: 教授，博士生导师，合肥工业大学计算机科学与信息工程学院，h-index 108",
                    "Zhiyong Cheng: 教授，合肥工业大学，h-index 41",
                ],
                "aminer_paper_url": "https://www.aminer.cn/pub/demo-paper",
                "abs_url": "https://arxiv.org/abs/2501.00001",
            },
            [],
        )
        contents = [item["text"]["content"] for item in card["elements"]]
        famous_block = next(content for content in contents if content.startswith("**大牛作者**"))
        self.assertIn("[Meng Wang](https://www.aminer.cn/profile/meng-wang)", famous_block)
        self.assertTrue(any("[Zhiyong Cheng](https://www.aminer.cn/profile/zhiyong-cheng)" in content for content in contents))

    def test_build_paper_card_omits_famous_author_block_when_empty(self) -> None:
        card = build_paper_card(
            1,
            {
                "title": "Demo",
                "keywords": ["multimodal"],
                "summary": "研究问题：A\n研究方法：B",
                "authors": ["Alice"],
                "author_entries": [{"display_name": "Alice", "profile_url": ""}],
                "aminer_author_profiles": [],
                "famous_authors": [],
                "aminer_paper_url": "https://www.aminer.cn/pub/69bc9f479be8eb7c4b4c72e5/retrieval-augmented-llm-agents-learning-to-learn-from-experience",
                "abs_url": "https://arxiv.org/abs/2501.00001",
            },
            [],
        )
        contents = [item["text"]["content"] for item in card["elements"]]
        self.assertFalse(any(content.startswith("**大牛作者**") for content in contents))
        links_block = next(content for content in contents if content.startswith("**AMiner 论文链接**"))
        self.assertIn("[查看论文](https://www.aminer.cn/pub/69bc9f479be8eb7c4b4c72e5/retrieval-augmented-llm-agents-learning-to-learn-from-experience)", links_block)

    def test_build_paper_card_shows_degraded_block_and_falls_back_to_arxiv_link(self) -> None:
        card = build_paper_card(
            1,
            {
                "title": "Demo",
                "keywords": ["sociology"],
                "summary": "研究问题：A\n研究方法：B",
                "authors": ["Alice"],
                "author_entries": [{"display_name": "Alice", "profile_url": ""}],
                "aminer_author_profiles": [],
                "famous_authors": [],
                "aminer_paper_url": "https://www.aminer.cn/search?t=pub&q=Demo",
                "abs_url": "https://arxiv.org/abs/2501.00001",
            },
            ["Summary=missing_api_key"],
        )
        contents = [item["text"]["content"] for item in card["elements"]]
        self.assertFalse(any(content.startswith("**降级说明**") for content in contents))
        links_block = next(content for content in contents if content.startswith("**AMiner 论文链接**"))
        self.assertIn("[查看论文](https://www.aminer.cn/search?t=pub&q=Demo)", links_block)

    def test_render_feishu_messages_follows_reference_project_message_payload(self) -> None:
        payload = {
            "status": "success",
            "degraded_reason": "Summary=missing_api_key",
            "profile_topics": ["多模态智能体", "tool-use"],
            "profile_name": "张帆进",
            "profile_source": "scholar_path",
            "papers": [
                {
                    "title": "Demo One",
                    "year": 2024,
                    "keywords": ["multimodal"],
                    "summary": "研究问题：A\n研究方法：B",
                    "authors": ["Alice"],
                    "author_entries": [{"display_name": "Alice", "profile_url": ""}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "aminer_paper_url": "https://www.aminer.cn/pub/demo-one",
                    "abs_url": "https://arxiv.org/abs/2501.00001",
                },
                {
                    "title": "Demo Two",
                    "published": "2023-05-01",
                    "keywords": ["agents"],
                    "summary": "研究问题：C\n研究方法：D",
                    "authors": ["Bob"],
                    "author_entries": [{"display_name": "Bob", "profile_url": ""}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "aminer_paper_url": "https://www.aminer.cn/pub/demo-two",
                    "abs_url": "https://arxiv.org/abs/2501.00002",
                },
            ],
        }
        rendered = render_feishu_messages(payload)
        first_card = rendered["messages"][0]["card_json"]
        second_card = rendered["messages"][1]["card_json"]
        self.assertNotIn("当前推荐方向", first_card)
        self.assertNotIn("当前推荐方向", second_card)
        self.assertNotIn("推荐理由", first_card)
        self.assertNotIn("研究问题", first_card)
        self.assertIn("小结", first_card)
        self.assertIn("年份", first_card)
        self.assertIn("2024", first_card)
        self.assertIn("2023", second_card)
        self.assertIn("A", first_card)
        self.assertNotIn("降级说明", first_card)
        self.assertNotIn("降级说明", second_card)
        self.assertEqual(rendered["degraded_reasons"], ["Summary=missing_api_key"])
        self.assertEqual(rendered["profile_topics"], ["多模态智能体", "tool-use"])


if __name__ == "__main__":
    unittest.main()
