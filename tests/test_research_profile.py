from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.research_profile import build_research_profile


class ResearchProfileTests(unittest.TestCase):
    def test_build_research_profile_uses_free_text_topics(self) -> None:
        with patch(
            "scripts.research_profile.call_segmentation_pro",
            return_value={"keywords": {"high": [["多模态智能体"], ["tool use"]]}},
        ):
            profile = build_research_profile(free_text="我做多模态智能体和 tool use", config={})
        self.assertEqual(profile["status"], "success")
        self.assertEqual(profile["profile_mode"], "topic_path")
        self.assertTrue(profile["topics"])
        self.assertNotIn("多模态智能体和 tool use", profile["topics"])

    def test_build_research_profile_builds_scholar_profile_from_manual_papers(self) -> None:
        with patch(
            "scripts.research_profile.search_papers_pro",
            return_value={
                "papers": [
                    {
                        "title": "Paper A",
                        "abstract": "multimodal planning for agents",
                        "keywords": ["multimodal planning", "agents"],
                        "authors": ["Jie Tang", "Alice"],
                        "aminer_paper_id": "paper-a",
                    }
                ]
            },
        ), patch(
            "scripts.research_profile.enrich_ranked_payload_with_aminer_details",
            side_effect=lambda payload, token="": {
                **payload,
                "papers": [
                    {
                        **payload["papers"][0],
                        "aminer_author_profiles": [
                            {"name": "Jie Tang", "affiliation": "Tsinghua University", "interests": ["academic graph mining"]},
                        ],
                        "venue": "KDD",
                        "year": 2025,
                        "citations": 10,
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.build_topics_profile",
            side_effect=lambda topics, config=None, enable_llm_topics=True: {
                "status": "success",
                "keywords": list(topics),
                "source_metadata": {"segmented_keyword_count": len(topics)},
            },
        ):
            profile = build_research_profile(
                scholar_name="Jie Tang",
                scholar_org="Tsinghua University",
                paper_titles=["Paper A"],
                config={"aminer": {"token": "demo"}, "llm": {"api_key": ""}},
            )
        self.assertEqual(profile["status"], "success")
        self.assertEqual(profile["profile_mode"], "scholar_path")
        self.assertEqual(profile["profile_name"], "Jie Tang")
        self.assertTrue(profile["seed_papers"])

    def test_build_research_profile_loads_local_papers_file(self) -> None:
        payload = {"papers": [{"title": "Graph Mining", "abstract": "academic graph mining", "keywords": ["graph mining"], "authors": ["A", "B"]}]}
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "papers.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch(
                "scripts.research_profile.build_topics_profile",
                side_effect=lambda topics, config=None, enable_llm_topics=True: {
                    "status": "success",
                    "keywords": list(topics),
                    "source_metadata": {"segmented_keyword_count": len(topics)},
                },
            ):
                profile = build_research_profile(papers_file=str(path), config={"llm": {"api_key": ""}})
        self.assertEqual(profile["status"], "success")
        self.assertEqual(profile["profile_mode"], "scholar_path")

    def test_build_research_profile_resolves_scholar_identity_via_person_search(self) -> None:
        with patch(
            "scripts.research_profile.search_persons",
            return_value={
                "status": "success",
                "persons": [
                    {
                        "id": "person-1",
                        "name": "Fanjin Zhang",
                        "name_zh": "张帆进",
                        "display_name": "张帆进",
                        "org": "Department of Computer Science and Technology, Tsinghua University",
                        "org_zh": "清华大学计算机科学与技术系",
                        "n_citation": 3256,
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.search_person_papers",
            return_value={
                "status": "success",
                "papers": [
                    {
                        "title": "OAG: Linking Entities Across Large-Scale Heterogeneous Knowledge Graphs",
                        "abstract": "academic graph mining and knowledge graph integration",
                        "keywords": ["academic graph mining", "knowledge graph"],
                        "authors": ["Fanjin Zhang", "Jie Tang"],
                        "author_entries": [{"display_name": "Fanjin Zhang", "profile_url": "https://www.aminer.cn/profile/person-1", "is_disambiguated": True}],
                        "aminer_author_profiles": [],
                        "famous_authors": [],
                        "aminer_paper_id": "paper-1",
                        "aminer_paper_url": "https://www.aminer.cn/pub/paper-1",
                        "venue": "TKDE",
                        "year": 2023,
                        "citations": 28,
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.enrich_ranked_payload_with_aminer_details",
            side_effect=lambda payload, token="": payload,
        ), patch(
            "scripts.research_profile.build_topics_profile",
            side_effect=lambda topics, config=None, enable_llm_topics=True: {
                "status": "success",
                "keywords": list(topics),
                "source_metadata": {"segmented_keyword_count": len(topics)},
            },
        ):
            profile = build_research_profile(
                scholar_name="张帆进",
                scholar_org="清华大学",
                config={"aminer": {"token": "demo"}, "llm": {"api_key": ""}},
            )
        self.assertEqual(profile["status"], "success")
        self.assertEqual(profile["profile_mode"], "scholar_path")
        self.assertEqual(profile["bind_scholar_ids"], ["person-1"])
        self.assertEqual(profile["profile_name"], "张帆进")
        self.assertTrue(profile["seed_papers"])
        self.assertTrue(profile["research_domains"])
        self.assertIn("excluded_keywords", profile)

    def test_build_research_profile_prefers_resolved_person_interests_over_sentence_fragments(self) -> None:
        with patch(
            "scripts.research_profile.search_persons",
            return_value={
                "status": "success",
                "persons": [
                    {
                        "id": "person-2",
                        "name": "Juanzi Li",
                        "name_zh": "李涓子",
                        "display_name": "李涓子",
                        "org": "Department of Computer Science and Technology, Tsinghua University",
                        "org_zh": "清华大学计算机科学与技术系计算机软件研究所",
                        "n_citation": 27877,
                        "interests": [
                            "Topic Modeling",
                            "Language Modeling",
                            "Named Entity Recognition",
                            "Knowledge Graph Embedding",
                            "Semantic Web",
                        ],
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.search_person_papers",
            return_value={
                "status": "success",
                "papers": [
                    {
                        "title": "XLink: an Unsupervised Bilingual Entity Linking System",
                        "abstract": "We propose a novel entity linking system and show that our approach works well.",
                        "keywords": ["we propose", "Entity linking system", "show that our"],
                        "authors": ["Juanzi Li", "Alice"],
                        "author_entries": [{"display_name": "Juanzi Li", "profile_url": "https://www.aminer.cn/profile/person-2"}],
                        "aminer_author_profiles": [],
                        "famous_authors": [],
                        "aminer_paper_id": "paper-2",
                        "aminer_paper_url": "https://www.aminer.cn/pub/paper-2",
                        "venue": "ACL",
                        "year": 2024,
                        "citations": 28,
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.enrich_ranked_payload_with_aminer_details",
            side_effect=lambda payload, token="": payload,
        ), patch(
            "scripts.research_profile.build_topics_profile",
            side_effect=lambda topics, config=None, enable_llm_topics=True: {
                "status": "success",
                "keywords": list(topics),
                "source_metadata": {"segmented_keyword_count": len(topics)},
            },
        ):
            profile = build_research_profile(
                scholar_name="李涓子",
                scholar_org="清华大学",
                config={"aminer": {"token": "demo"}, "llm": {"api_key": ""}},
            )
        self.assertEqual(profile["status"], "success")
        self.assertIn("Topic Modeling", profile["topics"])
        self.assertIn("Language Modeling", profile["topics"])
        self.assertNotIn("we propose", profile["topics"])
        self.assertNotIn("show that our", profile["topics"])

    def test_build_research_profile_skips_llm_topics_when_resolved_person_has_rich_interests(self) -> None:
        with patch(
            "scripts.research_profile.search_persons",
            return_value={
                "status": "success",
                "persons": [
                    {
                        "id": "person-3",
                        "name": "Juanzi Li",
                        "name_zh": "李涓子",
                        "display_name": "李涓子",
                        "org": "Tsinghua University",
                        "org_zh": "清华大学",
                        "n_citation": 27877,
                        "interests": ["Topic Modeling", "Language Modeling", "Named Entity Recognition", "Semantic Web"],
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.search_person_papers",
            return_value={
                "status": "success",
                "papers": [
                    {
                        "title": "XLink: an Unsupervised Bilingual Entity Linking System",
                        "abstract": "We propose a novel entity linking system and show that our approach works well.",
                        "keywords": ["Entity linking system"],
                        "authors": ["Juanzi Li", "Alice"],
                        "author_entries": [{"display_name": "Juanzi Li", "profile_url": "https://www.aminer.cn/profile/person-3"}],
                        "aminer_author_profiles": [],
                        "famous_authors": [],
                        "aminer_paper_id": "paper-3",
                        "aminer_paper_url": "https://www.aminer.cn/pub/paper-3",
                        "venue": "ACL",
                        "year": 2024,
                        "citations": 28,
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.enrich_ranked_payload_with_aminer_details",
            side_effect=lambda payload, token="": payload,
        ), patch(
            "scripts.research_profile._maybe_apply_llm_topics",
            side_effect=AssertionError("llm topics should be skipped"),
        ), patch(
            "scripts.research_profile.build_topics_profile",
            side_effect=lambda topics, config=None, enable_llm_topics=True: {
                "status": "success",
                "keywords": list(topics),
                "source_metadata": {"segmented_keyword_count": len(topics)},
            },
        ):
            profile = build_research_profile(
                scholar_name="李涓子",
                scholar_org="清华大学",
                config={"aminer": {"token": "demo"}, "llm": {"api_key": "demo"}},
            )
        self.assertEqual(profile["status"], "success")
        components = list(((profile.get("source_metadata") or {}).get("components") or []))
        self.assertEqual(components[0].get("llm_topic_reason"), "skipped_resolved_person_interests")

    def test_build_research_profile_generates_generic_retrieval_signal(self) -> None:
        with patch(
            "scripts.research_profile.search_persons",
            return_value={
                "status": "success",
                "persons": [
                    {
                        "id": "person-4",
                        "name": "Fanjin Zhang",
                        "name_zh": "张帆进",
                        "display_name": "张帆进",
                        "org": "Tsinghua University",
                        "org_zh": "清华大学",
                        "n_citation": 3256,
                        "interests": [
                            "Named Entity Recognition",
                            "Academic Graph Mining",
                            "Benchmark",
                            "Academic Knowledge Graph",
                            "Contrastive Learning",
                        ],
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.search_person_papers",
            return_value={
                "status": "success",
                "papers": [
                    {
                        "title": "OAG-Bench: A Human-Curated Benchmark for Academic Graph Mining",
                        "abstract": "We conduct extensive experiments for academic graph mining and named entity disambiguation.",
                        "keywords": ["academic graph mining", "extensive experiments", "name disambiguation"],
                        "authors": ["Fanjin Zhang", "Jie Tang"],
                        "author_entries": [{"display_name": "Fanjin Zhang", "profile_url": "https://www.aminer.cn/profile/person-4"}],
                        "aminer_author_profiles": [],
                        "famous_authors": [],
                        "aminer_paper_id": "paper-4",
                        "aminer_paper_url": "https://www.aminer.cn/pub/paper-4",
                        "venue": "KDD",
                        "year": 2024,
                        "citations": 28,
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.enrich_ranked_payload_with_aminer_details",
            side_effect=lambda payload, token="": payload,
        ), patch(
            "scripts.research_profile.build_topics_profile",
            side_effect=lambda topics, config=None, enable_llm_topics=True: {
                "status": "success",
                "keywords": list(topics),
                "source_metadata": {"segmented_keyword_count": len(topics)},
            },
        ):
            profile = build_research_profile(
                scholar_name="张帆进",
                scholar_org="清华大学",
                config={"aminer": {"token": "demo"}, "llm": {"api_key": ""}},
            )
        self.assertEqual(profile["status"], "success")
        self.assertIn("Named Entity Recognition", profile["retrieval_topics"])
        self.assertIn("Academic Knowledge Graph", profile["retrieval_topics"])
        self.assertTrue(any(item in profile["retrieval_topics"] for item in ("OAG", "name disambiguation")))
        self.assertNotIn("Benchmark", profile["retrieval_topics"])
        self.assertNotIn("Contrastive Learning", profile["retrieval_topics"])
        self.assertNotIn("extensive experiments", [item.casefold() for item in profile["retrieval_keywords"]])
        weights = dict(profile.get("retrieval_term_weights") or {})
        self.assertGreater(weights.get("Named Entity Recognition", 0.0), weights.get("Contrastive Learning", 0.0))

    def test_build_research_profile_applies_llm_scholar_term_labels(self) -> None:
        with patch(
            "scripts.research_profile.search_persons",
            return_value={
                "status": "success",
                "persons": [
                    {
                        "id": "person-5",
                        "name": "Fanjin Zhang",
                        "name_zh": "张帆进",
                        "display_name": "张帆进",
                        "org": "Tsinghua University",
                        "org_zh": "清华大学",
                        "n_citation": 3256,
                        "interests": [
                            "Named Entity Recognition",
                            "Academic Graph Mining",
                            "Academic Knowledge Graph",
                            "OAG",
                        ],
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.search_person_papers",
            return_value={
                "status": "success",
                "papers": [
                    {
                        "title": "OAG-Bench: A Human-Curated Benchmark for Academic Graph Mining",
                        "abstract": "We conduct extensive experiments for academic graph mining and named entity disambiguation.",
                        "keywords": ["academic graph mining", "name disambiguation", "entity linking"],
                        "authors": ["Fanjin Zhang", "Jie Tang"],
                        "author_entries": [{"display_name": "Fanjin Zhang", "profile_url": "https://www.aminer.cn/profile/person-5"}],
                        "aminer_author_profiles": [],
                        "famous_authors": [],
                        "aminer_paper_id": "paper-5",
                        "aminer_paper_url": "https://www.aminer.cn/pub/paper-5",
                        "venue": "KDD",
                        "year": 2024,
                        "citations": 28,
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.enrich_ranked_payload_with_aminer_details",
            side_effect=lambda payload, token="": payload,
        ), patch(
            "scripts.research_profile.build_topics_profile",
            side_effect=lambda topics, config=None, enable_llm_topics=True: {
                "status": "success",
                "keywords": list(topics),
                "source_metadata": {"segmented_keyword_count": len(topics)},
            },
        ), patch(
            "scripts.research_profile._resolve_llm_candidates",
            return_value=[{"api_key": "demo", "base_url": "https://example.com", "model": "demo", "timeout_seconds": "10"}],
        ), patch(
            "scripts.research_profile.llm_label_scholar_terms",
            return_value=(
                [
                    {"term": "Named Entity Recognition", "role": "broad_superordinate", "weight": 0.3, "rationale": "上位任务词"},
                    {"term": "Academic Graph Mining", "role": "scholar_specific", "weight": 1.2, "rationale": "更贴近学者场景"},
                    {"term": "Academic Knowledge Graph", "role": "core_domain", "weight": 1.0, "rationale": "稳定主方向"},
                    {"term": "OAG", "role": "scholar_specific", "weight": 1.3, "rationale": "学者专属实体"},
                ],
                '{"labels":[]}',
            ),
        ):
            profile = build_research_profile(
                scholar_name="张帆进",
                scholar_org="清华大学",
                config={"aminer": {"token": "demo"}, "llm": {"api_key": "demo", "enable_scholar_term_labeling": True}},
            )
        weights = dict(profile.get("retrieval_term_weights") or {})
        self.assertGreater(weights.get("Academic Graph Mining", 0.0), weights.get("Named Entity Recognition", 0.0))
        self.assertGreater(weights.get("OAG", 0.0), weights.get("Named Entity Recognition", 0.0))
        labeling = dict((profile.get("source_metadata") or {}).get("scholar_term_labeling") or {})
        self.assertEqual(labeling.get("reason"), "success")

    def test_build_research_profile_mixes_recent_priority_with_high_citation_anchors(self) -> None:
        current_year = datetime.now(timezone.utc).year
        recent_papers = []
        for index in range(14):
            recent_papers.append(
                {
                    "title": f"Recent Paper {index}",
                    "abstract": "academic knowledge graph and entity linking",
                    "keywords": ["academic knowledge graph", "entity linking"],
                    "authors": ["Juanzi Li", "Alice"],
                    "author_entries": [{"display_name": "Juanzi Li", "profile_url": "https://www.aminer.cn/profile/person-6"}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "aminer_paper_id": f"recent-{index}",
                    "aminer_paper_url": f"https://www.aminer.cn/pub/recent-{index}",
                    "venue": "KDD",
                    "year": current_year - (index % 3),
                    "citations": 20 - index,
                }
            )
        older_high_citation = {
            "title": "Older Highly Cited Paper",
            "abstract": "legacy topic",
            "keywords": ["legacy topic"],
            "authors": ["Juanzi Li", "Bob"],
            "author_entries": [{"display_name": "Juanzi Li", "profile_url": "https://www.aminer.cn/profile/person-6"}],
            "aminer_author_profiles": [],
            "famous_authors": [],
            "aminer_paper_id": "older-1",
            "aminer_paper_url": "https://www.aminer.cn/pub/older-1",
            "venue": "ACL",
            "year": current_year - 6,
            "citations": 999,
        }
        with patch(
            "scripts.research_profile.search_persons",
            return_value={
                "status": "success",
                "persons": [
                    {
                        "id": "person-6",
                        "name": "Juanzi Li",
                        "name_zh": "李涓子",
                        "display_name": "李涓子",
                        "org": "Tsinghua University",
                        "org_zh": "清华大学",
                        "n_citation": 27877,
                        "interests": ["Knowledge Graph Embedding", "Semantic Web"],
                    }
                ],
            },
        ), patch(
            "scripts.research_profile.search_person_papers",
            return_value={"status": "success", "papers": recent_papers + [older_high_citation]},
        ), patch(
            "scripts.research_profile.enrich_ranked_payload_with_aminer_details",
            side_effect=lambda payload, token="": payload,
        ), patch(
            "scripts.research_profile.build_topics_profile",
            side_effect=lambda topics, config=None, enable_llm_topics=True: {
                "status": "success",
                "keywords": list(topics),
                "source_metadata": {"segmented_keyword_count": len(topics)},
            },
        ):
            profile = build_research_profile(
                scholar_name="李涓子",
                scholar_org="清华大学",
                config={
                    "aminer": {"token": "demo"},
                    "llm": {"api_key": ""},
                    "search": {"scholar_profile_recent_years": 3, "scholar_profile_max_papers": 12},
                },
            )
        seed_titles = [paper.get("title") for paper in list(profile.get("seed_papers") or [])]
        self.assertEqual(profile["status"], "success")
        self.assertEqual(len(seed_titles), 12)
        self.assertIn("Older Highly Cited Paper", seed_titles)
        recent_titles = {f"Recent Paper {index}" for index in range(14)}
        self.assertGreaterEqual(sum(1 for title in seed_titles if title in recent_titles), 8)
        self.assertEqual((profile.get("source_metadata") or {}).get("authored_paper_count"), 12)
        self.assertEqual((profile.get("source_metadata") or {}).get("recent_year_window"), 3)


if __name__ == "__main__":
    unittest.main()
