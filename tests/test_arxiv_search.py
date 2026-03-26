from __future__ import annotations

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
import urllib.error


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.arxiv_search import (
    default_recent_top_tier_quota,
    _build_query_term_plans,
    annotate_recent_top_tier_metadata,
    build_aminer_paper_url_for_arxiv_paper,
    build_arxiv_query,
    enrich_ranked_payload_with_aminer_paper_urls,
    identify_top_tier_venue,
    parse_arxiv_xml,
    rank_arxiv_candidates,
    rebalance_recent_top_tier_papers,
)


XML_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <title>Agentic Multimodal Planning</title>
    <summary>We study multimodal planning for multi-agent systems.</summary>
    <published>2026-03-20T00:00:00Z</published>
    <author><name>Alice</name></author>
    <category term="cs.AI" />
    <category term="cs.CV" />
    <link href="http://arxiv.org/pdf/2501.00001v1" title="pdf" />
  </entry>
</feed>
"""

XML_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""


class ArxivSearchTests(unittest.TestCase):
    def test_build_query_term_plans_prefers_profile_topics_and_filters_noise(self) -> None:
        profile = {
            "topics": ["LLM推理优化", "学术知识图谱", "we present", "长文本与信息抽取"],
            "keywords": ["academic graph mining", "we present", "name disambiguation", "extensive experiments"],
            "retrieval_topics": ["学术知识图谱", "长文本与信息抽取"],
            "retrieval_keywords": ["name disambiguation", "academic graph mining"],
            "retrieval_term_weights": {
                "name disambiguation": 4.0,
                "academic graph mining": 3.5,
            },
            "source_metadata": {
                "internal_profile": {
                    "llm_topics": [
                        {"name": "学术场景LLM应用", "keywords": ["LLM Agent", "Deep Research", "RPC-Bench"]},
                    ]
                }
            },
        }
        plans = _build_query_term_plans(profile, top_k=5)
        self.assertTrue(plans)
        first_plan = [item.lower() for item in plans[0]]
        self.assertIn("academic graph mining", first_plan)
        self.assertIn("name disambiguation", first_plan)
        self.assertIn("llm agent", first_plan)
        self.assertNotIn("we present", first_plan)
        self.assertNotIn("extensive experiments", first_plan)

    def test_build_query_term_plans_uses_llm_scholar_roles_for_primary_and_fallback(self) -> None:
        profile = {
            "profile_mode": "scholar_path",
            "retrieval_topics": [
                "OAG",
                "Academic Knowledge Graph",
                "Entity linking",
                "Named Entity Recognition",
            ],
            "retrieval_keywords": [
                "Academic Graph Mining",
                "Open Academic Graph",
                "knowledge graphs",
            ],
            "retrieval_term_weights": {
                "OAG": 7.5,
                "Open Academic Graph": 7.2,
                "Academic Knowledge Graph": 6.9,
                "Academic Graph Mining": 6.5,
                "Entity linking": 6.0,
                "Named Entity Recognition": 5.8,
                "knowledge graphs": 2.0,
            },
            "source_metadata": {
                "scholar_term_labeling": {
                    "reason": "success",
                    "labels": [
                        {"term": "OAG", "role": "scholar_specific", "weight": 1.5},
                        {"term": "Open Academic Graph", "role": "scholar_specific", "weight": 1.4},
                        {"term": "Academic Knowledge Graph", "role": "core_domain", "weight": 1.2},
                        {"term": "Academic Graph Mining", "role": "core_domain", "weight": 1.1},
                        {"term": "Entity linking", "role": "core_domain", "weight": 1.0},
                        {"term": "Named Entity Recognition", "role": "broad_superordinate", "weight": 1.0},
                        {"term": "knowledge graphs", "role": "broad_superordinate", "weight": 0.8},
                    ]
                }
            },
        }
        plans = _build_query_term_plans(profile, top_k=5)
        self.assertTrue(plans)
        self.assertEqual(plans[0], ["Open Academic Graph", "Academic Knowledge Graph", "Academic Graph Mining"])
        self.assertIn(["Entity linking"], plans)
        self.assertIn(["Named Entity Recognition", "knowledge graphs"], plans)

    def test_build_arxiv_query_includes_categories_keywords_and_date(self) -> None:
        query = build_arxiv_query(["cs.AI", "cs.CV"], ["multimodal", "agent"], 30)
        self.assertIn("cat:cs.AI", query)
        self.assertIn('all:"multimodal"', query)
        self.assertIn("submittedDate", query)

    def test_build_arxiv_query_without_categories_does_not_force_cs_default(self) -> None:
        query = build_arxiv_query([], ["protein folding"], 30)
        self.assertNotIn("cat:cs.AI", query)
        self.assertNotIn("cat:", query)
        self.assertIn('all:"protein folding"', query)

    def test_parse_arxiv_xml_extracts_entry(self) -> None:
        papers = parse_arxiv_xml(XML_SAMPLE)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["arxiv_id"], "2501.00001v1")
        self.assertEqual(papers[0]["authors"], ["Alice"])

    def test_fetch_arxiv_candidates_falls_back_to_http_when_https_fails(self) -> None:
        profile = {"arxiv_categories": ["cs.AI"], "keywords": ["multimodal"]}

        class _FakeResponse:
            def __init__(self, text: str) -> None:
                self._text = text

            def read(self) -> bytes:
                return self._text.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeOpener:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def open(self, url: str, timeout: int = 60):
                self.calls.append(url)
                if url.startswith("https://"):
                    raise urllib.error.URLError("ssl eof")
                return _FakeResponse(XML_SAMPLE)

        opener = _FakeOpener()
        with patch("scripts.arxiv_search.urllib.request.build_opener", return_value=opener):
            from scripts.arxiv_search import fetch_arxiv_candidates

            payload = fetch_arxiv_candidates(profile, config={})
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(len(opener.calls), 2)
        self.assertTrue(opener.calls[0].startswith("https://"))
        self.assertTrue(opener.calls[1].startswith("http://"))

    def test_fetch_arxiv_candidates_uses_keyword_only_query_when_no_categories_are_available(self) -> None:
        profile = {"arxiv_categories": [], "keywords": ["protein folding"], "recall_strategy": {"arxiv_role": "supplemental"}}

        class _FakeResponse:
            def __init__(self, text: str) -> None:
                self._text = text

            def read(self) -> bytes:
                return self._text.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeOpener:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def open(self, url: str, timeout: int = 60):
                self.calls.append(url)
                if "cat%3A" in url:
                    raise AssertionError("no arxiv category should be forced for non-CS profiles")
                return _FakeResponse(XML_SAMPLE)

        opener = _FakeOpener()
        with patch("scripts.arxiv_search.urllib.request.build_opener", return_value=opener):
            from scripts.arxiv_search import fetch_arxiv_candidates

            payload = fetch_arxiv_candidates(profile, config={})
        self.assertEqual(payload["candidate_count"], 1)
        self.assertEqual(payload["recall_role"], "supplemental")
        self.assertTrue(all("cat%3A" not in call for call in opener.calls))

    def test_fetch_arxiv_candidates_retries_with_broader_query_when_first_query_too_narrow(self) -> None:
        profile = {
            "arxiv_categories": ["cs.AI", "cs.CL"],
            "topics": ["LLM推理优化", "学术知识图谱"],
            "keywords": ["academic graph mining", "name disambiguation", "LLM reasoning"],
        }

        class _FakeResponse:
            def __init__(self, text: str) -> None:
                self._text = text

            def read(self) -> bytes:
                return self._text.encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class _FakeOpener:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def open(self, url: str, timeout: int = 60):
                self.calls.append(url)
                if url.startswith("http://"):
                    raise AssertionError("fallback to http should not be needed in this test")
                if "all%3A%22academic+graph+mining%22" in url and "all%3A%22llm+agent%22" not in url:
                    return _FakeResponse(XML_EMPTY)
                return _FakeResponse(XML_SAMPLE)

        opener = _FakeOpener()
        with patch("scripts.arxiv_search.urllib.request.build_opener", return_value=opener), patch(
            "scripts.arxiv_search._build_query_term_plans",
            return_value=[["academic graph mining", "name disambiguation"], ["LLM Agent", "LLM reasoning"]],
        ):
            from scripts.arxiv_search import fetch_arxiv_candidates

            payload = fetch_arxiv_candidates(profile, config={"search": {"top_k": 5}})
        self.assertEqual(payload["candidate_count"], 1)
        self.assertGreaterEqual(len(payload["queries_tried"]), 2)
        self.assertIn('all:"llm agent"', payload["query"].lower())

    def test_rank_arxiv_candidates_uses_profile_signals(self) -> None:
        payload = {"papers": parse_arxiv_xml(XML_SAMPLE)}
        profile = {
            "keywords": ["multimodal", "multi-agent"],
            "arxiv_categories": ["cs.AI", "cs.CV"],
            "preferred_authors": ["Alice"],
            "preferred_venues": [],
            "seed_papers": [{"title": "Multimodal Planning", "keywords": ["planning"], "abstract": ""}],
            "source_metadata": {"source": "aminer_usr_bind_profile"},
        }
        ranked = rank_arxiv_candidates(payload, profile, top_k=5)
        self.assertEqual(ranked["paper_count"], 1)
        paper = ranked["papers"][0]
        self.assertIn("multimodal", paper["matched_keywords"])
        self.assertIn("Alice", paper["matched_authors"])
        self.assertGreater(paper["recommendation_score"], 0)
        self.assertTrue(paper["aminer_paper_url"].startswith("https://www.aminer.cn/search?"))

    def test_rank_arxiv_candidates_preserves_existing_aminer_metadata(self) -> None:
        payload = {
            "papers": [
                {
                    **parse_arxiv_xml(XML_SAMPLE)[0],
                    "author_entries": [{"display_name": "Alice", "profile_url": "https://www.aminer.cn/profile/alice", "is_disambiguated": True}],
                    "aminer_author_profiles": [{"display_name": "Alice", "profile_url": "https://www.aminer.cn/profile/alice"}],
                    "aminer_paper_url": "https://www.aminer.cn/pub/demo-paper",
                    "famous_authors": ["Alice"],
                }
            ]
        }
        profile = {
            "keywords": ["multimodal"],
            "arxiv_categories": ["cs.AI"],
            "preferred_authors": [],
            "preferred_venues": [],
            "seed_papers": [],
            "source_metadata": {"source": "topics_fallback"},
        }
        ranked = rank_arxiv_candidates(payload, profile, top_k=5)
        paper = ranked["papers"][0]
        self.assertEqual(paper["author_entries"][0]["profile_url"], "https://www.aminer.cn/profile/alice")
        self.assertEqual(paper["aminer_paper_url"], "https://www.aminer.cn/pub/demo-paper")
        self.assertEqual(paper["famous_authors"], ["Alice"])

    def test_rank_arxiv_candidates_prefers_effective_abstracts(self) -> None:
        payload = {
            "papers": [
                {
                    "title": "Rural Education Policy Study",
                    "abstract": "",
                    "authors": ["Author A"],
                    "categories": [],
                    "published_date": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
                {
                    "title": "Rural Education Policy Study With Evidence",
                    "abstract": "This paper studies rural education policy, teacher allocation, student outcomes, and education equity through a multi-region empirical analysis with detailed experiments and evaluations.",
                    "authors": ["Author B"],
                    "categories": [],
                    "published_date": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
            ]
        }
        profile = {
            "keywords": ["rural education"],
            "arxiv_categories": [],
            "preferred_authors": [],
            "preferred_venues": [],
            "seed_papers": [],
            "source_metadata": {"source": "topics_fallback"},
        }
        ranked = rank_arxiv_candidates(payload, profile, top_k=2)
        self.assertEqual(ranked["papers"][0]["title"], "Rural Education Policy Study With Evidence")
        self.assertTrue(ranked["papers"][0]["has_effective_abstract"])
        self.assertFalse(ranked["papers"][1]["has_effective_abstract"])

    def test_rank_arxiv_candidates_prefers_high_signal_weighted_terms(self) -> None:
        payload = {
            "papers": [
                {
                    "title": "Generic Graph Benchmark for Intrusion Detection",
                    "abstract": "This paper studies benchmark design with extensive experiments for graph learning and intrusion detection.",
                    "authors": ["Author A"],
                    "categories": [],
                    "published_date": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
                {
                    "title": "GPT-NER: Named Entity Recognition Via Large Language Models",
                    "abstract": "We study named entity recognition for academic entities with strong empirical results.",
                    "authors": ["Author B"],
                    "categories": [],
                    "published_date": datetime(2025, 11, 1, tzinfo=timezone.utc),
                },
            ]
        }
        profile = {
            "keywords": ["Benchmark", "Named Entity Recognition", "Academic Knowledge Graph"],
            "retrieval_keywords": ["Named Entity Recognition", "Academic Knowledge Graph"],
            "retrieval_term_weights": {
                "Named Entity Recognition": 5.0,
                "Academic Knowledge Graph": 4.2,
            },
            "arxiv_categories": [],
            "preferred_authors": [],
            "preferred_venues": [],
            "seed_papers": [],
            "source_metadata": {"source": "merged_research_profile"},
        }
        ranked = rank_arxiv_candidates(payload, profile, top_k=2)
        self.assertEqual(ranked["papers"][0]["title"], "GPT-NER: Named Entity Recognition Via Large Language Models")

    def test_rank_arxiv_candidates_uses_secondary_terms_as_weak_signal(self) -> None:
        payload = {
            "papers": [
                {
                    "title": "Benchmark-driven Systems Paper",
                    "abstract": "We report extensive experiments for a systems benchmark.",
                    "authors": ["Author A"],
                    "categories": [],
                    "published_date": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
                {
                    "title": "Entity Linking for Academic Graphs",
                    "abstract": "We study entity linking and author name disambiguation for academic knowledge graphs.",
                    "authors": ["Author B"],
                    "categories": [],
                    "published_date": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
            ]
        }
        profile = {
            "retrieval_keywords": ["Entity linking", "Academic Knowledge Graph"],
            "ranking_keywords": ["Entity linking", "Academic Knowledge Graph", "Benchmark", "extensive experiments"],
            "retrieval_term_weights": {
                "Entity linking": 5.0,
                "Academic Knowledge Graph": 4.5,
            },
            "arxiv_categories": [],
            "preferred_authors": [],
            "preferred_venues": [],
            "seed_papers": [],
            "source_metadata": {"source": "merged_research_profile"},
        }
        ranked = rank_arxiv_candidates(payload, profile, top_k=2)
        self.assertEqual(ranked["papers"][0]["title"], "Entity Linking for Academic Graphs")
        self.assertGreater(ranked["papers"][0]["primary_match_count"], ranked["papers"][1]["primary_match_count"])

    def test_rank_arxiv_candidates_downweights_auxiliary_and_meta_terms(self) -> None:
        payload = {
            "papers": [
                {
                    "title": "Academic Graph Mining for Entity Linking",
                    "abstract": "We study academic knowledge graph mining and author name disambiguation.",
                    "authors": ["Author A"],
                    "categories": [],
                    "published_date": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
                {
                    "title": "Benchmark-driven NER Systems",
                    "abstract": "We conduct extensive experiments on named entity recognition benchmarks.",
                    "authors": ["Author B"],
                    "categories": [],
                    "published_date": datetime(2026, 3, 1, tzinfo=timezone.utc),
                },
            ]
        }
        profile = {
            "retrieval_keywords": ["Academic Graph Mining", "Entity linking", "Named Entity Recognition"],
            "ranking_keywords": ["Academic Graph Mining", "Entity linking", "Named Entity Recognition", "Benchmark", "extensive experiments"],
            "retrieval_term_weights": {
                "Academic Graph Mining": 5.2,
                "Entity linking": 4.8,
                "Named Entity Recognition": 4.0,
                "Benchmark": 0.2,
                "extensive experiments": 0.1,
            },
            "arxiv_categories": [],
            "preferred_authors": [],
            "preferred_venues": [],
            "seed_papers": [],
            "source_metadata": {"source": "merged_research_profile"},
        }
        ranked = rank_arxiv_candidates(payload, profile, top_k=2)
        self.assertEqual(ranked["papers"][0]["title"], "Academic Graph Mining for Entity Linking")
        self.assertGreater(ranked["papers"][0]["recommendation_score"], ranked["papers"][1]["recommendation_score"])
        self.assertIn("academic graph mining", [item.casefold() for item in ranked["papers"][0]["matched_keywords"]])
        self.assertNotIn("Benchmark", ranked["papers"][0]["matched_keywords"])

    def test_recent_top_tier_helpers_promote_recent_top_venue_papers(self) -> None:
        self.assertEqual(identify_top_tier_venue("Conference on Computer Vision and Pattern Recognition (CVPR) 2025"), "CVPR")
        self.assertEqual(default_recent_top_tier_quota(5), 2)
        papers = [
            {"title": "Paper 1", "venue": "arXiv", "year": "2026"},
            {"title": "Paper 2", "venue": "Workshop on Agents", "year": "2026"},
            {"title": "Paper 3", "venue": "NeurIPS 2025", "year": "2025"},
            {"title": "Paper 4", "venue": "CVPR 2026", "year": "2026"},
            {"title": "Paper 5", "venue": "Preprint", "year": "2026"},
        ]
        annotated = annotate_recent_top_tier_metadata(papers[2])
        self.assertTrue(annotated["is_recent_top_tier"])
        self.assertEqual(annotated["top_tier_venue"], "NeurIPS")
        ranked_candidates, selected, policy = rebalance_recent_top_tier_papers(papers, top_k=3, min_recent_top_tier=2)
        self.assertEqual(len(ranked_candidates), 5)
        self.assertEqual(len(selected), 3)
        self.assertEqual(sum(1 for paper in selected if paper["is_recent_top_tier"]), 2)
        self.assertEqual(policy["promoted_count"], 1)
        self.assertIn("Paper 4", [paper["title"] for paper in selected])

    def test_build_aminer_paper_url_for_arxiv_paper_uses_title_search(self) -> None:
        paper = parse_arxiv_xml(XML_SAMPLE)[0]
        url = build_aminer_paper_url_for_arxiv_paper(paper)
        self.assertIn("https://www.aminer.cn/search?", url)
        self.assertIn("Agentic%20Multimodal%20Planning", url)

    def test_build_aminer_paper_url_for_arxiv_paper_prefers_exact_pub_page(self) -> None:
        paper = parse_arxiv_xml(XML_SAMPLE)[0]
        paper["aminer_paper_id"] = "69bc9f479be8eb7c4b4c72e5"
        url = build_aminer_paper_url_for_arxiv_paper(paper)
        self.assertEqual(
            url,
            "https://www.aminer.cn/pub/69bc9f479be8eb7c4b4c72e5",
        )

    def test_enrich_ranked_payload_with_aminer_paper_urls_uses_mapping_when_available(self) -> None:
        ranked = {
            "papers": [
                {
                    "arxiv_id": "2501.00001v1",
                    "title": "Agentic Multimodal Planning",
                    "aminer_paper_url": "",
                }
            ]
        }
        with patch(
            "scripts.arxiv_search.map_arxiv_ids_to_aminer_ids",
            return_value={"2501.00001": "69bc9f479be8eb7c4b4c72e5"},
        ):
            enriched = enrich_ranked_payload_with_aminer_paper_urls(ranked, config={"aminer": {"token": "demo"}})
        paper = enriched["papers"][0]
        self.assertEqual(paper["aminer_paper_id"], "69bc9f479be8eb7c4b4c72e5")
        self.assertEqual(
            paper["aminer_paper_url"],
            "https://www.aminer.cn/pub/69bc9f479be8eb7c4b4c72e5",
        )

if __name__ == "__main__":
    unittest.main()
