from __future__ import annotations

import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.internal_profile_provider import _score_terms


class InternalProfileProviderTests(unittest.TestCase):
    def test_score_terms_prefers_narrow_supported_phrases_and_filters_broad_keywords(self) -> None:
        authored_papers = [
            {
                "title": "OAG-Bench: A Human-Curated Benchmark for Academic Graph Mining",
                "abstract": "Academic graph mining benchmark for scholar profiling and paper source tracing.",
                "fields": ["计算机科学技术", "信息检索"],
                "topics": ["自然语言处理"],
                "keywords": ["academic graph mining", "academic knowledge graph", "Computer science", "OAG"],
                "venue": "KDD 2024",
                "year": 2024,
                "n_citation": 40,
                "coauthor_names": ["Jie Tang"],
            },
            {
                "title": "Small Language Model Makes an Effective Long Text Extractor",
                "abstract": "Named Entity Recognition for long text extraction.",
                "fields": ["自然语言处理", "机器学习"],
                "topics": ["Named Entity Recognition"],
                "keywords": ["named entity recognition", "long text extraction", "machine learning"],
                "venue": "AAAI 2025",
                "year": 2025,
                "n_citation": 3,
                "coauthor_names": ["Jie Tang"],
            },
            {
                "title": "RPC-Bench: A Fine-grained Benchmark for Research Paper Comprehension",
                "abstract": "Benchmark for research paper comprehension and evaluation.",
                "fields": ["计算机科学技术"],
                "topics": ["信息科学与系统科学"],
                "keywords": ["research paper comprehension", "benchmark", "evaluation", "computer science and technology"],
                "venue": "CoRR",
                "year": 2026,
                "n_citation": 0,
                "coauthor_names": ["Jie Tang"],
            },
        ]

        topics, keywords, preferred_authors, preferred_venues, core_topics = _score_terms(
            authored_papers,
            experts_topics=["Deep Research"],
        )

        self.assertIn("academic graph mining", [item.lower() for item in topics[:4]])
        self.assertIn("named entity recognition", [item.lower() for item in topics[:5]])
        self.assertTrue(
            {
                "research paper comprehension",
                "long text extraction",
            }.intersection({item.lower() for item in topics[:6]})
        )
        self.assertNotIn("计算机科学技术", topics[:4])
        self.assertNotIn("machine learning", [item.lower() for item in keywords[:10]])
        self.assertNotIn("computer science", [item.lower() for item in keywords[:10]])
        self.assertFalse(any("bench" in item.lower() for item in keywords[:10]))
        self.assertIn("academic graph mining", [item.lower() for item in keywords[:8]])
        self.assertIn("named entity recognition", [item.lower() for item in keywords])
        self.assertIn("academic graph mining", [str(item["name"]).lower() for item in core_topics[:4]])
        self.assertIn("Jie Tang", preferred_authors)
        self.assertIn("KDD 2024", preferred_venues)


if __name__ == "__main__":
    unittest.main()
