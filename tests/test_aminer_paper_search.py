from __future__ import annotations

import sys
import unittest
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.aminer_paper_search import _merge_paper_detail


class AminderPaperSearchTests(unittest.TestCase):
    def test_merge_paper_detail_prefers_detail_abstract_and_keywords(self) -> None:
        existing = {
            "title": "Connotation, Value and Path of Rural Education Revitalization",
            "abstract": "",
            "summary": "",
            "keywords": ["乡村教育"],
            "authors": ["Wanxue Qi"],
            "author_entries": [{"display_name": "Wanxue Qi", "profile_url": "", "is_disambiguated": False}],
            "aminer_author_profiles": [],
            "source_metadata": {"raw_id": "paper-a"},
        }
        detail = {
            "id": "paper-a",
            "title": "Connotation, Value and Path of Rural Education Revitalization",
            "title_zh": "乡村教育振兴的内涵、价值与路径",
            "abstract_zh": "乡村教育是中国教育的短板，也是推进教育现代化的重要议题。",
            "keywords_zh": ["乡村教育振兴", "教育现代化"],
            "year": 2020,
            "authors": [
                {"id": "author-1", "name": "Wanxue Qi", "name_zh": "戚万学", "org_zh": "曲阜师范大学"},
            ],
        }
        merged = _merge_paper_detail(existing, detail)
        self.assertEqual(merged["abstract"], "乡村教育是中国教育的短板，也是推进教育现代化的重要议题。")
        self.assertEqual(merged["summary"], "乡村教育是中国教育的短板，也是推进教育现代化的重要议题。")
        self.assertEqual(merged["keywords"], ["乡村教育振兴", "教育现代化"])
        self.assertEqual(merged["authors"], ["戚万学"])
        self.assertEqual(merged["author_entries"][0]["profile_url"], "https://www.aminer.cn/profile/author-1")
        self.assertTrue(merged["source_metadata"]["detail_enriched"])


if __name__ == "__main__":
    unittest.main()
