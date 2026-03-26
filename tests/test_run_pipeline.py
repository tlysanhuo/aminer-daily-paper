from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.run_pipeline import (
    _aminer_candidate_pre_detail_score,
    _fetch_aminer_candidates,
    _prioritize_aminer_candidates_before_detail,
    _select_aminer_queries,
    _summary_subprocess_timeout_seconds,
    run_pipeline,
)


def _fake_run_python_factory(ranked_payload: dict | None = None) -> callable:
    def _fake_run_python(script_path: Path, args: list[str], **kwargs) -> None:
        script_name = Path(script_path).name
        if script_name == "summarize_papers.py":
            output_path = Path(args[args.index("--output") + 1])
            payload = {
                "status": "success",
                "generated_at": "",
                "paper_count": len(list((ranked_payload or {}).get("papers") or [])),
                "papers": list((ranked_payload or {}).get("papers") or []),
                "profile_topics": list((ranked_payload or {}).get("profile_topics") or []),
                "profile_name": str((ranked_payload or {}).get("profile_name") or ""),
                "profile_source": str((ranked_payload or {}).get("profile_source") or ""),
            }
            output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return
        if script_name == "render_feishu_messages.py":
            output_path = Path(args[args.index("--output") + 1])
            payload = {"status": "success", "paper_count": 0, "messages": [], "final_response": "NO_REPLY"}
            output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return
        if script_name == "dispatch_feishu_messages.py":
            return
        raise AssertionError(f"unexpected script: {script_name}")

    return _fake_run_python


class RunPipelineTests(unittest.TestCase):
    def test_select_aminer_queries_uses_llm_topic_expansion_for_non_cs(self) -> None:
        profile = {
            "is_cs_user": False,
            "topics": ["乡村教育政策与发展"],
            "keywords": ["乡村教育", "教育公平"],
            "source_metadata": {
                "components": [
                    {
                        "llm_topics": [
                            {
                                "name": "乡村教育与乡村振兴",
                                "keywords": ["乡村振兴", "乡村人才", "教育扶贫"],
                            }
                        ]
                    }
                ]
            },
        }
        queries = _select_aminer_queries(profile, max_queries=8)
        self.assertIn({"title": "乡村教育与乡村振兴", "keyword": ""}, queries)
        self.assertIn({"title": "乡村教育与乡村振兴", "keyword": "乡村振兴 乡村人才"}, queries)
        self.assertIn({"title": "", "keyword": "乡村振兴 乡村人才 教育扶贫"}, queries)

    def test_fetch_aminer_candidates_enriches_non_cs_results_with_details(self) -> None:
        profile = {
            "is_cs_user": False,
            "topics": ["乡村教育"],
            "keywords": ["教育公平"],
            "recall_primary_source": "aminer",
            "source_metadata": {
                "components": [
                    {
                        "llm_topics": [
                            {
                                "name": "乡村教育与乡村振兴",
                                "keywords": ["乡村振兴", "乡村人才", "教育扶贫"],
                            }
                        ]
                    }
                ]
            },
        }
        search_payload = {
            "status": "success",
            "query": {"title": "乡村教育", "keyword": ""},
            "papers": [
                {
                    "title": "Paper A",
                    "abstract": "",
                    "summary": "",
                    "authors": ["Author A"],
                    "author_entries": [{"display_name": "Author A", "profile_url": "", "is_disambiguated": False}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "keywords": [],
                    "aminer_paper_id": "paper-a",
                    "source_metadata": {"raw_id": "paper-a"},
                }
            ],
        }
        enriched_papers = [{**search_payload["papers"][0], "abstract": "这是有效摘要。", "summary": "这是有效摘要。"}]
        with patch("scripts.run_pipeline.search_papers_pro", return_value=search_payload), patch(
            "scripts.run_pipeline.enrich_papers_with_details",
            return_value=enriched_papers,
        ) as enrich_mock:
            payload = _fetch_aminer_candidates(profile, config={"aminer": {"token": "demo"}})
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["papers"][0]["abstract"], "这是有效摘要。")
        enrich_mock.assert_called_once()

    def test_fetch_aminer_candidates_skips_detail_enrichment_for_cs(self) -> None:
        profile = {
            "is_cs_user": True,
            "topics": ["multimodal"],
            "keywords": ["agent"],
            "recall_primary_source": "aminer",
            "source_metadata": {},
        }
        search_payload = {
            "status": "success",
            "query": {"title": "multimodal", "keyword": ""},
            "papers": [
                {
                    "title": "Paper A",
                    "abstract": "",
                    "summary": "",
                    "authors": ["Author A"],
                    "author_entries": [{"display_name": "Author A", "profile_url": "", "is_disambiguated": False}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "keywords": [],
                    "aminer_paper_id": "paper-a",
                    "source_metadata": {"raw_id": "paper-a"},
                }
            ],
        }
        with patch("scripts.run_pipeline.search_papers_pro", return_value=search_payload), patch(
            "scripts.run_pipeline.enrich_papers_with_details",
        ) as enrich_mock:
            payload = _fetch_aminer_candidates(profile, config={"aminer": {"token": "demo"}})
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["papers"][0]["abstract"], "")
        enrich_mock.assert_not_called()

    def test_prioritize_aminer_candidates_before_detail_prefers_recent_and_cited(self) -> None:
        papers = [
            {"title": "Old Low", "year": 2018, "n_citation_bucket": "1-10", "venue": "Venue A"},
            {"title": "Recent Mid", "year": 2024, "n_citation_bucket": "11-50", "venue": "Venue B"},
            {"title": "Recent High", "year": 2023, "n_citation_bucket": "200-1000", "venue": "Venue C"},
        ]
        ranked = _prioritize_aminer_candidates_before_detail(papers)
        self.assertEqual(ranked[0]["title"], "Recent High")
        self.assertEqual(ranked[-1]["title"], "Old Low")
        self.assertGreater(
            _aminer_candidate_pre_detail_score({"title": "A", "year": 2024, "n_citation_bucket": "11-50"}),
            _aminer_candidate_pre_detail_score({"title": "B", "year": 2017, "n_citation_bucket": "1-10"}),
        )

    def test_run_pipeline_raises_stage_error_when_summary_is_not_success(self) -> None:
        profile = {
            "status": "success",
            "topics": ["multimodal"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "topics_fallback"},
            "is_cs_user": True,
            "recall_primary_source": "arxiv",
            "recall_secondary_source": "aminer",
            "recall_strategy": {"is_cs_user": True},
        }
        candidate_payload = {"status": "success", "papers": []}
        ranked_payload = {"status": "success", "papers": [{"title": "Paper A", "summary": "x", "authors": ["A"], "author_entries": [{"display_name": "A"}], "aminer_author_profiles": [], "famous_authors": [], "keywords": [], "aminer_paper_url": "https://www.aminer.cn/pub/demo"}]}

        def _fake_run_python_failure(script_path: Path, args: list[str], **kwargs) -> None:
            script_name = Path(script_path).name
            if script_name == "summarize_papers.py":
                output_path = Path(args[args.index("--output") + 1])
                payload = {
                    "status": "degraded",
                    "papers": [{"title": "Paper A", "summary_status": "degraded", "summary_reason": "llm_timeout"}],
                }
                output_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                return
            raise AssertionError(f"unexpected script: {script_name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "topic_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ), patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch("scripts.run_pipeline._run_python", side_effect=_fake_run_python_failure):
                with self.assertRaisesRegex(RuntimeError, "summary_failed:llm_timeout"):
                    run_pipeline(
                        base_dir=base_dir,
                        output_dir=base_dir / "outputs",
                        config={"search": {"top_k": 1}, "llm": {"api_key": "", "base_url": "", "model": ""}},
                        aminer_user_id="",
                        topics=["multimodal"],
                        scholar_name="",
                        scholar_org="",
                        paper_titles=[],
                        papers_file="",
                        free_text="",
                        target="",
                        account_id="main",
                        skip_dispatch=True,
                    )

    def test_run_pipeline_allows_partial_success_summary_to_render_and_dispatch(self) -> None:
        profile = {
            "status": "success",
            "topics": ["乡村教育"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "topics_fallback"},
            "is_cs_user": False,
            "recall_primary_source": "aminer",
            "recall_secondary_source": "arxiv",
            "recall_strategy": {"is_cs_user": False},
        }
        candidate_payload = {"status": "success", "papers": []}
        ranked_payload = {
            "status": "success",
            "papers": [
                {
                    "title": "Paper A",
                    "summary": "x",
                    "authors": ["A"],
                    "author_entries": [{"display_name": "A"}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "keywords": ["乡村教育"],
                    "aminer_paper_url": "https://www.aminer.cn/pub/demo-a",
                },
                {
                    "title": "Paper B",
                    "summary": "y",
                    "authors": ["B"],
                    "author_entries": [{"display_name": "B"}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "keywords": ["乡村教师"],
                    "aminer_paper_url": "https://www.aminer.cn/pub/demo-b",
                },
            ],
        }
        call_log: list[str] = []

        def _fake_run_python_partial_success(script_path: Path, args: list[str], **kwargs) -> None:
            script_name = Path(script_path).name
            call_log.append(script_name)
            if script_name == "summarize_papers.py":
                output_path = Path(args[args.index("--output") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "status": "partial_success",
                            "generated_at": "",
                            "paper_count": 2,
                            "profile_topics": ["乡村教育"],
                            "profile_name": "demo-profile",
                            "profile_source": "topics_fallback",
                            "papers": [
                                {
                                    **ranked_payload["papers"][0],
                                    "summary_status": "success",
                                    "summary_reason": "",
                                    "summary_provider": "primary",
                                    "structured_summary": {
                                        "research_problem": "问题A",
                                        "research_challenge": "挑战A",
                                        "research_method": "方法A",
                                    },
                                    "summary": "研究问题：问题A\n研究挑战：挑战A\n研究方法：方法A",
                                },
                                {
                                    **ranked_payload["papers"][1],
                                    "summary_status": "degraded",
                                    "summary_reason": "llm_client_error:NotFoundError",
                                    "summary_provider": "fallback",
                                    "structured_summary": {
                                        "research_problem": "问题B",
                                        "research_challenge": "挑战B",
                                        "research_method": "方法B",
                                    },
                                    "summary": "研究问题：问题B\n研究挑战：挑战B\n研究方法：方法B",
                                },
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                return
            if script_name == "render_feishu_messages.py":
                output_path = Path(args[args.index("--output") + 1])
                output_path.write_text(
                    json.dumps({"status": "partial_success", "paper_count": 2, "messages": [{"index": 1}, {"index": 2}], "final_response": "NO_REPLY"}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return
            if script_name == "dispatch_feishu_messages.py":
                return
            raise AssertionError(f"unexpected script: {script_name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "topic_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ), patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch("scripts.run_pipeline._run_python_with_timeout", side_effect=_fake_run_python_partial_success), patch(
                "scripts.run_pipeline._run_python",
                side_effect=_fake_run_python_partial_success,
            ):
                result = run_pipeline(
                    base_dir=base_dir,
                    output_dir=base_dir / "outputs",
                    config={"search": {"top_k": 2}, "llm": {"api_key": "", "base_url": "", "model": ""}},
                    aminer_user_id="",
                    topics=["乡村教育"],
                    scholar_name="",
                    scholar_org="",
                    paper_titles=[],
                    papers_file="",
                    free_text="",
                    target="user:test",
                    account_id="default",
                    skip_dispatch=False,
                )
        self.assertEqual(result["status"], "success")
        self.assertIn("render_feishu_messages.py", call_log)
        self.assertIn("dispatch_feishu_messages.py", call_log)

    def test_run_pipeline_uses_local_summary_fallback_after_timeout(self) -> None:
        profile = {
            "status": "success",
            "topics": ["multimodal"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "topics_fallback"},
            "is_cs_user": True,
            "recall_primary_source": "arxiv",
            "recall_secondary_source": "aminer",
            "recall_strategy": {"is_cs_user": True},
        }
        candidate_payload = {"status": "success", "papers": []}
        ranked_payload = {
            "status": "success",
            "papers": [
                {
                    "title": "Paper A",
                    "summary": "x",
                    "authors": ["A"],
                    "author_entries": [{"display_name": "A"}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "keywords": [],
                    "aminer_paper_url": "https://www.aminer.cn/pub/demo",
                }
            ],
        }
        call_log: list[tuple[str, list[str]]] = []

        def _fake_run_python_timeout_then_success(script_path: Path, args: list[str], **kwargs) -> None:
            call_log.append((Path(script_path).name, list(args)))
            script_name = Path(script_path).name
            if script_name == "summarize_papers.py":
                raise RuntimeError("summarize_papers.py_timeout:120s")
            if script_name == "render_feishu_messages.py":
                output_path = Path(args[args.index("--output") + 1])
                output_path.write_text(json.dumps({"status": "success", "paper_count": 0, "messages": [], "final_response": "NO_REPLY"}, ensure_ascii=False), encoding="utf-8")
                return
            if script_name == "dispatch_feishu_messages.py":
                return
            raise AssertionError(f"unexpected script: {script_name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "topic_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ), patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch(
                "scripts.run_pipeline.summarize_papers_locally",
                return_value={
                    "status": "degraded",
                    "papers": [
                        {
                            **ranked_payload["papers"][0],
                            "summary_status": "degraded",
                            "summary_reason": "missing_api_key",
                            "structured_summary": {
                                "research_problem": "问题",
                                "research_challenge": "挑战",
                                "research_method": "方法",
                            },
                            "summary": "问题 挑战 方法",
                            "summary_provider": "",
                        }
                    ],
                    "profile_topics": [],
                    "profile_name": "",
                    "profile_source": "",
                },
            ), patch("scripts.run_pipeline._run_python_with_timeout", side_effect=_fake_run_python_timeout_then_success), patch(
                "scripts.run_pipeline._run_python",
                side_effect=_fake_run_python_timeout_then_success,
            ):
                result = run_pipeline(
                    base_dir=base_dir,
                    output_dir=base_dir / "outputs",
                    config={"search": {"top_k": 1}, "llm": {"api_key": "", "base_url": "", "model": "", "timeout_seconds": 30, "max_concurrent_requests": 10}},
                    aminer_user_id="",
                    topics=["multimodal"],
                    scholar_name="",
                    scholar_org="",
                    paper_titles=[],
                    papers_file="",
                    free_text="",
                    target="",
                    account_id="main",
                    skip_dispatch=True,
                )
        self.assertEqual(result["status"], "success")
        summarize_calls = [args for name, args in call_log if name == "summarize_papers.py"]
        self.assertEqual(len(summarize_calls), 1)
        self.assertEqual(summarize_calls[0][summarize_calls[0].index("--max-concurrent-requests") + 1], "10")

    def test_run_pipeline_preserves_partial_summaries_after_timeout(self) -> None:
        profile = {
            "status": "success",
            "topics": ["embodied-ai"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "topics_fallback"},
            "is_cs_user": True,
            "recall_primary_source": "arxiv",
            "recall_secondary_source": "aminer",
            "recall_strategy": {"is_cs_user": True},
        }
        candidate_payload = {"status": "success", "papers": []}
        ranked_payload = {
            "status": "success",
            "papers": [
                {
                    "arxiv_id": "paper-a",
                    "title": "Paper A",
                    "summary": "x",
                    "authors": ["A"],
                    "author_entries": [{"display_name": "A"}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "keywords": [],
                    "aminer_paper_url": "https://www.aminer.cn/pub/demo-a",
                },
                {
                    "arxiv_id": "paper-b",
                    "title": "Paper B",
                    "summary": "y",
                    "authors": ["B"],
                    "author_entries": [{"display_name": "B"}],
                    "aminer_author_profiles": [],
                    "famous_authors": [],
                    "keywords": [],
                    "aminer_paper_url": "https://www.aminer.cn/pub/demo-b",
                },
            ],
        }

        def _fake_run_python_timeout_with_partial(script_path: Path, args: list[str], **kwargs) -> None:
            script_name = Path(script_path).name
            if script_name == "summarize_papers.py":
                output_path = Path(args[args.index("--output") + 1])
                output_path.write_text(
                    json.dumps(
                        {
                            "status": "success",
                            "generated_at": "",
                            "paper_count": 1,
                            "papers": [
                                {
                                    **ranked_payload["papers"][0],
                                    "summary_status": "success",
                                    "summary_reason": "",
                                    "summary_provider": "primary",
                                    "structured_summary": {
                                        "research_problem": "问题A",
                                        "research_challenge": "挑战A",
                                        "research_method": "方法A",
                                    },
                                    "summary": "问题A 挑战A 方法A",
                                }
                            ],
                            "profile_topics": ["embodied-ai"],
                            "profile_name": "demo-profile",
                            "profile_source": "scholar_path",
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                raise RuntimeError("summarize_papers.py_timeout:120s")
            if script_name == "render_feishu_messages.py":
                output_path = Path(args[args.index("--output") + 1])
                output_path.write_text(
                    json.dumps({"status": "success", "paper_count": 0, "messages": [], "final_response": "NO_REPLY"}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return
            if script_name == "dispatch_feishu_messages.py":
                return
            raise AssertionError(f"unexpected script: {script_name}")

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "topic_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ), patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch(
                "scripts.run_pipeline.summarize_papers_locally",
                return_value={
                    "status": "degraded",
                    "papers": [
                        {
                            **ranked_payload["papers"][0],
                            "summary_status": "degraded",
                            "summary_reason": "missing_api_key",
                            "structured_summary": {
                                "research_problem": "问题A",
                                "research_challenge": "挑战A",
                                "research_method": "方法A",
                            },
                            "summary": "问题A 挑战A 方法A",
                            "summary_provider": "",
                        },
                        {
                            **ranked_payload["papers"][1],
                            "summary_status": "degraded",
                            "summary_reason": "missing_api_key",
                            "structured_summary": {
                                "research_problem": "问题B",
                                "research_challenge": "挑战B",
                                "research_method": "方法B",
                            },
                            "summary": "问题B 挑战B 方法B",
                            "summary_provider": "",
                        },
                    ],
                    "profile_topics": ["embodied-ai"],
                    "profile_name": "demo-profile",
                    "profile_source": "scholar_path",
                },
            ), patch("scripts.run_pipeline._run_python_with_timeout", side_effect=_fake_run_python_timeout_with_partial), patch(
                "scripts.run_pipeline._run_python",
                side_effect=_fake_run_python_timeout_with_partial,
            ):
                run_pipeline(
                    base_dir=base_dir,
                    output_dir=base_dir / "outputs",
                    config={"search": {"top_k": 2}, "llm": {"api_key": "", "base_url": "", "model": "", "timeout_seconds": 45, "max_concurrent_requests": 10}},
                    aminer_user_id="",
                    topics=["embodied-ai"],
                    scholar_name="",
                    scholar_org="",
                    paper_titles=[],
                    papers_file="",
                    free_text="",
                    target="",
                    account_id="main",
                    skip_dispatch=True,
                )
            summarized_payload = json.loads((base_dir / "outputs" / "papers_summarized.json").read_text(encoding="utf-8"))

        self.assertEqual(summarized_payload["status"], "success")
        self.assertEqual(len(summarized_payload["papers"]), 2)
        self.assertEqual(summarized_payload["papers"][0]["summary_provider"], "primary")
        self.assertEqual(summarized_payload["papers"][1]["summary_provider"], "local_fallback")

    def test_summary_subprocess_timeout_seconds_is_not_hard_capped_to_sixty(self) -> None:
        timeout_seconds = _summary_subprocess_timeout_seconds(
            llm_timeout_seconds=45,
            fallback_timeout_seconds=45,
            paper_count=5,
            max_concurrent_requests=10,
            retry_attempts=0,
        )
        self.assertGreater(timeout_seconds, 60)

    def test_run_pipeline_defaults_to_recent_top_tier_mix(self) -> None:
        profile = {
            "status": "success",
            "topics": ["multimodal"],
            "keywords": ["tool use"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "topics_fallback"},
            "is_cs_user": True,
            "recall_primary_source": "arxiv",
            "recall_secondary_source": "aminer",
            "recall_strategy": {
                "is_cs_user": True,
                "primary_recall_source": "arxiv",
                "secondary_recall_source": "aminer",
                "arxiv_role": "primary",
            },
        }
        candidate_payload = {"status": "success", "papers": []}
        ranked_payload = {
            "status": "success",
            "papers": [
                {"title": "Paper 1", "recommendation_score": 4.5, "venue": "Workshop on Agents", "year": "2026"},
                {"title": "Paper 2", "recommendation_score": 4.4, "venue": "arXiv", "year": "2026"},
                {"title": "Paper 3", "recommendation_score": 4.3, "venue": "NeurIPS 2025", "year": "2025"},
            ],
            "ranked_candidates": [
                {"title": "Paper 1", "recommendation_score": 4.5, "venue": "Workshop on Agents", "year": "2026"},
                {"title": "Paper 2", "recommendation_score": 4.4, "venue": "arXiv", "year": "2026"},
                {"title": "Paper 3", "recommendation_score": 4.3, "venue": "NeurIPS 2025", "year": "2025"},
                {"title": "Paper 4", "recommendation_score": 4.2, "venue": "CVPR 2026", "year": "2026"},
                {"title": "Paper 5", "recommendation_score": 4.1, "venue": "Preprint", "year": "2026"},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "topic_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ), patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch("scripts.run_pipeline._run_python", side_effect=_fake_run_python_factory(ranked_payload)):
                run_pipeline(
                    base_dir=base_dir,
                    output_dir=base_dir / "outputs",
                    config={"search": {"top_k": 3}, "llm": {"api_key": "", "base_url": "", "model": ""}},
                    aminer_user_id="",
                    topics=["multimodal"],
                    scholar_name="",
                    scholar_org="",
                    paper_titles=[],
                    papers_file="",
                    free_text="",
                    target="",
                    account_id="main",
                    skip_dispatch=True,
                )
                ranked_path = base_dir / "outputs" / "papers_ranked.json"
                ranked_data = __import__("json").loads(ranked_path.read_text(encoding="utf-8"))

        self.assertEqual(sum(1 for paper in ranked_data["papers"] if paper["is_recent_top_tier"]), 1)
        self.assertEqual(ranked_data["recent_top_tier_policy"]["selected_recent_top_tier_count"], 1)

    def test_run_pipeline_propagates_recall_strategy_fields(self) -> None:
        profile = {
            "status": "success",
            "topics": ["multimodal"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "authored_papers_bind_profile"},
            "is_cs_user": True,
            "recall_primary_source": "arxiv",
            "recall_secondary_source": "aminer",
            "recall_strategy": {
                "is_cs_user": True,
                "primary_recall_source": "arxiv",
                "secondary_recall_source": "aminer",
                "arxiv_role": "primary",
            },
        }

        candidate_payload = {"status": "success", "papers": []}
        ranked_payload = {"status": "success", "papers": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "scholar_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ) as fetch_mock, patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ) as rank_mock, patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ) as enrich_mock, patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch("scripts.run_pipeline._run_python", side_effect=_fake_run_python_factory(ranked_payload)) as run_python_mock:
                result = run_pipeline(
                    base_dir=base_dir,
                    output_dir=base_dir / "outputs",
                    config={"search": {"top_k": 5}, "llm": {"api_key": "", "base_url": "", "model": ""}},
                    aminer_user_id="demo",
                    topics=["multimodal"],
                    scholar_name="",
                    scholar_org="",
                    paper_titles=[],
                    papers_file="",
                    free_text="",
                    target="",
                    account_id="main",
                    skip_dispatch=True,
                )

        self.assertEqual(result["is_cs_user"], True)
        self.assertEqual(result["recall_primary_source"], "arxiv")
        self.assertEqual(result["recall_secondary_source"], "aminer")
        expected_profile = {**profile, "profile_mode": "scholar_path"}
        fetch_mock.assert_called_once_with(expected_profile, config={"search": {"top_k": 5}, "llm": {"api_key": "", "base_url": "", "model": ""}})
        rank_mock.assert_called_once_with(candidate_payload, expected_profile, top_k=5)
        enrich_mock.assert_called_once()
        self.assertTrue(run_python_mock.called)

    def test_run_pipeline_uses_non_cs_strategy_fields(self) -> None:
        profile = {
            "status": "success",
            "topics": ["gastric cancer"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "topics_fallback"},
            "is_cs_user": False,
            "recall_primary_source": "aminer",
            "recall_secondary_source": "arxiv",
            "recall_strategy": {
                "is_cs_user": False,
                "primary_recall_source": "aminer",
                "secondary_recall_source": "arxiv",
                "arxiv_role": "supplemental",
            },
        }
        candidate_payload = {"status": "success", "papers": [], "recall_plan": ["aminer", "arxiv"], "errors": []}
        ranked_payload = {"status": "success", "papers": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "topic_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ) as fetch_mock, patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch("scripts.run_pipeline._run_python", side_effect=_fake_run_python_factory(ranked_payload)):
                result = run_pipeline(
                    base_dir=base_dir,
                    output_dir=base_dir / "outputs",
                    config={"search": {"top_k": 5}, "aminer": {"token": "demo"}, "llm": {"api_key": "", "base_url": "", "model": ""}},
                    aminer_user_id="demo",
                    topics=["gastric cancer"],
                    scholar_name="",
                    scholar_org="",
                    paper_titles=[],
                    papers_file="",
                    free_text="",
                    target="",
                    account_id="main",
                    skip_dispatch=True,
                )

        self.assertFalse(result["is_cs_user"])
        self.assertEqual(result["recall_primary_source"], "aminer")
        self.assertEqual(result["recall_secondary_source"], "arxiv")
        fetch_mock.assert_called_once()

    def test_run_pipeline_applies_non_cs_llm_rerank_before_summarize(self) -> None:
        profile = {
            "status": "success",
            "topics": ["socioeconomics"],
            "keywords": ["social inequality"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "topics_fallback"},
            "is_cs_user": False,
            "recall_primary_source": "aminer",
            "recall_secondary_source": "arxiv",
            "recall_strategy": {
                "is_cs_user": False,
                "primary_recall_source": "aminer",
                "secondary_recall_source": "arxiv",
                "arxiv_role": "supplemental",
            },
        }
        candidate_payload = {"status": "success", "papers": []}
        ranked_payload = {
            "status": "success",
            "papers": [
                {"title": "Paper A", "recommendation_score": 1.0},
                {"title": "Paper B", "recommendation_score": 2.0},
            ],
            "ranked_candidates": [
                {"title": "Paper A", "recommendation_score": 1.0},
                {"title": "Paper B", "recommendation_score": 2.0},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "topic_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ), patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch(
                "scripts.run_pipeline.llm_rerank_non_cs",
                return_value=(
                    [
                        {"index": 0, "relevance": 95, "quality": 90, "reason": "高度相关"},
                        {"index": 1, "relevance": 20, "quality": 30, "reason": "泛词命中"},
                    ],
                    '{"results":[]}',
                ),
            ), patch("scripts.run_pipeline._run_python", side_effect=_fake_run_python_factory(ranked_payload)):
                result = run_pipeline(
                    base_dir=base_dir,
                    output_dir=base_dir / "outputs",
                    config={"search": {"top_k": 2}, "llm": {"api_key": "demo", "base_url": "https://example.com", "model": "demo-model"}},
                    aminer_user_id="",
                    topics=["socioeconomics"],
                    scholar_name="",
                    scholar_org="",
                    paper_titles=[],
                    papers_file="",
                    free_text="",
                    target="",
                    account_id="main",
                    skip_dispatch=True,
                )
                ranked_path = base_dir / "outputs" / "papers_ranked.json"
                ranked_data = __import__("json").loads(ranked_path.read_text(encoding="utf-8"))

        self.assertEqual(result["is_cs_user"], False)
        self.assertEqual(ranked_data["papers"][0]["title"], "Paper A")
        self.assertEqual(ranked_data["llm_rerank_non_cs"]["status"], "success")
        self.assertEqual(ranked_data["papers"][0]["llm_rerank_relevance"], 95)
        self.assertEqual(ranked_data["llm_rerank"]["top_n"], 2)

    def test_run_pipeline_applies_llm_rerank_for_cs_profiles(self) -> None:
        profile = {
            "status": "success",
            "topics": ["named entity recognition"],
            "keywords": ["academic knowledge graph"],
            "profile_name": "demo-profile",
            "source_metadata": {"source": "authored_papers_bind_profile"},
            "is_cs_user": True,
            "recall_primary_source": "arxiv",
            "recall_secondary_source": "aminer",
            "recall_strategy": {
                "is_cs_user": True,
                "primary_recall_source": "arxiv",
                "secondary_recall_source": "aminer",
                "arxiv_role": "primary",
            },
        }
        candidate_payload = {"status": "success", "papers": []}
        ranked_payload = {
            "status": "success",
            "papers": [
                {"title": "Paper A", "recommendation_score": 1.0},
                {"title": "Paper B", "recommendation_score": 2.0},
            ],
            "ranked_candidates": [
                {"title": "Paper A", "recommendation_score": 1.0},
                {"title": "Paper B", "recommendation_score": 2.0},
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "scripts").mkdir(parents=True)
            with patch("scripts.run_pipeline.build_research_profile", return_value={**profile, "profile_mode": "scholar_path"}), patch(
                "scripts.run_pipeline._fetch_candidates_by_strategy",
                return_value=candidate_payload,
            ), patch(
                "scripts.run_pipeline.rank_arxiv_candidates",
                return_value=ranked_payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_paper_urls",
                side_effect=lambda payload, config=None: payload,
            ), patch(
                "scripts.run_pipeline.enrich_ranked_payload_with_aminer_details",
                side_effect=lambda payload, token="": payload,
            ), patch(
                "scripts.run_pipeline.llm_rerank_non_cs",
                return_value=(
                    [
                        {"index": 0, "relevance": 99, "quality": 85, "reason": "更贴合学者方向"},
                        {"index": 1, "relevance": 15, "quality": 20, "reason": "泛词命中"},
                    ],
                    '{"results":[]}',
                ),
            ), patch("scripts.run_pipeline._run_python", side_effect=_fake_run_python_factory(ranked_payload)):
                run_pipeline(
                    base_dir=base_dir,
                    output_dir=base_dir / "outputs",
                    config={"search": {"top_k": 2, "llm_rerank_top_n": 15}, "llm": {"api_key": "demo", "base_url": "https://example.com", "model": "demo-model"}},
                    aminer_user_id="",
                    topics=[],
                    scholar_name="张帆进",
                    scholar_org="清华大学",
                    paper_titles=[],
                    papers_file="",
                    free_text="推荐论文",
                    target="",
                    account_id="main",
                    skip_dispatch=True,
                )
                ranked_path = base_dir / "outputs" / "papers_ranked.json"
                ranked_data = __import__("json").loads(ranked_path.read_text(encoding="utf-8"))

        self.assertEqual(ranked_data["papers"][0]["title"], "Paper A")
        self.assertEqual(ranked_data["llm_rerank"]["status"], "success")
        self.assertEqual(ranked_data["papers"][0]["llm_rerank_relevance"], 99)


if __name__ == "__main__":
    unittest.main()
