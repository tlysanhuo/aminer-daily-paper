from __future__ import annotations

from urllib.parse import quote

DEFAULT_ARXIV_API_URL = "https://export.arxiv.org/api/query"
DEFAULT_ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.MM", "cs.MA", "cs.RO"]
DEFAULT_ARXIV_LOOKBACK_DAYS = 60
DEFAULT_ARXIV_MAX_RESULTS = 120
DEFAULT_TOP_K = 5

DEFAULT_LLM_MODEL = "gpt-5-mini"
DEFAULT_LLM_TIMEOUT_SECONDS = 30
DEFAULT_LLM_MAX_CONCURRENT_REQUESTS = 10
DEFAULT_LLM_RETRY_ATTEMPTS = 2
DEFAULT_LLM_RERANK_TOP_N = 15

DEFAULT_SCHOLAR_PROFILE_RECENT_YEARS = 3
DEFAULT_SCHOLAR_PROFILE_MAX_PAPERS = 24
DEFAULT_SCHOLAR_PROFILE_SEED_PAPERS = 12

# Dual-bucket configuration
DUAL_BUCKET_RECENT_LOOKBACK_DAYS = 1095  # 3 years
DUAL_BUCKET_RECENT_MAX_PAPERS = 6
DUAL_BUCKET_ANCHOR_MAX_PAPERS = 12

# Source prior weights for dual-bucket ranking
SOURCE_PRIOR_WEIGHTS = {
    "primary": 1.2,   # Intersection of anchor and recent - highest weight
    "anchor": 0.8,    # Long-term academic identity anchor
    "recent": 0.4,    # Recent new directions
}
DUAL_BUCKET_MATCH_BONUS = 0.8  # Bonus for papers matching both anchor and recent

DEFAULT_ENV_FILE_NAME = ".env.daily"

AMINER_AUTHOR_URL_TEMPLATE = "https://www.aminer.cn/profile/{author_id}"
AMINER_PAPER_URL_TEMPLATE = "https://www.aminer.cn/pub/{paper_id}"
AMINER_PAPER_SEARCH_URL_TEMPLATE = "https://www.aminer.cn/search?t=pub&q={query}"
DEFAULT_AMINER_MAP_URL = "https://datacenter.aminer.cn/gateway/api/v3/paper/get/by/arxiv/ids"
DEFAULT_AMINER_DETAIL_URL = "https://datacenter.aminer.cn/gateway/api/v3/paper/detail/batch/order"
DEFAULT_AMINER_AUTHOR_SEARCH_URL = "https://apiv2.aminer.cn/magic?a=SEARCH__search.search___"
DEFAULT_AMINER_PERSON_SEARCH_URL = "https://datacenter.aminer.cn/gateway/open_platform/api/person/search"
DEFAULT_AMINER_PERSON_PAPERS_URL = "https://apiv2.aminer.cn/n?a=__person.SearchPersonPaper__"


def build_aminer_paper_search_url(query: str) -> str:
    text = str(query or "").strip()
    if not text:
        return ""
    return AMINER_PAPER_SEARCH_URL_TEMPLATE.format(query=quote(text))


def build_aminer_paper_url(paper_id: str) -> str:
    text = str(paper_id or "").strip()
    if not text:
        return ""
    return AMINER_PAPER_URL_TEMPLATE.format(paper_id=text)


def build_aminer_author_url(author_id: str) -> str:
    text = str(author_id or "").strip()
    if not text:
        return ""
    return AMINER_AUTHOR_URL_TEMPLATE.format(author_id=text)
