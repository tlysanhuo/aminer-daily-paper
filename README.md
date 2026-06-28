<div align="center">

# 📚 aminer-rec

**Stop drowning in arXiv. Start reading what actually matters to *you*.**

A personalized paper-recommendation engine that turns *one sentence* about your research into a ranked, summarized reading list — built on AMiner + arXiv + LLMs.

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![CLI](https://img.shields.io/badge/Built%20for-Researchers-orange.svg)
![Status](https://img.shields.io/badge/status-public%20beta-success)

</div>

---

## ✨ Why you'll like it

Picture this: you walk in Monday morning, type

> *"I work on multimodal agents and tool use"*

…and a minute later you get a clean **ranked shortlist** of recent papers, each with a **plain-language summary** and a **one-line reason** for why it landed in your feed. No more 200-tab arXiv sessions. No more "did I miss the important one?".

`aminer-rec` does the whole loop for you:

| You give it | It gives you back |
|---|---|
| a sentence, a topic list, **or** a scholar name | a focused list of recent, relevant papers |
| your AMiner scholar id / seed papers | a *profile-aware* ranking tuned to *your* taste |
| `--language-sort en` / `--start-year 2024` | filtering by language and year |

## 🎯 Two ways to start, one unified pipeline

Pick whichever feels lazier:

- 🧠 **Topic bootstrap** — just describe what you do in plain language.
  > `--free-text "I work on multimodal agents and tool use"`
- 🎓 **Scholar bootstrap** — start from an `aminer_user_id`, a name + org, or a few seed paper titles. The pipeline builds a `ResearchProfile` from your real publication history and uses it to rank.

Both paths collapse into a single `ResearchProfile`, then flow through the same pipeline:

```
                  ┌─────────────────────────────────────────────┐
  topics / text ──▶│                                             │
                  │            build ResearchProfile             │
  scholar / id  ──▶│                                             │
                  └──────────────────────┬──────────────────────┘
                                         │
          ┌──────────────────────────────▼──────────────────────────────┐
          │   arXiv retrieval   ▶   AMiner enrichment   ▶   ranking      │
          └──────────────────────────────┬──────────────────────────────┘
                                         │
          ┌──────────────────────────────▼──────────────────────────────┐
          │   LLM summaries + recommendation reasons   ▶   rendering     │
          └──────────────────────────────┬──────────────────────────────┘
                                         │
                          Markdown · JSON · (optional) Feishu cards
```

## 🚀 Quick Start

### 1 · Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install --upgrade pip
pip install -e .
```

### 2 · Configure

```bash
cp config.example.yaml config.yaml
```

Fill in three lines in `config.yaml`:

```yaml
aminer:
  token: "<your AMiner token>"
llm:
  api_key:  "<your OpenAI-compatible key>"
  base_url: "<your provider endpoint>"
  model:    "gpt-5-mini"
```

> 💡 `datacenter.segmentation_url` is optional — leave it empty and the pipeline falls back to lighter local parsing.

### 3 · Get recommendations

```bash
aminer-rec recommend \
  --config config.yaml \
  --topics "multimodal agents, tool use" \
  --start-year 2024
```

You'll find the results in `outputs_cli/`:

- `recommendation.md` — your readable reading list with summaries
- `recommendation_result.json` — the full structured result
- plus intermediate artifacts (`user_profile.json`, `papers_ranked.json`, `papers_summarized.json`)

## 🧪 More examples

Describe yourself in one line:

```bash
aminer-rec recommend \
  --free-text "I work on multimodal agents and tool use. Recommend recent papers." \
  --output-markdown outputs/mine.md \
  --output-json outputs/mine.json
```

English papers since 2024 on LLM reasoning:

```bash
aminer-rec recommend --topics "LLM reasoning" --language-sort en --start-year 2024
```

Cold-start from a scholar + their signature papers:

```bash
aminer-rec recommend \
  --scholar-name "Jie Tang" --scholar-org "Tsinghua University" \
  --paper-title "OAG-Bench" --paper-title "RPC-Bench"
```

Or use the in-repo entrypoint:

```bash
python3 scripts/recommend.py --config config.yaml --topics "multimodal agents, tool use"
```

## 🐦 Feishu / OpenClaw mode (optional)

Drop the repo into your skills folder:

```bash
cp -R ./aminer-rec ~/.openclaw/skills/aminer-rec5
```

Then in Feishu:

```text
/aminer-rec5 topics: multimodal agents, tool use
/aminer-rec5 topics: LLM reasoning  language_sort: en  start_year: 2024
/aminer-rec5 scholar: Jie Tang  org: Tsinghua University  papers: OAG-Bench | RPC-Bench
/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37  topics: multimodal, tool use
```

The OpenClaw command name is `/aminer-rec5`.

## ✅ What you get out of the box

- 💬 **Natural-language-first** — one sentence is enough
- 🔍 **arXiv + AMiner enrichment** — broad recall, deep metadata
- 👤 **Scholar-aware cold start** — ranking tuned to your real profile
- 📝 **Structured summaries** — every paper gets a plain-language summary *and* a recommendation reason
- 📄 **CLI output** — Markdown / JSON, version-control friendly
- 🐧 **Graceful degradation** — missing optional components never break the run
- 🚦 **Input guardrails** — sensible limits keep runs safe and reproducible

## 📂 Repository layout

| Path | Purpose |
|---|---|
| `SKILL.md` / `SKILL_zh.md` | OpenClaw skill contract |
| `pyproject.toml` | install metadata + `aminer-rec` console command |
| `aminer_rec/` | package-level CLI dispatcher |
| `scripts/recommend.py` | standalone CLI entrypoint |
| `scripts/handle_trigger.py` | Feishu / OpenClaw trigger entrypoint |
| `scripts/` | core pipeline: parsing, profile, retrieval, summarization, rendering, dispatch |
| `config.example.yaml` | safe example configuration |

## 🛠️ Interface contract & guardrails

Public entrypoints:

```bash
aminer-rec recommend --base-dir . --topics "multimodal agents, tool use"
python3 scripts/recommend.py      --base-dir . --topics "multimodal agents, tool use"
python3 scripts/handle_trigger.py --base-dir . --text "<message>"
```

- `aminer_user_id` — 24-character hex string
- `topics` — up to 8, ≤ 80 chars each
- `paper_titles` — up to 8, ≤ 300 chars each
- `scholar_name` ≤ 80 chars · `scholar_org` ≤ 160 chars · `free_text` ≤ 600 chars
- `papers_file` — JSON only, must stay inside the skill directory
- `language_sort` — `zh` or `en`
- `start_year` / `end_year` — integers in 1900–2100

Routing fields are truncated to safe lengths before dispatch.

## 🔌 Optional internal hooks

Missing any of these? The pipeline still runs — some features just degrade gracefully.

- `DATACENTER_SEGMENTATION_URL` — better query segmentation
- `RECSYS_NEXT_DIR` — internal UID profile lookup
- `OPENCLAW_HOME` / `OPENCLAW_CONFIG_PATH` / `OPENCLAW_SESSIONS_PATH` — override local OpenClaw paths

## 📄 License

[MIT](LICENSE) — read papers, ship ideas, attribute kindly.
