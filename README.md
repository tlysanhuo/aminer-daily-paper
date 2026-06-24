# aminer-rec

Public repository for an AMiner-powered personalized paper recommendation pipeline, with two smooth entry points:

- scholar bootstrap: start from `aminer_user_id`, `scholar + org`, representative papers, or a local `papers_file`
- topic bootstrap: start from `topics` or free-form natural language such as "I work on multimodal agents and tool use"

The repository turns both paths into one unified `ResearchProfile`, then runs retrieval, AMiner enrichment, ranking, summarization, and local output rendering.

Feishu / OpenClaw integration is optional. The OpenClaw command name remains `/aminer-rec5`.

## Why This Version Exists

This is the public-shareable repo cut of `aminer-rec5`:

- secrets removed
- local output artifacts removed
- hard-coded machine paths replaced with portable defaults
- README and setup flow rewritten for external users

If you want to post this on social platforms and drive traffic to a personal GitHub repo, this version is the one to publish.

## Highlights

- Natural-language-first paper recommendation
- Scholar-aware cold start from AMiner person signals
- Unified topic and scholar profile building
- arXiv retrieval plus AMiner enrich
- Structured summaries and recommendation reasons
- Standalone CLI output as Markdown / JSON
- Optional Feishu / OpenClaw-friendly output format
- Graceful degradation when optional internal components are unavailable

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -e .
```

### 2. Create your config

```bash
cp config.example.yaml config.yaml
```

For standalone CLI usage, fill `config.yaml` explicitly.

- `aminer.token`: your own AMiner token
- `llm.api_key`: your OpenAI-compatible model key
- `llm.base_url` and `llm.model`: the provider/model you want to use

The Feishu / OpenClaw trigger entrypoint can still discover local OpenClaw model settings, but the standalone CLI is intentionally config-first.

`datacenter.segmentation_url` is optional. Leave it empty if you do not have access to an internal segmentation service; the pipeline will fall back to lighter local parsing.

### 3. Run the standalone CLI

```bash
aminer-rec recommend \
  --base-dir . \
  --config config.yaml \
  --topics "multimodal agents, tool use" \
  --start-year 2024
```

The CLI does not require Feishu or OpenClaw. It writes:

- `outputs_cli/recommendation.md`
- `outputs_cli/recommendation_result.json`
- intermediate artifacts such as `user_profile.json`, `papers_ranked.json`, and `papers_summarized.json`

You can also write to explicit paths:

```bash
aminer-rec recommend \
  --config config.yaml \
  --free-text "I work on multimodal agents and tool use. Recommend recent papers." \
  --output-markdown outputs/my_recommendation.md \
  --output-json outputs/my_recommendation.json
```

The script entrypoint remains available for repository-local use:

```bash
python3 scripts/recommend.py --config config.yaml --topics "multimodal agents, tool use"
```

## Example Inputs

Standalone CLI:

```bash
aminer-rec recommend --topics "multimodal agents, tool use"
```

```bash
aminer-rec recommend --topics "LLM reasoning" --language-sort en --start-year 2024
```

```bash
aminer-rec recommend --scholar-name "Jie Tang" --scholar-org "Tsinghua University" --paper-title "OAG-Bench" --paper-title "RPC-Bench"
```

Feishu / OpenClaw command text:

```text
/aminer-rec5 topics: multimodal agents, tool use
```

```text
/aminer-rec5 topics: LLM reasoning language_sort: en start_year: 2024
```

```text
/aminer-rec5 scholar: Jie Tang org: Tsinghua University papers: OAG-Bench | RPC-Bench
```

```text
/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: multimodal, tool use
```

## Interface Contract

The public repo exposes standalone and Feishu / OpenClaw entrypoints:

```bash
aminer-rec recommend --base-dir . --topics "multimodal agents, tool use"
python3 scripts/recommend.py --base-dir . --topics "multimodal agents, tool use"
python3 scripts/handle_trigger.py --base-dir . --text "<message>"
```

Use `aminer-rec recommend` for standalone local usage. `scripts/recommend.py` is the repository-local compatibility entrypoint. Use `scripts/handle_trigger.py` only for Feishu / OpenClaw trigger handling. Everything else under `scripts/` is internal implementation detail and may change without compatibility guarantees.

Input guardrails at the entrypoint:

- `aminer_user_id` must be a 24-character hex string
- `topics`: up to 8 items, 80 characters each
- `paper_titles`: up to 8 items, 300 characters each
- `scholar_name`: up to 80 characters
- `scholar_org`: up to 160 characters
- `free_text`: up to 600 characters
- `papers_file`: JSON only, and must stay inside the current skill directory
- `language_sort`: must be `zh` or `en`; filters papers by language
- `start_year` / `end_year`: integer between 1900–2100; filters papers by publication year
- delivery routing fields are truncated to safe lengths before dispatch

## Outputs

Standalone CLI artifacts are written to `outputs_cli/` by default:

- `recommendation.md`
- `recommendation_result.json`

Pipeline runtime artifacts are written to the selected output directory:

- `request_context.json`
- `user_profile.json`
- `arxiv_candidates.json`
- `papers_ranked.json`
- `papers_summarized.json`

These files are local runtime outputs and should stay out of git.

## Repository Layout

- `SKILL.md` / `SKILL_zh.md`: OpenClaw skill contract
- `pyproject.toml`: install metadata and `aminer-rec` console command
- `aminer_rec/`: package-level CLI dispatcher
- `scripts/recommend.py`: standalone CLI entrypoint
- `scripts/handle_trigger.py`: Feishu / OpenClaw trigger entrypoint
- `scripts/`: internal implementation for parsing, profile building, retrieval, summarization, rendering, and dispatch
- `config.example.yaml`: safe example configuration
- `.env.example`: optional environment variables for local overrides

## Optional Internal Hooks

This public repo keeps a few optional extension points for internal environments:

- `DATACENTER_SEGMENTATION_URL`: enables better query segmentation if you have that service
- `RECSYS_NEXT_DIR`: enables internal UID profile lookup if you have the private dependency tree
- `OPENCLAW_HOME`, `OPENCLAW_CONFIG_PATH`, `OPENCLAW_SESSIONS_PATH`: override local OpenClaw locations

Without them, the public version still runs, but some scholar-boost and routing features will degrade gracefully.

## OpenClaw Install

Clone or copy this repository into your OpenClaw skills directory:

```bash
cp -R ./aminer-rec ~/.openclaw/skills/aminer-rec5
```

Then invoke it in Feishu:

```text
/aminer-rec5 topics: multimodal agents, tool use
```

## License

[MIT](LICENSE)
