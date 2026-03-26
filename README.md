# aminer-rec

Public repository for the `aminer-rec5` OpenClaw / Feishu paper recommendation skill, with two smooth entry points:

- scholar bootstrap: start from `aminer_user_id`, `scholar + org`, representative papers, or a local `papers_file`
- topic bootstrap: start from `topics` or free-form natural language such as "I work on multimodal agents and tool use"

The repository turns both paths into one unified `ResearchProfile`, then runs retrieval, AMiner enrichment, summarization, card rendering, and delivery.

The OpenClaw command name remains `/aminer-rec5`.

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
- Feishu / OpenClaw-friendly output format
- Graceful degradation when optional internal components are unavailable

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Create your config

```bash
cp config.example.yaml config.yaml
```

Replace the placeholder values in `config.yaml`:

- `aminer.token`: your own AMiner token
- `llm.api_key`: your OpenAI-compatible model key
- `llm.base_url` and `llm.model`: the provider/model you want to use

`datacenter.segmentation_url` is optional. Leave it empty if you do not have access to an internal segmentation service; the pipeline will fall back to lighter local parsing.

### 3. Run a local check

```bash
python3 scripts/handle_trigger.py \
  --base-dir . \
  --config config.yaml \
  --text "/aminer-rec5 I work on multimodal agents and tool use. Recommend recent papers."
```

## Example Inputs

```text
/aminer-rec5 topics: multimodal agents, tool use
```

```text
/aminer-rec5 scholar: Jie Tang org: Tsinghua University papers: OAG-Bench | RPC-Bench
```

```text
/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: multimodal, tool use
```

## Outputs

Runtime artifacts are written to `outputs/`:

- `request_context.json`
- `runtime_config.yaml`
- `user_profile.json`
- `arxiv_candidates.json`
- `papers_ranked.json`
- `papers_summarized.json`
- `feishu_messages.json`

These files are local runtime outputs and should stay out of git.

## Repository Layout

- `SKILL.md` / `SKILL_zh.md`: OpenClaw skill contract
- `scripts/`: trigger parsing, profile building, retrieval, summarization, rendering, dispatch
- `tests/`: regression tests
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

## Development

```bash
pip install -r requirements-dev.txt
pytest -q
```

Note: there are currently a few pre-existing `research_profile` test failures in the original project logic. They are unrelated to the public packaging changes in this repo cut.

## License

[MIT](LICENSE)
