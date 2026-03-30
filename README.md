# aminer-rec

Public repository for the `aminer-rec5` OpenClaw / Feishu paper recommendation skill client.

This repo is no longer positioned as the full recommendation pipeline. Its job is intentionally narrow:

- parse `/aminer-rec5 ...` messages
- validate a constrained public input contract
- resolve Feishu delivery route metadata
- call a configured backend recommendation API
- return the backend response contract

The OpenClaw command name remains `/aminer-rec5`.

## Why This Version Exists

This is the public-shareable client cut of `aminer-rec5`:

- secrets removed
- local output artifacts removed
- interface surface narrowed to one entrypoint plus backend API schemas
- README and setup flow rewritten for external users

If you want to post this on social platforms and drive traffic to a personal GitHub repo, this version is the one to publish.

## Highlights

- Thin OpenClaw skill client instead of a full exposed pipeline
- Strict parameter guardrails at the public entrypoint
- Backend-first architecture for profile building, retrieval, ranking, and dispatch
- JSON Schema files for the minimal request and response contract
- Feishu / OpenClaw-friendly route forwarding

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

Set the backend your skill client should call:

- `backend.base_url`: your recommendation backend base URL
- `backend.recommend_path`: endpoint path, default `/v1/recommend-and-dispatch`
- `backend.api_key`: optional bearer token
- `backend.timeout_seconds`: HTTP timeout for the backend call

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

## Interface Contract

The public repo exposes exactly one supported external entrypoint:

```bash
python3 scripts/handle_trigger.py --base-dir . --text "<message>"
```

Everything else under `scripts/` is internal implementation detail and may change without compatibility guarantees.

Input guardrails at the entrypoint:

- `aminer_user_id` must be a 24-character hex string
- `topics`: up to 8 items, 80 characters each
- `paper_titles`: up to 8 items, 300 characters each
- `scholar_name`: up to 80 characters
- `scholar_org`: up to 160 characters
- `free_text`: up to 600 characters
- `papers_file`: JSON only, must stay inside the current skill directory, and is converted into inline `seed_papers` before calling the backend
- delivery routing fields are truncated to safe lengths before dispatch

Minimal backend protocol:

- Request schema: [`schemas/recommend_and_dispatch.request.schema.json`](/Users/tly/work/aminer-rec5-public/schemas/recommend_and_dispatch.request.schema.json)
- Response schema: [`schemas/recommend_and_dispatch.response.schema.json`](/Users/tly/work/aminer-rec5-public/schemas/recommend_and_dispatch.response.schema.json)

Recommended backend behavior:

- accept one normalized request from the skill client
- do profile resolution, retrieval, ranking, summarization, and optional dispatch on the backend side
- return `NO_REPLY` when the backend has already dispatched to Feishu
- return `TEXT` plus `reply_text` when the client should surface a user-visible fallback

## Repository Layout

- `SKILL.md` / `SKILL_zh.md`: OpenClaw skill contract
- `scripts/handle_trigger.py`: the only supported external interface
- `schemas/`: minimal backend request / response schemas
- `config.example.yaml`: backend client configuration example
- `scripts/`: internal or legacy implementation details, not public API

## Backend Config

You can configure the backend with `config.yaml` or environment variables:

- `AMINER_REC_BACKEND_BASE_URL`
- `AMINER_REC_BACKEND_PATH`
- `AMINER_REC_BACKEND_API_KEY`
- `AMINER_REC_BACKEND_TIMEOUT_SECONDS`
- `AMINER_REC_LANGUAGE`

`OPENCLAW_HOME` and `OPENCLAW_SESSIONS_PATH` are still respected for local route inference.

## OpenClaw Install

Clone or copy this repository into your OpenClaw skills directory:

```bash
cp -R ./aminer-rec ~/.openclaw/skills/aminer-rec5
```

Then invoke it in Feishu:

```text
/aminer-rec5 topics: multimodal agents, tool use
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tlysanhuo/aminer-rec&type=Date)](https://www.star-history.com/#tlysanhuo/aminer-rec&Date)

## License

[MIT](LICENSE)
