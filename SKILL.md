---
name: aminer-rec5
description: "OpenClaw personalized paper recommendation skill. When the user invokes /aminer-rec5 or /skill aminer-rec5 in Feishu, immediately run the local pipeline under {baseDir}/scripts/, accept aminer_user_id, scholar hints, seed paper titles, papers_file, or free-form topic text, build a unified ResearchProfile, retrieve papers, enrich with AMiner, dispatch Feishu cards, and return NO_REPLY."
homepage: https://github.com/tlysanhuo/aminer-rec
user-invocable: true
disable-model-invocation: false
metadata: { "openclaw": { "emoji": "📚", "requires": { "bins": ["python3"] } } }
---

# aminer-rec5

Use this skill only for explicit `/aminer-rec5` or `/skill aminer-rec5` requests.

## Contract

- Every explicit invocation is a new run.
- Do not answer with status-only text.
- Do not search, install, or repair skills.
- After a successful dispatch, return exactly `NO_REPLY`.

## Inputs

- `aminer_user_id`
- `scholar` / `name`
- `org`
- `papers`
- `papers_file`
- `topics`
- free-form natural-language interest description

## Execution

```bash
python3 "{baseDir}/scripts/handle_trigger.py" \
  --base-dir "{baseDir}" \
  --text "<original Feishu message>"
```

`handle_trigger.py` is the only supported entrypoint.
