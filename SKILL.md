---
name: aminer-rec5
description: "OpenClaw personalized paper recommendation skill client. When the user invokes /aminer-rec5 or /skill aminer-rec5 in Feishu, validate a narrow local input contract, resolve the delivery route, call a configured backend recommendation API, and return the backend's final response contract."
homepage: https://github.com/tlysanhuo/aminer-rec
user-invocable: true
disable-model-invocation: false
metadata: { "openclaw": { "emoji": "📚", "requires": { "bins": ["python3"] } } }
---

# aminer-rec5

Use this skill only for explicit `/aminer-rec5` or `/skill aminer-rec5` requests.

## Contract

- Every explicit invocation is a new backend request.
- Do not answer with status-only text.
- Do not search, install, or repair skills.
- After a successful backend-side dispatch, return exactly `NO_REPLY`.

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

It only does these local steps:

- parse `/aminer-rec5 ...`
- validate and normalize the public input contract
- resolve Feishu route metadata
- call the configured backend API
- return the backend response contract
