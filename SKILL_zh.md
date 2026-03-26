---
name: aminer-rec5
description: "OpenClaw 个性化论文推荐 skill。当用户在飞书里触发 /aminer-rec5 或 /skill aminer-rec5 时，必须直接运行 {baseDir}/scripts/ 下的本地流水线，支持 aminer_user_id、scholar/name+org、代表论文、papers_file 和自然语言 topics 输入，统一构建 ResearchProfile，完成召回、AMiner enrich、摘要、Feishu 派发，并返回 NO_REPLY。"
homepage: https://github.com/tlysanhuo/aminer-rec
user-invocable: true
disable-model-invocation: false
metadata: { "openclaw": { "emoji": "📚", "requires": { "bins": ["python3"] } } }
---

# aminer-rec5

这个 skill 只处理显式的 `/aminer-rec5` 或 `/skill aminer-rec5` 触发。

## Slash 命令契约

- 每次显式调用都必须重新执行，不复用旧 outputs。
- 不要只回复状态、诊断或“要不要重跑”。
- 不要搜索、安装、升级或修复 skill。
- 成功发送后，只返回 `NO_REPLY`。

## 输入

- 学者增强：
  - `aminer_user_id`
  - `scholar` / `name`
  - `org`
  - `papers`
  - `papers_file`
- topic 增强：
  - `topics`
  - 命令后的自然语言自由描述

示例：

- `/aminer-rec5 topics: 多模态, 智能体`
- `/aminer-rec5 scholar: Jie Tang org: Tsinghua papers: OAG-Bench | RPC-Bench`
- `/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, tool-use`
- `/aminer-rec5 我做多模态智能体和 tool-use，帮我推荐最近论文`

## 执行方式

只允许走统一入口：

```bash
python3 "{baseDir}/scripts/handle_trigger.py" \
  --base-dir "{baseDir}" \
  --text "<原始飞书消息>"
```

`handle_trigger.py` 已经负责：

- 从飞书包装文本里解析 `/aminer-rec5 ...`
- 推断 Feishu 投递目标
- 读取 `{baseDir}/config.example.yaml`
- 调用 `{baseDir}/scripts/run_pipeline.py`
- 通过本地 dispatcher 发送 Feishu 消息

## 失败处理

- 如果没有任何 profile 输入，提示用户补充研究方向、论文或学者线索。
- 如果 `aminer_user_id` 归纳不出方向，提示用户补 `topics` 或代表论文。
- 如果 `SegmentationPro` 失败但用户给了足够 topics / paper signal，继续走降级推荐。
- 如果摘要失败，允许降级。
- 不要退回到其他 skill。
