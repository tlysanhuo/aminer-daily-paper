# aminer-rec

`aminer-rec5` 这个 OpenClaw / 飞书论文推荐 skill 的公开仓库版，现在定位成一个很薄的 skill client。

它只负责这几件事：

- 解析 `/aminer-rec5 ...` 消息
- 校验一组受限的公开输入参数
- 推断 Feishu 投递路由信息
- 调用你配置好的推荐后端 API
- 遵守后端返回的响应契约

OpenClaw 里的实际命令名仍然是 `/aminer-rec5`。

## 这版适合公开发

这份目录是专门为“放个人仓库、发社交平台引流”整理过的公开 client 版：

- 已去掉真实 token / key
- 已去掉本地运行产物
- 已把公开接口压缩到一个入口和一组后端 schema
- 已补齐对外更友好的 README 和安装说明

如果你要发 GitHub 链接、写推文、发朋友圈或技术社区帖子，推荐直接用这版。

## 核心卖点

- 一个很薄的 OpenClaw skill client，而不是把整条推荐流水线暴露出去
- 公开入口参数有明确限制
- 画像、召回、排序、摘要、派发都可以收敛到后端做
- 仓库内直接给出最小 request / response schema
- 直接对接 Feishu / OpenClaw 的路由场景

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备配置

```bash
cp config.example.yaml config.yaml
```

这里不再要求你把整套推荐能力都装在 skill 里，只需要把后端地址配好：

- `backend.base_url`：推荐后端的基础地址
- `backend.recommend_path`：接口路径，默认 `/v1/recommend-and-dispatch`
- `backend.api_key`：可选 bearer token
- `backend.timeout_seconds`：后端请求超时

### 3. 本地跑一个 demo

```bash
python3 scripts/handle_trigger.py \
  --base-dir . \
  --config config.yaml \
  --text "/aminer-rec5 我做多模态智能体和 tool use，帮我推荐最近论文"
```

## 常见触发方式

```text
/aminer-rec5 topics: 多模态智能体, tool use
```

```text
/aminer-rec5 scholar: Jie Tang org: Tsinghua University papers: OAG-Bench | RPC-Bench
```

```text
/aminer-rec5 aminer_user_id: 696259801cb939bc391d3a37 topics: 多模态, tool use
```

## 接口约束

这个公开仓库对外只支持一个入口：

```bash
python3 scripts/handle_trigger.py --base-dir . --text "<message>"
```

`scripts/` 目录里的其它模块都属于内部实现细节，不承诺对外兼容。

入口层现在会做这些限制：

- `aminer_user_id` 必须是 24 位十六进制字符串
- `topics` 最多 8 个，每个最多 80 个字符
- `paper_titles` 最多 8 个，每个最多 300 个字符
- `scholar_name` 最多 80 个字符
- `scholar_org` 最多 160 个字符
- `free_text` 最多 600 个字符
- `papers_file` 只允许 `.json`，路径必须在当前 skill 目录内，客户端会先把它转成内联 `seed_papers`
- 派发用的路由字段会先截断到安全长度再使用

最小后端协议：

- 请求 schema：[`schemas/recommend_and_dispatch.request.schema.json`](/Users/tly/work/aminer-rec5-public/schemas/recommend_and_dispatch.request.schema.json)
- 响应 schema：[`schemas/recommend_and_dispatch.response.schema.json`](/Users/tly/work/aminer-rec5-public/schemas/recommend_and_dispatch.response.schema.json)

推荐后端的职责建议是：

- 接收 skill client 归一化后的请求
- 在后端完成画像、召回、排序、摘要和可选派发
- 如果后端已经完成飞书派发，就返回 `NO_REPLY`
- 如果希望客户端回一段用户可见文本，就返回 `TEXT` 和 `reply_text`

## 仓库结构

- `SKILL.md` / `SKILL_zh.md`：OpenClaw skill 契约
- `scripts/handle_trigger.py`：唯一受支持的对外入口
- `schemas/`：最小后端 request / response schema
- `config.example.yaml`：后端 client 示例配置
- `scripts/`：内部或历史实现细节，不作为公开 API

## 后端配置

你可以通过 `config.yaml` 或环境变量配置后端：

- `AMINER_REC_BACKEND_BASE_URL`
- `AMINER_REC_BACKEND_PATH`
- `AMINER_REC_BACKEND_API_KEY`
- `AMINER_REC_BACKEND_TIMEOUT_SECONDS`
- `AMINER_REC_LANGUAGE`

本地路由推断仍然会使用 `OPENCLAW_HOME` 和 `OPENCLAW_SESSIONS_PATH`。

## 安装到 OpenClaw

把仓库复制或克隆到你的 OpenClaw skills 目录：

```bash
cp -R ./aminer-rec ~/.openclaw/skills/aminer-rec5
```

然后就可以在飞书里触发：

```text
/aminer-rec5 topics: 多模态智能体, tool use
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tlysanhuo/aminer-rec&type=Date)](https://www.star-history.com/#tlysanhuo/aminer-rec&Date)

## License

[MIT](LICENSE)
