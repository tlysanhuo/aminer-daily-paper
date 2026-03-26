# aminer-rec

`aminer-rec5` 这个 OpenClaw / 飞书论文推荐 skill 的公开仓库版，支持两种非常顺手的输入方式：

- 学者启动：从 `aminer_user_id`、`scholar + org`、代表论文，或者本地 `papers_file` 出发
- 主题启动：直接给 `topics`，或者一句自然语言，比如“我做多模态智能体和 tool use，帮我推荐最近论文”

两条路径最终都会统一归纳成一个 `ResearchProfile`，再进入同一条推荐流水线：召回、AMiner enrich、摘要生成、卡片渲染和消息派发。

OpenClaw 里的实际命令名仍然是 `/aminer-rec5`。

## 这版适合公开发

这份目录是专门为“放个人仓库、发社交平台引流”整理过的公开版：

- 已去掉真实 token / key
- 已去掉本地运行产物
- 已把硬编码的本机路径改成可移植写法
- 已补齐对外更友好的 README 和安装说明

如果你要发 GitHub 链接、写推文、发朋友圈或技术社区帖子，推荐直接用这版。

## 核心卖点

- 自然语言直接触发论文推荐
- 支持从学者线索冷启动
- topic / scholar 双路径统一画像
- `arXiv + AMiner enrich` 的组合式召回
- 输出结构化摘要和推荐理由
- 直接对接 Feishu / OpenClaw 使用场景
- 内部依赖不可用时可以优雅降级，不至于整个链路跑不起来

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

你可以直接把 `config.yaml` 填完整，也可以把一部分字段留空，让运行时自动从本机 OpenClaw / 环境变量里兜底读取。

- `aminer.token`：你自己的 AMiner token
- `llm.api_key`：你自己的 OpenAI 兼容模型 key
- `llm.base_url` / `llm.model`：你实际使用的模型服务

推荐做法：

- 如果你本机 OpenClaw 已经配好了模型，可以把 `llm.api_key` / `llm.base_url` 留空
- 如果你更想走环境变量，可以把 `aminer.token` 留空，然后提供 `AMINER_TOKEN`

`datacenter.segmentation_url` 是可选项。如果你没有内部分词/解析服务，直接留空即可，链路会退化到较轻量的本地解析逻辑。

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

## 输出产物

运行后会在 `outputs/` 下生成：

- `request_context.json`
- `runtime_config.yaml`
- `user_profile.json`
- `arxiv_candidates.json`
- `papers_ranked.json`
- `papers_summarized.json`
- `feishu_messages.json`

这些都是本地运行产物，公开仓库里不要提交。

## 仓库结构

- `SKILL.md` / `SKILL_zh.md`：OpenClaw skill 契约
- `scripts/`：触发解析、画像构建、召回、摘要、渲染、派发
- `tests/`：回归测试
- `config.example.yaml`：安全版示例配置
- `.env.example`：可选环境变量模板

## 可选内部扩展点

这个公开版保留了几个可选扩展口，方便你在内外部环境都能跑：

- `DATACENTER_SEGMENTATION_URL`：如果你有分词/参数抽取服务，可以接上
- `RECSYS_NEXT_DIR`：如果你有内部画像依赖目录，可以打开 UID 画像增强
- `OPENCLAW_HOME`、`OPENCLAW_CONFIG_PATH`、`OPENCLAW_SESSIONS_PATH`：可以覆盖 OpenClaw 的本地路径

如果这些都没有配置，公开版依然能运行，只是部分增强能力会降级。

## 安装到 OpenClaw

把仓库复制或克隆到你的 OpenClaw skills 目录：

```bash
cp -R ./aminer-rec ~/.openclaw/skills/aminer-rec5
```

然后就可以在飞书里触发：

```text
/aminer-rec5 topics: 多模态智能体, tool use
```

## 开发与测试

```bash
pip install -r requirements-dev.txt
pytest -q
```

说明：当前项目原始实现里还存在几条 `research_profile` 相关的既有失败用例；这些不是这次“公开仓库整理”带出来的问题。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tlysanhuo/aminer-rec&type=Date)](https://www.star-history.com/#tlysanhuo/aminer-rec&Date)

## License

[MIT](LICENSE)
