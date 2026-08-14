# DSH-Wiki（LLM-Wiki 引擎）

对论文 **《Retrieval as Reasoning: Self-Evolving Agent-Native Retrieval via LLM-Wiki》**（arXiv:2605.25480）的独立实现：把文档**编译**成带双向链接的结构化 Wiki，查询时由 Agent 组合 `wiki_search` / `wiki_read` 进行遍历推理，并通过 **Error Book** 实现持久的自我纠错。

本仓库是**纯 Python 引擎 + CLI**。它有两种用法：

1. **独立使用** —— 直接用命令行编译文档、提问、校验、修复（本文档主要内容）；
2. **作为 DeepSeek Harness 的 agent 插件引擎** —— 搭配
   [dsh-llm-wiki](https://github.com/detpecca/dsh-llm-wiki) 插件，让 DSH agent
   通过 `wiki_search` / `wiki_read` / `wiki_ingest` 等工具直接管理知识库。
   插件通过本 CLI 的 `--json` 通道工作，安装与配置方法见插件仓库的 README。

核心概念（论文 §3）：

- **编译（Compilation）**：文档不是切块嵌入，而是由 LLM 改写成结构化、互链的 Wiki 页面（索引期，离线）；
- **遍历（Traversal）**：查询时 Agent 搜索、阅读、跟随 `[[wikilink]]`，直到证据充分才作答（查询期，在线）；
- **Error Book**：编译中的系统性错误被记录、归因、转化为约束注入后续编译，并由双层修复机制清理。

## 安装

要求 Python ≥ 3.10，唯一运行时依赖是 `pyyaml`。

**方式一：直接从 GitHub 安装（推荐，无需克隆）**

```bash
pip install git+https://github.com/detpecca/DSH-Wiki.git
```

**方式二：克隆后可编辑安装（开发用）**

```bash
git clone https://github.com/detpecca/DSH-Wiki.git
cd DSH-Wiki
pip install -e ".[dev]"   # dev 含 pytest
```

安装后提供两个等价入口：`python -m llm_wiki` 和 `llm-wiki` 命令。下文统一用前者。

## 配置 LLM

`ingest` / `query` / `fix --finalize` 需要 LLM（编译与修复本质是 LLM 驱动的）；
`search` / `read` / `stats` / `validate` / `errorbook` / 不带 `--finalize` 的 `fix`
**不需要** LLM key。

任何 OpenAI 兼容端点均可，用三个环境变量配置：

```bash
export LLM_WIKI_BASE_URL="https://api.moonshot.cn/v1"   # 默认值，可换任何兼容端点
export LLM_WIKI_API_KEY="sk-..."                        # 必填（仅 LLM 相关命令）
export LLM_WIKI_MODEL="kimi-k2-0711-preview"            # 默认值，可换
```

本地模型（Ollama 等）：`LLM_WIKI_BASE_URL=http://localhost:11434/v1`。

> 通过 DSH 插件使用时，这三个值也可以写进插件配置（`llmWiki*` 键），见
> [dsh-llm-wiki 的配置说明](https://github.com/detpecca/dsh-llm-wiki#配置项)。

## 快速开始

```bash
# 1. 编译文档进 Wiki（算法 1 全流程：SelectPages → 编译 → 校验 → Error Book → 修复 → 定稿）
python -m llm_wiki --wiki ./wiki ingest my_notes.txt

# 2. 提问（Agent 遍历：搜索 → 阅读 → 跟链接 → 充分性检查 → 作答）
python -m llm_wiki --wiki ./wiki query "哪部电影的导演更年长？"

# 3. 校验与修复
python -m llm_wiki --wiki ./wiki validate
python -m llm_wiki --wiki ./wiki fix --finalize
```

无 API key 时可跑脚本化端到端演示（编译本论文自身 + 多跳查询）：

```bash
python examples/demo_paper.py
```

## CLI 完整参考

> **`--wiki` 是顶层选项，必须放在子命令之前**（argparse 规则）：
> `python -m llm_wiki --wiki ./wiki <子命令> ...`。写反会报
> `unrecognized arguments`。`--wiki` 默认值为 `./wiki`。

所有命令（除 `query`）都支持 `--json`：输出单一 JSON 文档到 stdout，供程序消费
（DSH 插件即走此通道）。JSON 均 `ensure_ascii=False`，中文原样输出。

### `ingest <file> [--json]` — 编译入库

把源文本文件编译进 Wiki。完整执行论文算法 1：逐段落 SelectPages（LLM 选目标页，
上限 k=5）→ 编译（注入 Error Book 约束）→ 结构/内容校验 → 错误归因 → 代码自动修复
→ 应用更新（写 digest/页面/回链/重建索引）→ 每 10 篇 LLM 周期修复 → `finalize()`
定稿。编译循环内**恒以 exit 0 结束**，单条段落的跳过/失败体现在 `skipped` 中
（例外：源文件路径不存在时 `read_text` 在循环前抛异常，以非 0 退出）。

`--json` 输出：

```json
{
  "source": "my_notes.txt",
  "passages": 12,
  "written": ["concepts/retrieval", "entities/paper"],
  "pages": 42,
  "openErrorEntries": 0,
  "skipped": [{"id": "p007", "reason": "no pages selected"}]
}
```

### `query "<问题>"` — Agent 遍历问答

论文 §3.2 的 ReAct 遍历循环：`T_max=15` 工具调用预算、连续空搜索耐心 `P=3`、
作答前必须至少一次 `wiki_read`。工具协议为 JSON action（兼容不支持 function
calling 的端点）。无 `--json`。

### `search "<query>" [--limit N] [--json]` — 检索原语

结构化信号打分：页名 8 > 别名 6 > 标签 4 > 摘要 2 > 正文 1（CJK bigram 分词，
无向量嵌入）。`--limit` 默认 10。`--json` 输出数组（`score` 为整数权重累加，
越高匹配越强，非相似度）：

```json
[{"path": "concepts/retrieval", "score": 14, "aliases": ["检索"], "tags": ["ai"],
  "summary": "summary of Retrieval"}]
```

### `read <path...> [--json]` — 读页原语

批量读页，路径相对于 wiki 根、不带 `.md` 后缀（带后缀会自动去除）。可读普通页
（`concepts/retrieval`）、目录索引（`concepts/_index`）、根索引（`index`）、
digest（`sources/digests/s-001`）。`--json` 输出 `{path: content}` 对象：

- 缺失页 → 值为 `"(page not found)"`；
- 不安全路径（`..`、绝对路径等）→ 值为 `"(invalid or unreadable path)"`。

### `stats [--json]` — 统计

```json
{"pages": 42, "categories": {"concepts": 20, "entities": 22},
 "digests": 12, "errorBookEntries": 0}
```

### `validate [--json]` — 结构校验

4 类**确定性**结构校验：悬空链接（`dangling_link`）、不完整页面
（`incomplete_page`）、畸形来源引用（`malformed_reference`）、索引不一致
（`index_inconsistency`）。**无错误 exit 0，有错误 exit 1**（`--json` 下同）。

```json
{"ok": false, "errors": [{"type": "dangling_link", "page": "concepts/retrieval",
  "detail": "links to entities/missing which does not exist"}]}
```

> 第 5 类确定性检查 `unseen_overwrite`（LLM 改了 SelectPages 没选的页）需要编译期
> 的选中集合，**只在 `ingest` 内部触发**，`validate` 不含此项。
> 另有 2 类 LLM 内容校验（`unsupported_fact`、`cross_page_contradiction`），
> 只在 `ingest` / `fix --finalize` 流程内运行。

### `fix [--finalize] [--json]` — 修复

- **不带 `--finalize`**：只跑确定性代码修复（`code_fix_wiki`：重建目录/全局索引、
  补双向 `[[wikilink]]` 回链），**不需要 LLM key**；
- **带 `--finalize`**：追加论文 §3.3 定稿阶段——3 轮 代码修复 ↔ LLM 修复 +
  跨页一致性扫描 + Error Book Verify & Close。**需要 LLM key**，较慢。

`--json` 输出：

```json
{"codeFixes": ["entities/paper <- concepts/retrieval", "rebuilt indices"],
 "finalized": true,
 "repaired": ["concepts/retrieval"], "closedErrorEntries": 2, "openErrorEntries": 0}
```

### `errorbook [--json]` — 错误记录本

查看 Error Book 全部条目（五阶段状态机：Discover → Attribute → Constrain →
Inject → Verify & Close）。`--json` 输出 `{"entries": [...]}`，条目含
`id` / `status` / `type` / `page` / `occurrences` / `constraint_rule` 等字段。

## Wiki 目录结构

```
./wiki/                      # --wiki 指向的根目录
├── index.md                 # 全局索引（自动重建）
├── concepts/                # 6 个固定分类之一
│   ├── _index.md            # 目录索引（自动重建）
│   └── retrieval.md         # 知识页：frontmatter + 摘要 + Key Facts/Related Pages/Related Sources
├── entities/ events/ systems/ benchmarks/ topics/
└── sources/
    ├── digests/             # 每段的结构化摘要（事实溯源）
    └── articles/
./error_book.yaml            # ⚠ 在 wiki 目录的【同级】，不在 wiki 内部
```

页面之间通过 `[[dir/Page]]` 双向链接（写入时自动维护回链）。**请通过 CLI/插件修改
wiki**，手动编辑文件会绕过索引与回链维护，导致 `index_inconsistency`。

## 论文 → 代码 映射

| 论文 | 实现 |
|---|---|
| 算法 1 索引期编译（附录 D） | `llm_wiki/compile.py`（`Compiler.compile_passage` 逐行对应） |
| 页面 Schema（附录 E） | `llm_wiki/schema.py` + `store.py`（frontmatter + 三个必备章节 + 双向 wikilink） |
| 7 类错误分类（附录 F） | `llm_wiki/validators.py`（5 类确定性 + 2 类 LLM 验证） |
| Error Book 五阶段（§3.3） | `llm_wiki/error_book.py`（Discover→Attribute→Constrain→Inject→Verify&Close） |
| 双层修复（§3.3） | `code_autofix` / `llm_periodic_fix` / `finalize`（3 轮循环） |
| wiki_search/wiki_read（§3.2） | `llm_wiki/search.py`（结构化信号优先）+ `store.read_many` |
| 遍历策略与终止（§3.2、附录 H） | `llm_wiki/agent.py`（Tmax=15，P=3，作答前至少一次 wiki_read） |

## 超参数（论文 §4.4）

`T_max=15`（工具调用预算）、`P=3`（连续空搜索耐心阈值）、`k=5`（SelectPages 上限）、每 10 篇文章触发一次 LLM 周期修复——均为默认值，可在 `agent.py` / `compile.py` 中调整。

## 测试

```bash
pytest tests/ -q   # FakeLLM 驱动，无需 API key
```

## 与论文的差异（刻意取舍）

- **wiki_search 无向量嵌入**：以页名/别名/标签/摘要的结构化匹配为主、正文匹配回退（论文本就如此排序优先级）；未接入嵌入模型。
- **未复现实验**：不含基准评测代码（HotpotQA/MuSiQue/2Wiki/AuthTrace）。
- **Agent 工具协议为 JSON action** 而非原生 function calling：兼容任意 OpenAI 兼容端点（含不支持 tools 参数的本地模型）。

## 相关仓库与文档

- [dsh-llm-wiki](https://github.com/detpecca/dsh-llm-wiki) — DeepSeek Harness 插件，
  把本引擎包装成 agent 工具（`wiki_search` / `wiki_read` / `wiki_stats` /
  `wiki_validate` / `wiki_fix` / `wiki_errorbook` / `wiki_ingest`），安装只需两条命令；
- [LLM-Wiki](https://github.com/detpecca/LLM-Wiki) — 论文原始参考实现；
- `DOCUMENTATION.md` — 本代码库的逐模块导读（推荐阅读顺序 + 模块详解）；
- `LLM Wiki.pdf` — 论文原文。

## 许可证

MIT（见 `LICENSE`）。
