# FinChartAudit -- Implementation Plan

> 2026-03-20 ~ 2026-04-17 (4 weeks)
> Member C (you): Engineering, text pipeline, demo
> Member B: Vision pipeline, Misviz evaluation, prompt engineering
> Member A: SEC data, financial domain, ground truth annotation

---

## Week 0 (Now): Alignment & Unblock

**Goal**: 让三个人能同时开始干活，互不阻塞。

| Task | Owner | Deliverable | Due |
|------|-------|------------|-----|
| 发送 DATA_REQUEST_FOR_A_v3.docx 给 A | C | A 开始搜索 SEC 案例 | 3/20 (Today) |
| 与 B 确认 Misviz-synth 数据获取方式 | C+B | 确认 TUdatalib 下载链接 + data table 格式 | 3/21 |
| 与 B 对齐 2x2 实验设计 | C+B | 确认 vision+text 的 text context 格式（用 Misviz-synth 的 data table） | 3/21 |
| 搭建 git repo + 项目骨架 | C | `finchartaudit/` 目录结构，所有 `__init__.py` | 3/21 |
| 安装 PaddleOCR 并验证 | C | 在一张示例图表上跑通 OCR | 3/21 |
| 安装 Streamlit | C | `pip install streamlit` | 3/20 |

---

## Week 1 (3/22 - 3/28): P0 Foundation + T2 Minimum Viable

**Goal**: 跑通一个端到端最小闭环——上传一张图表 → OCR 提取轴标签 → 规则引擎判断截断轴 → 返回 finding。

### C 的任务 (Engineering)

| Day | Task | Input/Output | 验收标准 |
|-----|------|-------------|---------|
| Mon-Tue | **VLM 统一接口 + Tool Registry** | | |
| | `vlm/base.py` — 抽象接口 | | 定义 `analyze(image, prompt, tools)` 接口 |
| | `vlm/claude_client.py` — Claude function calling | API key | 能发送图片 + tool schema，收到 tool call 响应 |
| | `tools/registry.py` — tool schema 定义 | | 6 个 tool 的 JSON schema |
| | `tools/traditional_ocr.py` — PaddleOCR 封装 | 图片 → text+bbox | 对示例图表提取出轴标签和字号 |
| | `tools/rule_check.py` — 规则引擎 | 轴数值 → 判断 | `truncated_axis` 规则能正确判断 |
| Wed | **最小闭环集成** | | |
| | Claude agent 看图 → 调 OCR → 调 rule_check → 返回 finding | 一张截断轴图表 | 端到端跑通，finding 内容正确 |
| Thu | **PDF Parser** | | |
| | `parser/pdf_parser.py` — PyMuPDF 封装 | PDF → 页面图片 | 能渲染任意页为 PNG |
| | `parser/chart_extractor.py` — 图表提取 | 页面 → 图表图片列表 | 从示例 10-K 中提取出 >=3 张图表 |
| | `parser/text_extractor.py` — 文本提取 | 页面 → 文本 | 提取嵌入文本 |
| | `tools/extract_pdf_text.py` — tool 封装 | | 作为 agent tool 可调用 |
| Fri | **FilingMemory + Data Models** | | |
| | `memory/models.py` — 所有 dataclass | | ChartRecord, AuditFinding, TraceEntry 等 |
| | `memory/filing_memory.py` — 内存存储 | | chart_registry, findings, ocr_cache 读写 |
| | `memory/trace.py` — 审计轨迹 | | TraceEntry 记录 agent 每一步 |

### B 的任务 (Vision Pipeline)

| Day | Task |
|-----|------|
| Mon-Tue | 下载 Misviz-synth 数据集（TUdatalib），检查 data table 和 axis metadata 格式 |
| Wed-Thu | 设计 T2 detection prompt（12 类 misleader 的结构化逐项检测） |
| Fri | 用 Claude API 在 10 张 Misviz-synth 图表上测试 prompt，评估初步效果 |

### A 的任务 (SEC Data)

| Day | Task |
|-----|------|
| All week | 用搜索关键词找 T3 补充案例（3-4 家），P0 目标：Ticker + CIK |
| Fri | **交付 P0**：补充公司的 Ticker + CIK 列表 |

### Week 1 Check-in (3/28 Fri)

- [ ] C: 最小闭环 demo（截断轴检测端到端）
- [ ] B: Misviz-synth 数据就绪 + T2 prompt 初版
- [ ] A: 补充公司 CIK 列表

---

## Week 2 (3/29 - 4/4): T2 Agent Complete + T1/T3 开工

**Goal**: T2 agent 完整支持 12 类 misleader 检测，Single Chart 页面可用。T1 和 T3 开始开发。

### C 的任务

| Day | Task | 验收标准 |
|-----|------|---------|
| Mon | **T2 Agent 完整版** | |
| | `agents/base.py` — Agent 基类 (plan/execute/reflect + tool-use) | |
| | `agents/t2_visual.py` — 完整的 T2 agent | 支持 12 类 misleader，调用 OCR + rule_check |
| | `prompts/t2_visual.py` — T2 prompt 模板（与 B 协作） | |
| Tue | **Single Chart Audit 页面** | |
| | `app.py` — Streamlit 入口 | |
| | Page 1: 上传图表 → 选择模型 → Run Audit → 显示 findings + tool trace | 可以在浏览器中完整使用 |
| Wed | **T1 Agent** | |
| | `agents/t1_numerical.py` | 提取文本 claim + 图表数值 → rule_check 比对 |
| | `prompts/t1_numerical.py` | |
| Thu-Fri | **T3 Agent 开发** | |
| | `agents/t3_pairing.py` — GAAP/Non-GAAP 分类 + 配对矩阵 | 读取 chart_registry → 构建 pairing matrix |
| | `prompts/t3_pairing.py` — SEC 配对规则 prompt | |
| | `tools/query_memory.py` — memory 查询 tool | agent 可查询 chart_registry 和 ocr_cache |

### B 的任务

| Day | Task |
|-----|------|
| Mon-Wed | 在 Misviz-synth 上跑 **vision-only** 条件（Claude + Qwen），先跑 1000 张样本验证 prompt |
| Thu | 准备 **vision+text** 条件的 text context 格式（从 Misviz-synth data table 构建） |
| Fri | 开始跑 vision+text 条件 |

### A 的任务

| Day | Task |
|-----|------|
| Mon-Wed | **P1 交付**：T3 补充公司的 comment letter 摘录 |
| Thu-Fri | 下载已确认公司的 10-K filing（或提供 CIK 让 C 用工具下载） |

### Week 2 Check-in (4/4 Fri)

- [ ] C: Single Chart Audit 页面可用（T2 完整 + T1 基础版）
- [ ] C: T3 agent 初版（能构建 pairing matrix）
- [ ] B: Misviz-synth vision-only 结果（1000 张样本），vision+text 开始跑
- [ ] A: Comment letter 摘录 + filing 下载

---

## Week 3 (4/5 - 4/11): Filing Scanner + DSPM + 评估

**Goal**: Filing Scanner 页面可用（完整 T1-T4），T2 2x2 实验出结果。

### C 的任务

| Day | Task | 验收标准 |
|-----|------|---------|
| Mon | **Orchestrator + Cross-Validator** | |
| | `agents/orchestrator.py` — 调度 T1→T2→T3→T4 | 自动按顺序运行所有 tier |
| | `agents/cross_validator.py` — 跨 tier 关联 | 能升级风险等级（如 T2+T1 同时命中→HIGH） |
| Tue | **Filing Scanner 页面** | |
| | Page 2: 上传 PDF → 解析 → T1-T4 → Summary + Findings + Trace | 在 Myers 的 10-K 上端到端跑通 |
| Wed-Thu | **DSPM Filing Edition** | |
| | 从 `../memory/dspm/` 复制核心代码 | |
| | `dspm/filing_schema.py` — filing 域定义 | 5 个域（mda, financial_statements 等） |
| | `dspm/filing_dspm.py` — 适配后的主编排器 | `process_section()` + `get_context_for_question()` + `get_definition_inconsistencies()` |
| | `tools/query_dspm.py` — DSPM 查询 tool | 4 种模式（context/changelog/events/synthesis） |
| | `agents/t4_cross_section.py` — T4 agent | T4-A 时间窗口 + T4-B 定义一致性 |
| Fri | **T3 Case Study 执行** | |
| | 在 MYE/HURN/FXLV 的 filing 上运行系统 | 产出 pairing matrix + findings |
| | 对比系统输出 vs SEC comment letter 内容 | 记录 match/miss |
| | 验证 T1 输出（人工检查 ~30 个样本） | 记录 precision |

### B 的任务

| Day | Task |
|-----|------|
| Mon-Wed | 完成 Misviz-synth 全量 2x2 实验（81,814 x 4 条件）|
| Thu | 跑 Misviz real (2,604 张) vision-only |
| Fri | 整理实验结果：per-misleader-type accuracy, F1, confusion matrix |

### A 的任务

| Day | Task |
|-----|------|
| Mon-Wed | **P2 交付**：干净对照组公司 2-3 家 |
| Thu-Fri | 协助 T3 case study：确认系统输出与 SEC findings 的匹配度 |

### Week 3 Check-in (4/11 Fri)

- [ ] C: Filing Scanner 页面可用（T1-T4 全流程）
- [ ] C: DSPM 适配完成，T4 能检测定义不一致
- [ ] C: T3 case study 初步结果
- [ ] B: 2x2 实验完整结果
- [ ] A: 干净对照组 + case study 协助

---

## Week 4 (4/12 - 4/17): Tool-Use Ablation + Report + Paper

**Goal**: 完成所有评估，报告生成，论文撰写。

### C 的任务

| Day | Task | 验收标准 |
|-----|------|---------|
| Mon | **Tool-Use Ablation 实验** | |
| | 从 Misviz-synth 采样 ~1,000 张 | |
| | 跑两个条件：VLM-only vs VLM+OCR+rules | |
| | 用 axis metadata 作为 OCR ground truth 评估提取精度 | |
| | 输出：对比表格 + OCR 精度数据 | |
| Tue | **Report Generator** | |
| | `report/generator.py` — PDF/HTML 报告生成 | |
| | `report/templates/` — Jinja2 模板 | |
| | Filing Scanner Export Tab 功能 | 一键下载 PDF 审计报告 |
| Wed | **在干净对照组上运行系统** | |
| | 验证 false positive rate | 干净 filing 上 findings 应该少/无 |
| Thu | **CLI 工具 + Demo 打磨** | |
| | `cli.py` — typer CLI | `finchartaudit audit` 和 `finchartaudit scan` |
| | Demo 录屏准备 | |
| Fri | **Demo 录制 + README** | |
| | 录制 demo video（~3 min） | |
| | 写 README.md（安装、使用、示例） | |

### B 的任务

| Day | Task |
|-----|------|
| Mon-Tue | 分析 2x2 实验结果：哪些 misleader 类型最难？textual context 帮助最大的是哪类？ |
| Wed-Fri | **论文撰写**：Experiment section（2x2 结果 + 对比表 + per-misleader 分析） |

### A 的任务

| Day | Task |
|-----|------|
| Mon-Tue | **论文撰写**：T3 Case Study section（per-company narrative + 与 SEC findings 对比） |
| Wed-Thu | 论文 Related Work + Introduction |
| Fri | 全员 review + 最终提交 |

### Week 4 Check-in (4/17 Thu)

- [ ] C: Tool-use ablation 结果 + 报告生成功能 + CLI + Demo
- [ ] B: 论文 Experiment section 完稿
- [ ] A: 论文 Case Study + Related Work 完稿
- [ ] All: Paper 完整初稿

---

## Critical Path

```
Week 1                Week 2                Week 3                Week 4
  │                     │                     │                     │
C:P0 Foundation ──→ T2 Agent + Page 1 ──→ Orchestrator + ──→ Ablation +
  │ VLM+OCR+Rules      │ T1/T3 Agents        │ DSPM + T4        │ Report + Demo
  │ 最小闭环            │ Single Chart        │ Filing Scanner   │ CLI + Polish
  │                     │                     │ T3 Case Study    │
  │                     │                     │                     │
B:Misviz 数据 ────→ Vision-only 实验 ──→ 完整 2x2 结果 ──→ 论文实验章节
  │ Prompt 设计         │ Vision+text 开始    │ + Real-world     │ 结果分析
  │                     │                     │                     │
A:搜索补充案例 ──→ Comment letter ───→ 干净对照组 ────→ 论文 Case Study
  │ CIK 列表            │ 摘录 + Filing      │ Case study 协助  │ + Related Work
```

**最大风险点**：
- Week 1 Wed: 最小闭环（Claude tool-use + OCR + rule_check）如果跑不通，整个后续都延迟
- Week 2 Fri: B 的 vision+text 条件如果 Misviz-synth data table 格式对不上，需要处理数据格式

**缓冲策略**：
- 如果 Week 1 Claude tool-use 调试慢 → 先用硬编码 pipeline（不走 function calling），demo 可以后续再改成 tool-use
- 如果 Misviz-synth data table 格式有问题 → 先只跑 vision-only（仍然是有效实验，只是少了 RQ2 的 text 条件）
- 如果 T4 DSPM 适配比预期复杂 → T4 降级为论文中的 "future work"，系统仍然完整（T1-T3）

---

## Deliverables Checklist

### Code
- [ ] `finchartaudit/` 完整 Python 包
- [ ] Streamlit demo（Page 1: Single Chart, Page 2: Filing Scanner）
- [ ] CLI 工具（`finchartaudit audit`, `finchartaudit scan`）
- [ ] `data_tools/` 数据收集和标注工具套件

### Evaluation
- [ ] T2 2x2 实验结果（81,814 charts x 4 conditions）
- [ ] T2 real-world 泛化测试（2,604 charts）
- [ ] T2 tool-use ablation（~1,000 charts）
- [ ] T3 case study（3-5 companies vs SEC findings）
- [ ] T4 DSPM demonstration（2-3 filings）
- [ ] False positive test on clean filings（2-3 companies）

### Paper
- [ ] Introduction + System Overview (A+C)
- [ ] Four-Tier Framework (C)
- [ ] Experiment: T2 2x2 + ablation (B)
- [ ] Application: T3 Case Study (A)
- [ ] Method: T4 DSPM (C)
- [ ] Related Work (A)
- [ ] Demo video (~3 min)
- [ ] README.md
