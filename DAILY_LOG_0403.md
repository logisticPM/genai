# Daily Log — 2026-04-03

## Overview

PR 重构、poster 制作、code review 修复。将原始的单一 PR #5 拆分为 3 个独立 PR，修复了 Copilot code review 的 12 个问题，制作了学术 poster 从 v1 迭代到 v17。

---

## 1. PR 管理

### Wiki 发布
- 修复 wiki 中所有 "B's" 内部简称 → 正式表述（"baseline", "original prompt"）
- 更新文件路径引用为 `src/` 结构
- Push 到 GitHub wiki

### PR #5 创建与重构
- 在 PCBZ/FinChartAudit 创建 PR #5（feat/t2-pipeline-experiments-0402）
- 将 1 个 monolithic commit 拆分为 5 个逻辑 commit：
  1. `infra: add OCR, DePlot, and rule engine tools`
  2. `feat: add VLM client abstraction layer`
  3. `feat: add multi-tier agent framework`
  4. `feat: add T2 pipeline experiment scripts`
  5. `docs: add T2 experiment report`
- 文件从根目录重构到 `src/` + `experiments/` + `docs/`

### Copilot Code Review（12 comments）
- 修复 `from finchartaudit.xxx` → `from src.xxx`（13 个文件）
- 修复 8 个实验脚本的硬编码绝对路径 → 环境变量
- registry.py 补充 4 个缺失的 check type enum
- t3_pairing.py 增加 PDF 文件类型检测
- claude_client.py 增加 close()/context manager
- print() → logging
- 修复 Windows tempfile 权限问题

### 深度 Code Review → PR 拆分
- 收到全面的架构级 code review（"Do not merge"）
- 核心问题：新代码重复了 eval_runner.py, prompts.py, config.py, run_pipeline.py
- 关闭 PR #5
- 拆分为 3 个独立 PR：
  - **PR A (PR #6)**: 基础设施重构 — 提取共享 VLM client + metrics 模块
  - **PR B**: 工具层（OCR, DePlot, rule engine）— 等 PR A 合并
  - **PR C**: 实验脚本 — 等 PR B 合并

### Git Worktree 冲突修复
- `.claude/worktrees/keen-lamarr` submodule 残留阻止所有 push
- 根因：PCBZ 在 3/29 意外 commit 了 Claude Code worktree 引用
- 修复：删除所有旧分支（dev, add_readme, refactor, feat/...）+ .gitignore 加 .claude/
- PR #6 成功 push 和创建

---

## 2. Poster 制作（v1 → v17）

### 迭代过程

| 版本 | 改进 |
|------|------|
| v1-v6 | 基础布局、内容填充、图文配对 |
| v7 | 学习 poster 设计准则：<400 词，>=16pt，40-50% 图 |
| v8 | 严格模板区域合规，隐藏干扰元素 |
| v9 | 字体全部 >= 16pt，图 400 DPI |
| v10 | 学习 Ryan 海报风格：底部汇总表格 |
| v11 | 内容和 section 标题对齐（Methodology=方法，Results=结果） |
| v13 | 所有图表重新生成 400 DPI |
| v14 | 系统架构图替代 pipeline 线性图 |
| v15 | 竖向架构图作为 methodology hero |
| v16 | 实验设计流程图替代系统架构（methodology=实验设计不是最终系统） |
| v17 | 修复图重叠、保持正确宽高比 |

### 关键设计准则（学到的）
1. **文字 < 400 词**，图占 40-50%，白空间 25-40%
2. **Title >= 48pt**，Headers >= 28pt，Body >= 18pt，最小 16pt
3. **每张图有结论性 caption**（说结论，不是描述）
4. **Headline finding** 从 6 英尺外可读
5. **Methodology = 实验设计**（控制变量、实验条件），不是最终系统架构
6. **内容必须和 section 标题一致**
7. **颜色 3+1**（蓝/绿/红 + 灰）
8. **图的宽高比**必须和原图一致，不能拉伸

### 生成的图表（400 DPI）
- `fig0_framework.png` — 3-Phase 实验框架图
- `fig_exp_design.png` — 实验设计流程图（Phase 1/2/3 控制变量）
- `fig_architecture.png` — 系统架构图（竖向）
- `fig5_t2_pipeline_comparison.png` — Pipeline 对比柱状图
- `fig7_pipeline_diagram.png` — Pipeline 线性流程图
- `fig8_attention_dilution.png` — Attention dilution 对比图
- `fig9_results_table.png` — 结果汇总表
- `fig1_rq1_rq2_heatmap.png` — RQ1/RQ2 heatmap（重新生成高 DPI）

### 最终 Poster 故事线
Background → RQ → 例图 → Research Gap → 实验设计(Phase 1/2/3) → 
Headline: "Tools as verifiers: +5.9 pp / as augmenters: −1.7 pp" →
Pipeline bar chart → Findings → Attention dilution → Results table → 
Heatmap → Per-type Δ + SEC → Conclusion

---

## 3. 技术工具链
- PowerPoint COM automation（win32com）用于 PPTX → PNG 预览
- python-pptx 用于程序化 poster 生成
- matplotlib 400 DPI 图表生成
- 迭代式设计：生成 → 预览 → 审查 → 修改

---

## 4. 待办
- [ ] 等待 PR #6 code review 通过并合并
- [ ] PR A 合并后创建 PR B（工具层：OCR, DePlot, rule engine）
- [ ] PR B 合并后创建 PR C（实验脚本 + base_pipeline.py）
- [ ] Poster v17 后续微调（如有需要）
- [ ] 准备最终 presentation
