# Daily Log — 2026-04-02

## Overview

Pipeline vs Vision-Only 论证可行性分析 + OCR 基础设施优化调研。核心结论：当前 benchmark 不支持 "pipeline > vision-only"，需要重新定位贡献；OCR 从 PaddleOCR CPU 切 RapidOCR 可大幅提速。

---

## 1. Pipeline vs Vision-Only — 能否用 Benchmark 证明？

### 现有数据（271-chart Misviz real-world）

| Mode | Model | F1 | Precision | Recall |
|------|-------|-----|-----------|--------|
| **VLM-only** | Haiku 4.5 | **83.0%** | 78.2% | 88.3% |
| LLM-OCR+Rules pipeline | Haiku 4.5 | 80.1% | 77.9% | 82.5% |
| VLM+Rules | Haiku 4.5 | 72.6% | 79.7% | 66.7% |
| **VLM-only** | Sonnet 4 | **86.6%** | 85.3% | 87.9% |
| LLM-OCR+Rules pipeline | Sonnet 4 | 84.9% | 84.9% | 84.9% |

**结论：Vision-only 在 Misviz 上反而赢了**，差异统计不显著 (p≥0.05)。

### 原因分析

1. **85%+ FP 来自 VLM 幻觉**，不是 rule engine 问题——加 rule 解决不了主要矛盾
2. **OCR 证据干扰视觉推理**：结构类 pipeline 赢（axis +36~79pp），视觉类 pipeline 输（misrepresentation -5pp）
3. Clean chart FP rate 几乎相同（42% vs 40%），rule 对抑制误报帮助有限

### Pipeline 的真正价值：互补而非全面优势

| 对比 | 数量 | 典型类型 |
|------|------|---------|
| Pipeline 能检出、VLM-only 漏掉 | **54** | discretized continuous, inconsistent tick, axis range |
| VLM-only 能检出、Pipeline 漏掉 | **17** | misrepresentation, inappropriate line chart |

### 论文定位建议

- **路线 A（推荐）**：承认现状，贡献 = "complementary detection profiles" + 四层框架 + "工具增强不一定提升整体表现"的反直觉发现
- **路线 B**：跑 V3 prompt 全量 eval，若 F1 > 83% 可翻盘

---

## 2. V3 Pipeline 架构梳理

完整流程：

```
图片 → OCR (y_axis, right_axis, x_axis)
     → Rule Engine (truncated, broken_scale, inverted, axis_range, dual, binning)
     → 构建分层 Rule Verdicts
         [CLEAN]/[FLAGGED]: truncated_axis, dual_axis (可靠)
         [INFO]: inverted, axis_range, tick_intervals, binning (不可靠)
     → 单次 VLM 调用 (三段 prompt)
         Section A: 结构类 (用 OCR + rule 证据)
         Section B: 视觉类 (纯看图，不用 OCR)
         Section C: 完整性检查
     → JSON 解析 + confidence 过滤 (< 0.3 去掉)
     → Post-processing Rule Veto
         truncated_axis: 需 rule 确认才保留
         dual_axis: 需 OCR 检到右轴才保留
     → 最终 Findings
```

---

## 3. OCR 基础设施调研

### 环境现状

| 项目 | 状态 |
|------|------|
| GPU | NVIDIA RTX 5060 Laptop 8GB |
| CUDA Version | 13.2 |
| PaddlePaddle | 3.3.0 **CPU 版**（未用 GPU） |
| RapidOCR | 新安装 1.2.3 |

**发现：PaddlePaddle 安装的是 CPU 版，GPU 未被利用。** CUDA 13.2 (RTX 50 系) 太新，PaddlePaddle 官方最高支持 CUDA 12.6。

### OCR 引擎对比（5 张 Misviz real-world 图）

| 引擎 | Y轴识别质量 | 速度 (per crop) |
|------|------------|----------------|
| PaddleOCR 3.4 CPU | 准确 | ~8s |
| RapidOCR CPU | **基本一致**，个别更完整 | **~3.7s** (2x 快) |
| RapidOCR GPU | 同上 | ~3.5s (GPU 对小模型无意义) |

5 张图逐一对比，识别结果基本一致：
- misrepresentation: 完全一致
- truncated axis: 完全一致 (0.6~0.5 六个值)
- 3d: 一致 (SETOS)
- dual axis: Rapid 多抓到 "60"，更完整
- inconsistent tick: 略有差异，都够用

### GPU 加速结论

**GPU 对 OCR 没意义**——ONNX 模型只有几 MB，GPU kernel launch 开销 > 计算收益，仅快 1.1x。真正的加速来自减少 crop 次数。

### OCR Crop 覆盖率分析

| OCR Crop | 服务的检测类型 | 实例占比 | Rule 可靠度 | 能省？ |
|----------|--------------|---------|------------|-------|
| 无需 OCR | misrepresentation, 3d, pie, line, item order | 56.2% | — | 本就不跑 |
| y_axis | truncated axis, tick intervals, inverted, axis range | 16.0% | truncated=[RELIABLE], 其余=[INFO] | 必须保留 |
| right_axis | dual axis | 3.0% | [RELIABLE] | 不能省 |
| x_axis | inconsistent binning, discretized continuous | 4.7% | 都是 [INFO] | **可以省** |

### 推荐方案：RapidOCR + 2 crops (y_axis + right_axis)

| 方案 | 271张耗时 | Rule 覆盖 | 风险 |
|------|----------|----------|------|
| PaddleOCR 3 crops (现状) | 130 min | 100% | 无 |
| RapidOCR 3 crops | 50 min | 100% | 无 |
| **RapidOCR 2 crops (y + right)** | **33 min** | **所有 RELIABLE rules** | x_axis [INFO] 丢失，影响极小 |
| RapidOCR 1 crop (y only) | 17 min | 少了 dual axis veto | dual axis FP 可能上升 |

---

## 4. 未执行的改动

以上均为分析和调研，**尚未修改任何代码**。待确认后执行：

1. `finchartaudit/tools/traditional_ocr.py` — 切换默认 backend 为 RapidOCR
2. `finchartaudit/agents/t2_pipeline.py` — lazy crop 策略（默认 y + right，按需 x）
3. OCR 结果缓存层 — 跑一次后存 JSON，后续 eval 直接读
4. V3 prompt 全量 271-sample eval

## Next Steps

1. **切 RapidOCR + 2 crops** — 改 pipeline，加缓存
2. **跑 V3 prompt 271-sample eval** — 验证 F1 是否超 83% (vision-only baseline)
3. **决定论文定位** — 根据 eval 结果选路线 A 或 B
