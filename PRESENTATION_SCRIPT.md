# FinChartAudit — Presentation Script

**Total time: ~5 minutes**

---

## Slide 1: Title (30 seconds)

Hi everyone. Our project is **FinChartAudit** — detecting misleading financial charts using Vision-Language Models.

I'm Tong Wu, and this is a collaboration with Mengshan Li and Wenshuang Zhou from the Khoury College at Northeastern.

---

## Slide 2: Background & Research Questions (1 minute)

So why does this matter?

A study found that **86% of S&P 500 companies** include at least one misleading chart in their annual reports. These include things like truncated axes that exaggerate trends, or Non-GAAP metrics presented without their GAAP equivalents. In fact, Non-GAAP prominence has been the **number one SEC comment letter topic for nine consecutive years**. And yet, detection is still done entirely manually — it's slow, subjective, and doesn't scale.

No prior work has applied Vision-Language Models to this problem. So we asked three questions:

- **RQ1**: How accurately can current VLMs like Claude and Qwen detect different types of chart misleaders?
- **RQ2**: Does adding textual context — like bounding box coordinates — improve detection?
- And **RQ3**: Can we use tools like OCR and classifiers to improve the pipeline's per-type detection quality and reduce false positives?

---

## Slide 3: Methodology (1.5 minutes)

Our methodology has three phases.

**Phase 1** is a baseline experiment. We ran a two-by-two factorial design: two models — Claude Haiku-4.5 and Qwen3-VL-8B — crossed with two conditions — vision-only and vision-plus-text. We evaluated on 271 charts from the Misviz benchmark, which covers 12 different misleader types.

The baseline revealed something important: Claude achieved an 83% binary F1, which sounds great. But when we measured **per-type F1** — whether the model correctly identifies the *specific type* of misleading — it dropped to just 39.3%, with a 42% false positive rate on clean charts. Binary F1 was masking poor type-level accuracy.

This motivated **Phase 2**: a pipeline ablation study. We tested six different tool-use strategies, all on the same 271 charts using Claude. The key question was: **how should tools augment VLM inference?** We compared injecting tool data into the prompt versus using tools as post-processing verifiers. We used RapidOCR for axis extraction, DePlot for chart-to-table conversion, and a ViT-B/16 classifier trained on 57,000 synthetic charts.

**Phase 3** validated our best pipeline on real SEC 10-K filings from Constellation Brands — STZ — comparing against SEC comment letter ground truth.

---

## Slide 4: Results (1.5 minutes)

Here are our key results.

For **RQ1 and RQ2**: Claude significantly outperforms Qwen — 83% versus 72.3%, a gap of 10.7 percentage points. Adding textual context improves precision but reduces recall, with a marginal net effect on F1. So textual grounding doesn't fundamentally change detection performance.

Now for **RQ3** — this is our main contribution. Look at the results table. The red rows show that **tool injection hurts**: injecting OCR or DePlot data into the VLM prompt actually *decreased* per-type F1 by up to 1.7 points. The noisy OCR data was misleading the VLM.

But the green rows tell a different story. Using the **same tools as post-processing verifiers** — applying them *after* VLM inference to filter false positives — improved per-type F1 significantly. Our best pipeline, CLEAN Veto plus ViT Classifier, achieved **45.2% per-type F1** — a 5.9 point improvement with only 82 false positives, down from 157.

We also found that self-consistency voting — running three votes and taking the majority — actually *increased* false positives. VLM errors are systematic, not random, so voting doesn't help.

For the **SEC case study**: on 9 real charts from STZ's 10-K filing, the baseline VLM flagged all 9 as misleading — but 6 were false positives, giving only 33% precision. Our pipeline correctly vetoed all 6 false positives — pie charts and hallucinated misrepresentation — while keeping the true truncated-axis detections. Precision went from 0.33 to 1.0, and F1 from 0.50 to 0.80.

---

## Slide 5: Conclusion & Future Work (30 seconds)

To summarize our key findings:

The core insight is that **tools work as verifiers, not augmenters**. Post-processing verification improves per-type F1 by 15%, while injecting tool data into the prompt degrades it.

We also discovered that VLM errors follow a pattern we call **attention dilution** — the model can detect subtle misleader types when asked individually, but misses them in a multi-type prompt. This is an attention problem, not a capability limitation, which is why post-processing verifiers that check for specific types are effective.

For future work, we'd like to fine-tune VLMs specifically on financial charts, address the domain gap in our classifier, and build a chart-specific SEC dataset to scale coverage beyond the current case study.

Thank you. Happy to take any questions.
