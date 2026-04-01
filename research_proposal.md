# Research Proposal
## Cross-Document Consistency Verification for Private Equity Due Diligence Using LLM-Powered Multi-Agent Systems

**Course:** CS6180 Generative AI
**Proposed By:** [Your Name]
**Date:** March 2026

---

## 1. Problem Statement

Private equity due diligence (DD) requires analysts to cross-reference hundreds of pages across fundamentally different document types—legal contracts, financial filings, and news—within tight timelines of 4–8 weeks. The core challenge is not information extraction from a single document, but **detecting semantic inconsistencies across heterogeneous document sources** about the same target company. A contract may stipulate revenue-sharing terms that contradict reported financials; an MD&A section may claim no material litigation while news archives report an active regulatory investigation.

Current LLM-based document analysis tools (Harvey, CoCounsel, TalkToEDGAR) address single-document extraction and QA. Academic work—CUAD for contracts, FinanceBench for financial QA, CLAUSE for legal contradiction detection—is similarly siloed within document types. **No existing system evaluates cross-heterogeneous-document consistency verification**, the task most central to PE DD workflows.

This project proposes **XDCV-DD** (Cross-Document Consistency Verification for Due Diligence): a multi-agent LLM pipeline for detecting contradictions across legal contracts, financial filings, and news signals about the same company, evaluated on real-world corporate inconsistency cases sourced from SEC enforcement actions and public investigations.

---

## 2. Research Questions

**RQ1 (Core):** Can a multi-agent LLM pipeline reliably detect semantic contradictions across heterogeneous document types (legal contracts, financial filings, news) about the same entity, and which contradiction types are hardest to detect?

**RQ2 (Architecture):** Does structured multi-agent decomposition (specialized extraction → cross-document comparison) outperform naive full-context prompting for cross-document contradiction detection, and under what conditions?

**RQ3 (Taxonomy):** What taxonomy of cross-document contradiction types exists in real corporate filings, and how does detection difficulty vary across types?

---

## 3. Related Work

### 3.1 Contract Understanding
CUAD (Hendrycks et al., 2021) provides 510 annotated contracts across 41 clause types, serving as the standard extraction benchmark. ContractNLI (Koreeda & Manning, 2021) extends this to document-level NLI for contract clauses. CLAUSE (2025) stress-tests LLMs by injecting perturbations into 7,500+ CUAD-derived contracts, finding that current models are highly fragile to subtle legal contradictions.

### 3.2 Financial Document QA
FinanceBench (Islam et al., 2023) provides 10,231 QA pairs over SEC filings (10-K, 10-Q, 8-K), covering numerical reasoning, retrieval, and logical inference. Critically, it explicitly excludes multi-document and cross-filing tasks. FinQA and ConvFinQA focus on numerical reasoning within single documents.

### 3.3 Contradiction Detection
LegalWiz (NeurIPS 2025 Workshop) proposes a multi-agent framework for generating and detecting contradictions in legal documents, finding that even GPT-4 performs near chance on subtle intra-document inconsistencies. Fraunhofer (2024) applies embedding-based pre-filtering followed by LLM querying to detect contradictions within German financial reports. ContraDoc (Li et al., 2023) studies self-contradictions within individual documents using LLMs.

### 3.4 The Gap
All existing work operates within single document types. No prior system addresses cross-heterogeneous-document contradiction detection—the setting where contracts, financial filings, and external news must be jointly analyzed for consistency.

| System | Contracts | Financials | News | Cross-Doc | Contradiction |
|--------|-----------|------------|------|-----------|---------------|
| CUAD | ✅ | ❌ | ❌ | ❌ | ❌ |
| ContractNLI | ✅ | ❌ | ❌ | ❌ | Partial |
| FinanceBench | ❌ | ✅ | ❌ | ❌ | ❌ |
| CLAUSE | ✅ | ❌ | ❌ | ❌ | ✅ |
| Fraunhofer | ❌ | ✅ | ❌ | ❌ | ✅ |
| ContraDoc | ✅ | ❌ | ❌ | ❌ | ✅ |
| **XDCV-DD (Ours)** | ✅ | ✅ | **✅** | **✅** | **✅** |

---

## 4. Contradiction Taxonomy

We propose a structured taxonomy of cross-document contradiction types in corporate DD contexts, grounded in analysis of real SEC enforcement cases:

**Type 1 — Financial-Contract Contradiction**
Quantitative terms stated in contracts (revenue splits, loan covenants, earn-out thresholds) that conflict with values reported in financial filings.
> *Example: Under Armour's 10-K reported organic revenue growth narrative while SEC enforcement later revealed systematic "pull-forward" of $408M in orders across six quarters.*

**Type 2 — Disclosure-News Contradiction**
Risk disclosures or forward-looking statements in MD&A/10-K that conflict with externally reported events.
> *Example: Super Micro Computer's FY2023 10-K Risk Factors contained no disclosure of accounting manipulation risk; Hindenburg Research published a detailed report alleging revenue recognition manipulation and related-party transaction issues (August 2024).*

**Type 3 — Intra-Filing Temporal Contradiction**
Inconsistencies between the same company's filings across time periods, where a material change occurs without required disclosure.
> *Example: Super Micro Computer's FY2023 10-K assessed internal controls as effective; FY2024 10-K received an adverse opinion from BDO on internal controls—with no 8-K timely disclosing the deterioration between periods.*

**Type 4 — Contract-Contract Contradiction** *(Stretch Goal)*
Conflicting terms across multiple contracts relating to the same asset or counterparty.
> *Example: Two Exhibit 10.x agreements assign overlapping IP licensing rights to different parties.*

We prioritize Types 1–3 for implementation and treat Type 4 as a stretch goal, as it requires multiple contract exhibits from the same company with overlapping scope—a rarer occurrence in public filings.

---

## 5. Dataset & Data Strategy

### 5.1 Primary Data Source: SEC EDGAR
All document streams are aligned using the company's **CIK (Central Index Key)** as primary key:

```
EDGAR CIK
  ├── 10-K / 10-Q  →  Financial text (MD&A, Risk Factors, footnotes)
  ├── 8-K Exhibit 10.x  →  Material contracts
  ├── 8-K (Item 4.02, Item 2.06, etc.)  →  Event disclosures
  └── Company ticker  →  News articles (via news archives)
```

This eliminates the VDR access problem: 8-K Exhibit 10.x filings contain real, material legal contracts filed with the SEC. These are structurally analogous to contracts reviewed in PE DD, and are fully public.

### 5.2 Company Selection: Known-Inconsistency Cases

Unlike synthetic injection, we select companies with **naturally occurring, publicly documented contradictions** sourced from SEC enforcement actions, auditor resignations, and financial restatements. This provides ecologically valid ground truth without manual construction.

We select **5 companies** based on:
- Presence of a documented, post-hoc confirmed cross-document inconsistency (SEC enforcement order, restatement, or auditor action)
- Sufficient EDGAR filing history (10-K, 8-K exhibits, 10-Q) covering the inconsistency period
- Availability of contemporaneous news coverage

**Candidate companies and their known contradictions:**

| Company | CIK | Known Event | Contradiction Types |
|---------|-----|-------------|-------------------|
| **Super Micro Computer (SMCI)** | 1375365 | Hindenburg report → EY resignation → delayed 10-K → BDO adverse opinion (2024) | Type 2, Type 3 |
| **Under Armour (UAA)** | 1336917 | Revenue pull-forward, $9M SEC settlement (2021) | Type 1, Type 2 |
| **Revlon** | 887921 | Financial restatement + bankruptcy, material weakness disclosures (2020–2022) | Type 1, Type 3 |
| **Nikola Corp (NKLA)** | 1731289 | Hindenburg report → SEC fraud charges against founder (2020–2021) | Type 2 |
| **Plug Power (PLUG)** | 1093691 | Restatement of 3 years of financials due to accounting errors (2021) | Type 1, Type 3 |

For each company, we compile a **ground-truth contradiction checklist** from post-hoc regulatory findings, enforcement orders, and investigative reporting. This checklist serves as the evaluation target.

### 5.3 Ground Truth Construction Protocol

For each selected company:
1. Identify the inconsistency event from SEC enforcement orders or public investigations
2. Locate the specific filings (10-K sections, 8-K exhibits) that contain the contradictory claims
3. Record each contradiction as a tuple: `(DocA_section, DocB_section, contradiction_type, description)`
4. Cross-validate with at least two independent public sources (SEC order + news report)

**Target: 4–6 ground-truth contradictions per company, 20–30 total instances across 5 companies.**

---

## 6. System Architecture

### 6.1 Overview

```
┌──────────────────────────────────────────────────┐
│              Document Acquisition Layer            │
│  EDGAR Fetcher (10-K, 8-K, 10-Q) │ News Fetcher  │
└───────────────────┬──────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────┐
│         Structured Claim Extraction Layer          │
│                                                    │
│  ┌────────────────────┐  ┌─────────────────────┐ │
│  │  Contract Claim     │  │ Financial Claim      │ │
│  │  Extractor          │  │ Extractor            │ │
│  │                     │  │                      │ │
│  │  Input: 8-K Exh.10  │  │ Input: 10-K MD&A,   │ │
│  │  Output: structured │  │ Risk Factors, Notes  │ │
│  │  (party, obligation,│  │ Output: structured   │ │
│  │   value, condition) │  │ (metric, value,      │ │
│  │                     │  │  period, assertion)  │ │
│  └─────────┬──────────┘  └──────────┬──────────┘ │
│            │                        │              │
│  ┌─────────┴────────────────────────┴──────────┐ │
│  │         News Event Extractor                  │ │
│  │  Input: news articles                         │ │
│  │  Output: (entity, event_type, date, source)  │ │
│  └──────────────────────┬───────────────────────┘ │
└─────────────────────────┼────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────┐
│   Cross-Document Consistency Engine (XDCE)         │  ← Core
│                                                    │
│  Step 1: Entity Alignment                          │
│    CIK-based linking + LLM co-reference for       │
│    entity mentions across doc types                │
│                                                    │
│  Step 2: Claim Pairing                             │
│    For each entity, generate candidate claim       │
│    pairs (DocA_claim, DocB_claim) where both       │
│    reference the same subject (revenue, litigation,│
│    contract terms, etc.)                           │
│    Method: embedding similarity (top-k) +          │
│    subject-tag matching                            │
│                                                    │
│  Step 3: Pairwise Contradiction Classification     │
│    For each candidate pair, structured CoT prompt: │
│    "Given Claim A from [source] and Claim B from   │
│     [source], classify as: CONSISTENT /            │
│     CONTRADICTORY / INSUFFICIENT_INFO"             │
│    + extract reasoning chain                       │
│                                                    │
│  Step 4: Taxonomy Labeling & Confidence            │
│    Assign Type 1–3 label based on source types;   │
│    aggregate confidence from extraction +           │
│    classification steps                            │
└──────────────────────────┬───────────────────────┘
                           ↓
┌──────────────────────────────────────────────────┐
│               DD Risk Report Generator             │
│  Contradiction flags with:                         │
│  - Source attribution (exact doc section + quote)  │
│  - Taxonomy type label                             │
│  - Confidence score                                │
│  - Suggested follow-up actions                     │
└──────────────────────────────────────────────────┘
```

### 6.2 XDCE Method Detail

**Entity Alignment:** For companies in our evaluation set, entity alignment is simplified by CIK-based document grouping. Within documents, we use LLM-based co-reference resolution to link entity mentions (e.g., "the Company," subsidiary names, counterparty names) to canonical entities. We do not attempt to solve general-purpose cross-document co-reference—we scope to entities identifiable from SEC filing metadata.

**Claim Extraction Prompt (Contract):**
```
Given the following contract excerpt from [8-K Exhibit 10.x]:
{text}

Extract all verifiable claims as structured JSON:
[{
  "claim_text": "exact quote",
  "subject": "revenue | debt | obligation | IP | ...",
  "parties": ["Party A", "Party B"],
  "value": "$50M" or null,
  "condition": "if revenue exceeds..." or null,
  "temporal": "FY2023" or null
}]
```

**Pairwise Contradiction Prompt:**
```
You are analyzing two claims about the same company from different documents.

Claim A (from {source_A}): {claim_A}
Claim B (from {source_B}): {claim_B}

Step 1: Do these claims reference the same subject? If not, answer UNRELATED.
Step 2: Are the claims logically consistent, contradictory, or is there
        insufficient information to determine?
Step 3: If contradictory, explain the specific inconsistency.

Output: {label: CONSISTENT|CONTRADICTORY|UNRELATED|INSUFFICIENT_INFO,
         reasoning: "...", severity: HIGH|MEDIUM|LOW}
```

---

## 7. Evaluation Framework

### 7.1 Evaluation 1 — Contradiction Detection on Real Cases (Primary)

**Method:** Run XDCE pipeline over the 5 selected companies. Compare system output against the ground-truth contradiction checklist compiled from SEC enforcement orders and public investigations.

**Metrics:**
- **Recall@Type:** Per taxonomy type, what fraction of known contradictions did the system detect?
- **Precision:** Of all contradictions flagged by the system, what fraction correspond to real inconsistencies?
- **F1:** Harmonic mean of precision and recall
- **Miss analysis:** For each undetected contradiction, root-cause analysis (entity alignment failure? claim not extracted? classifier error?)

**Ground truth standard:** A system-flagged contradiction "matches" a ground-truth item if it identifies the same two source documents and the same semantic inconsistency, regardless of exact wording.

### 7.2 Evaluation 2 — Architecture Ablation (RQ2)

We compare three approaches on the same 5-company evaluation set:

| Condition | Method | Purpose |
|-----------|--------|---------|
| **Naive Long-Context** | Concatenate all documents into a single prompt; ask LLM to identify contradictions | Upper bound on what raw LLM capability achieves without structured decomposition |
| **Single-Doc Only** | Run extraction on each document independently; no cross-document comparison | Lower bound; demonstrates necessity of cross-document analysis |
| **XDCE Pipeline** | Full multi-agent pipeline (extraction → pairing → classification) | Our proposed method |

**Key research finding this enables:** If naive long-context matches XDCE performance, the multi-agent architecture adds engineering complexity without research value. If XDCE outperforms, we can analyze *why*—likely because structured claim extraction catches quantitative contradictions (Type 1) that LLMs miss in unstructured long context.

**Note on long-context feasibility:** A single company's filing set (10-K + 8-Ks + news) typically exceeds 200K tokens. We test long-context with Claude's 200K window, truncating to fit where necessary, and document what is dropped.

### 7.3 Evaluation 3 — Extraction Quality Spot-Check

To diagnose pipeline failures, we manually evaluate extraction quality on a sample:
- **Contract claim extraction:** For 20 randomly sampled contract excerpts, do extracted claims accurately capture the key obligations and values? (Human-judged accuracy)
- **Financial claim extraction:** For 10 key financial assertions in 10-K filings, does the extractor capture the correct metric, value, and period? (Compared against XBRL-tagged values where available)

This is a diagnostic evaluation, not a standalone contribution.

---

## 8. Technical Stack

| Component | Technology |
|-----------|------------|
| LLM | Claude Sonnet (claude-sonnet-4-20250514) via Anthropic API |
| Orchestration | Custom Python pipeline (no framework overhead) |
| Embedding / Retrieval | text-embedding-3-small + FAISS (for claim pairing) |
| Financial Data | SEC EDGAR REST API (EFTS full-text search + filing index) |
| News | Media archives via web search APIs; LexisNexis if available |
| Entity Linking | CIK-based primary key + spaCy NER for within-doc co-reference |
| Evaluation | Custom Python scripts with manual verification |

---

## 9. Proposed Timeline (6 Weeks)

| Week | Milestones |
|------|-----------|
| 1 | Company selection finalization; EDGAR data pipeline (10-K, 8-K fetcher + HTML parser); ground-truth contradiction checklist compilation for 5 companies |
| 2 | Claim extraction prompts for contracts, financials, and news; extraction quality spot-check; entity alignment implementation |
| 3 | XDCE core: claim pairing (embedding + subject matching) and pairwise contradiction classifier |
| 4 | End-to-end pipeline integration; run on all 5 companies; naive long-context baseline runs |
| 5 | Evaluation: compute metrics, miss analysis, architecture ablation comparison; iterate on failure cases |
| 6 | DD report generator; paper writeup; ablation analysis |

**Risk mitigation:** XDCE implementation begins Week 3 (not Week 4 as in prior plan), giving 3 weeks for the core contribution. Weeks 1–2 are parallelizable (data pipeline + ground truth compilation can proceed concurrently with prompt engineering).

---

## 10. Expected Contributions

1. **Cross-Document Contradiction Detection Method:** A multi-agent pipeline (structured claim extraction → entity-aligned pairing → pairwise NLI classification) that extends single-document contradiction detection to heterogeneous corporate document types. We provide detailed prompt designs and analyze which pipeline stages contribute most to detection accuracy.

2. **Contradiction Taxonomy with Real-World Grounding:** A 3-type taxonomy (Financial-Contract, Disclosure-News, Temporal) of cross-document contradictions in corporate filings, instantiated with real SEC enforcement cases rather than synthetic examples.

3. **Architecture Comparison:** Empirical comparison of structured multi-agent decomposition vs. naive long-context prompting for cross-document contradiction detection, identifying the conditions under which decomposition helps (e.g., quantitative Type 1 contradictions) vs. where it may be unnecessary.

4. **Failure Analysis:** Systematic categorization of detection failures by pipeline stage (extraction miss, alignment failure, classifier error), providing actionable insights for future work on cross-document reasoning.

---

## 11. Limitations & Risks

**Sample size:** 5 companies with 20–30 contradiction instances is small for statistical claims. We frame results as case-study evidence with detailed qualitative analysis, not population-level generalization. Future work can scale this approach to larger company sets.

**Ground truth subjectivity:** Some contradictions are matters of interpretation (e.g., whether a risk factor disclosure is "adequate" given external events). Mitigation: restrict ground truth to contradictions confirmed by SEC enforcement orders or formal restatements, not editorial judgment.

**Entity alignment scope:** We rely on CIK-based alignment and do not solve general cross-document co-reference. This is appropriate for our SEC-filing-centric evaluation but would need extension for arbitrary document sets.

**News data access:** Historical news coverage contemporaneous with filing dates may be incomplete through free APIs. Mitigation: supplement with SEC litigation releases, press releases, and short-seller reports (Hindenburg, Muddy Waters), which are permanently archived.

**LLM consistency:** Contradiction classification via prompting is sensitive to phrasing. Mitigation: use structured chain-of-thought prompts with explicit output schemas; report variance across 3 runs per company.

**Generalizability:** Results on public companies with SEC filings may not directly transfer to private PE targets using VDR documents. We frame the contribution as a method and evaluation approach, not a production system.

---

## References (Preliminary)

- Hendrycks et al. (2021). CUAD: An Expert-Annotated NLP Dataset for Legal Contracts. *NeurIPS*.
- Koreeda & Manning (2021). ContractNLI: A Dataset for Document-Level NLI for Contracts. *EMNLP Findings*.
- Islam et al. (2023). FinanceBench: A New Benchmark for Financial Question Answering. *arXiv:2311.11944*.
- LegalWiz (2025). A Multi-Agent Generation Framework for Contradiction Detection in Legal Documents. *NeurIPS Workshop*.
- CLAUSE (2025). Better Call CLAUSE: A Discrepancy Benchmark for Auditing LLMs Legal Reasoning Capabilities. *arXiv:2511.00340*.
- Fraunhofer (2024). Uncovering Inconsistencies and Contradictions in Financial Reports.
- Li et al. (2023). ContraDoc: Understanding Self-Contradictions in Documents with Large Language Models.
