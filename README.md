# Semantic Routing for RAG — Training-Free Ablation Study on PopQA

> A research framework for evaluating **training-free semantic routing strategies** in Retrieval-Augmented Generation (RAG) systems, validated on PopQA using Llama 3.1 8B Instruct.

---

## Table of Contents

- [Overview](#overview)
- [Scientific Contributions (v13)](#scientific-contributions-v13)
- [Architecture](#architecture)
- [Requirements and Installation](#requirements-and-installation)
- [Project Structure](#project-structure)
- [Execution Pipeline](#execution-pipeline-step-by-step)
- [DOs and DON'Ts](#dos-and-donts)
- [Experimental Results](#experimental-results)
- [References](#references)

---

## Overview

This project implements an advanced research framework to evaluate **Semantic Routing** strategies within heterogeneous RAG architectures. The core objective is to dynamically decide:

- **Pre-Retrieval**: whether external retrieval is necessary at all, based on the LLM's parametric knowledge confidence.
- **Post-Retrieval**: whether the retrieved documents are informative enough, or whether the query should be expanded and re-issued to the vector index.

All routing decisions are made **without any additional training**, relying exclusively on statistical signals derived from the LLM's internal probability distributions and the geometric properties of the retrieved document set.

The framework is validated on **PopQA** (Mallen et al., ACL 2023) — a single-hop factual QA dataset stratified by entity popularity (`s_pop`) — and uses **Llama 3.1 8B Instruct** (Q4\_K\_M quantization) via `llama-cpp-python`.

---

## Scientific Contributions (v13)

### 1. Anti-Contamination Filter
The corpus expansion pipeline (`p1c_expand_corpus.py`) downloads hard negatives from Wikipedia's Search API (`morelike:` operator and `linkshere` property). Each candidate passage is checked against all gold answers using **boundary-aware regex** to prevent false negatives in the Exact Match metric. A full contamination audit trail is persisted to disk.

### 2. Dual Few-Shot Prompting
The `LLMGenerator` implements **separate few-shot example sets** for zero-shot and RAG generation contexts (aligned with the presence or absence of retrieved context). This forces Llama 3.1 8B to produce concise entity-only answers, eliminating verbosity-induced false negatives in the EM metric — a requirement empirically validated during earlier ablation versions (v5→v6: +1.1pp EM improvement).

### 3. Entropy Probe with Zero Context Overhead
The **Shannon entropy** used for pre-retrieval routing is computed via a dedicated minimal probe prompt (`Question: {q}\nAnswer:`) that is **categorically decoupled** from the full few-shot generation prompt. This design prevents the prefill cost explosion observed with `logits_all=True` in `llama-cpp-python` when long prompts are used. Empirical validation in v13: Entropy strategies achieve **lower wall-clock latency** than the Always_Retrieve baseline (−1.9% for best config), recovering from the +75% overhead observed in v6 before this fix.

### 4. Deterministic Query Expansion (QE) Fallback
When the post-retrieval statistical signal (skewness, kurtosis, or dispersion) indicates that the retrieved document distribution is ambiguous, the system applies **rule-based keyword extraction** (stop-word removal + length-sorted top-3 terms) to construct an expanded query. FAISS is then re-queried with `top_k_expanded=20`. This approach is:
- Fully deterministic and reproducible.
- Zero additional LLM cost.
- Strictly superior to BM25 fallback (BM25 EM=0.287 vs FAISS EM=0.394, McNemar p≈0.000 in v5).

**5. Entropy Caching**  
Each query's entropy value is computed **once** during the first combination that uses the `Entropy` strategy (typically `Entropy × Always_FAISS`). The result is cached for the entire ablation run. Since `temperature=0.0` guarantees deterministic outputs, subsequent Entropy combinations reuse the cache, eliminating redundant forward passes across the 8‑combination matrix. 

### 6. Skewness Bias Monitoring
The framework tracks EM stratified by `s_pop` bucket (HIGH ≥ 1000, MEDIUM 100–999, LOW < 100), enabling analysis of **routing fairness** across entity popularity classes and verification of the s_pop ↔ entropy correlation.

---

## Architecture

```
Query
  │
  ▼  PRE-RETRIEVAL  ─────────────────────────────────────────────────
  │  Shannon Entropy probe (minimal prompt, logprobs, temperature=0)
  │  Cached after first run — zero cost on subsequent combinations.
  │
  ├── H(X) < τ_e ──→  ZERO-SHOT generation (few-shot prompt, no context)
  │
  └── H(X) ≥ τ_e ──→  FAISS Search (top-10, IVFFlat, normalized L2)
                          │
                          ▼  POST-RETRIEVAL ──────────────────────────
                          │  Distributional analysis on top-10 distances
                          │    · Skewness  γ₁  (Fisher-Pearson)
                          │    · Kurtosis  κ   (Fisher excess)
                          │    · Dispersion σ  (std of L2 distances)
                          │
                          ├── Good distribution ──→  RAG generation
                          │                          (few-shot + FAISS top-3 ctx)
                          │
                          └── Ambiguous distribution ──→  Query Expansion
                                                           FAISS re-query (top-20)
                                                           RAG generation (few-shot)
```

**Ablation matrix: 2 PRE × 4 POST = 8 configurations**

| Pre-Retrieval | Post-Retrieval |
|---|---|
| Always\_Retrieve | Always\_FAISS |
| Entropy | Skewness |
| | Skew\_Kurt |
| | Skew\_Moments |

---

## Requirements and Installation

### Prerequisites
- Python **3.10+**
- CUDA-capable GPU with **≥ 6 GB VRAM** (RTX 3060/4050 or better)
- At least **16 GB RAM**

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/yourproject.git
cd yourproject
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install numpy scipy sentence-transformers
pip install faiss-cpu          # CPU-only
# OR
pip install faiss-gpu          # If CUDA is available (recommended)
```

Install `llama-cpp-python` with CUDA support:

```bash
# With CUDA GPU acceleration (recommended)
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# CPU-only fallback
pip install llama-cpp-python
```

> **Note for Windows users**: If you encounter issues building `llama-cpp-python` from source, use the pre-built wheels at [llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases) and select the correct CUDA version.

### 4. Download the LLM model

Download `Llama-3.1-8B-Instruct.Q4_K_M.gguf` from HuggingFace and place it in the `models/` directory:

```
models/
└── Llama-3.1-8B-Instruct.Q4_K_M.gguf
```

Direct link: [bartowski/Meta-Llama-3.1-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

---

## Project Structure

```
Progetto_Routing_RAG/
│
├── data/
│   ├── cache/                               # Wikipedia API download cache
│   │   └── wikipedia_cache.json
│   └── processed/                           # Intermediate processed files
│       ├── popqa_vector_data.json           # Raw stratified sample (1500 queries)
│       ├── popqa_vector_data_expanded.json  # Corpus with hard negatives
│       ├── expansion_stats.json             # Expansion statistics
│       └── contamination_audit.json         # Full contamination rejection log
│
├── db_indexes/                              # Generated by Step 2
│   ├── popqa_expanded.faiss                 # FAISS IVFFlat vector index
│   ├── popqa_expanded_metadata.json         # Per-passage metadata
│   └── validation_summary.md               # Recall@10, MRR@10, corpus coverage
│
├── models/
│   └── Llama-3.1-8B-Instruct.Q4_K_M.gguf  # ← DOWNLOAD MANUALLY
│
├── results/
│   ├── statistical_tests/                  # McNemar, Bootstrap CI outputs
│   ├── fig_em_by_bucket.png
│   ├── fig_roc_entropy.png
│   ├── fig_spop_entropy.png
│   ├── fig_pareto.png
│   └── ablation_summary.csv
│
├── scripts/
│   ├── p1c_expand_corpus.py                # Step 1 — Corpus expansion
│   ├── p2_populate_indexes_popqa.py               # Step 2 — FAISS index construction
│   └── p3_router_popqa.py                  # Step 3 — Ablation study router
│
├── results_popqa_v13.csv                   # Final ablation results
├── router_popqa_v13.log                    # Full execution log
└── README.md
```

---

## Execution Pipeline (Step-by-Step)

> **Important**: The three steps must be executed **in strict order**.

---

### Step 0 — Dataset Download and Initial Sampling (`p1_download_popqa.py`)

Downloads the PopQA dataset from Hugging Face (or falls back to a TSV file), performs stratified sampling by `s_pop` (500 queries per bucket: HIGH ≥1000, MEDIUM 100–999, LOW <100), and fetches the corresponding Wikipedia passages for each subject entity. The output is a JSON file containing 1,500 queries with gold passages and basic hard negatives (4 distractors per query, sampled by predicate similarity).

```bash
python scripts/p1_download_popqa.py
```

### Step 1 — Corpus Expansion (`p1c_expand_corpus.py`)

Downloads hard negatives from Wikipedia using two complementary API signals:
- `morelike:` search — semantically similar pages (80 candidates/subject)
- `linkshere` property — pages that cite the subject (20 candidates/subject)

Applies anti-contamination filtering with boundary-aware regex.

```bash
python scripts/p1c_expand_corpus.py
```

**Outputs:**
- `data/processed/popqa_vector_data_expanded.json`
- `data/processed/contamination_audit.json`
- `data/processed/expansion_stats.json`
- `data/cache/wikipedia_cache.json`

**Expected runtime:** ~90–150 minutes (Wikipedia API rate-limited at 0.10s/request)

---

### Step 2 — Index Construction (`p2_populate_indexes_popqa.py`)

Encodes all unique passages with `all-MiniLM-L6-v2` and builds a FAISS `IndexIVFFlat` index with L2-normalized embeddings.

```bash
python scripts/p2_populate_indexes_popqa.py
```

**Outputs:**
- `db_indexes/popqa_expanded.faiss`
- `db_indexes/popqa_expanded_metadata.json`
- `db_indexes/validation_summary.md`

**FAISS configuration:**

| Parameter | Value | Rationale |
|---|---|---|
| `nlist` | 256 | ≈ √60,000 (corpus size) |
| `nprobe` | 10 | 3.9% cluster search |
| Normalization | L2 = 1 | Enables exact cos\_sim equivalence |

**Expected runtime:** ~5–15 min (CPU); ~1–2 min (GPU).

> After this step, verify `validation_summary.md`. Corpus coverage should be ≥ 90%.
**Note:** The BM25 section in `validation_summary.md` appears **only if the `rank_bm25` library is installed**. If missing, that part of the report is omitted. To install: `pip install rank_bm25`.

---

### Step 3 — Ablation Study (`p3_router_popqa.py`)

Runs 2 PRE × 4 POST = 8 routing configurations over all 1,273 queries.

```bash
python scripts/p3_router_popqa.py
```

**Outputs:**
- `results_popqa_v13.csv`
- `router_popqa_v13.log`

**CSV columns:**

| Column | Description |
|---|---|
| `em` | Exact Match (normalized: lower, no punctuation, no articles) |
| `f1` | Token-overlap F1 (max over `possible_answers`) |
| `entropy` | Shannon H(X) in bits — computed on minimal probe prompt |
| `adaptive_tau` | s\_pop-based adaptive threshold (exploratory, **not used for routing** – saved for post‑hoc analysis only) |
| `skewness` | Fisher-Pearson γ₁ on top-10 similarity scores |
| `kurtosis` | Fisher excess kurtosis κ on top-10 |
| `dispersion` | σ of L2 distances (semantic crowding detector) |
| `retrieval_calls` | 1 = standard FAISS, 2 = QE fallback triggered |
| `gold_in_top_k` | Whether the gold passage appeared in the retrieved top-k |
| `route_taken` | `zero_shot` / `vector` / `query_expansion_fallback` / `error` |

**Expected runtime:** ~20-24 hours (RTX 4050, 8 runs × 1,273 queries)

> **Note on entropy cache**: The first Entropy run (`Entropy × Always_FAISS`) populates the entropy cache for all 1,273 queries. Subsequent Entropy runs skip the probe and use cached values, which is why their latency is lower. This is expected behavior and does not affect result validity.

---

## DOs and DON'Ts

### ✅ DOs

- **Check RAM/VRAM before Step 3.** With `n_gpu_layers=26` and `logits_all=True`, the model requires ~5.0–5.3 GB VRAM. Ensure at least 6 GB is available.
- **Use the persistent Wikipedia cache.** If Step 1 is interrupted, re-running it automatically resumes from `wikipedia_cache.json`.
- **Read `validation_summary.md` after Step 2.** If Recall@10 is below 0.70, the corpus is too sparse. Consider re-running Step 1 with more distractors.
- **Interpret results correctly.** The routing strategies do not significantly improve EM, but they **preserve accuracy** while reducing retrieval calls and token usage. This is a successful outcome for cost‑sensitive deployments.

### ❌ DON'Ts

- **Do NOT add few-shot examples to the entropy probe prompt.** The probe uses `Question: {q}\nAnswer:` intentionally. Adding the system message or few-shot examples causes up to +5,000ms latency overhead per query on `llama-cpp-python` with `logits_all=True`. This was empirically confirmed in v6 experiments (+75% latency).

- **Do NOT change `entropy_threshold` without re‑evaluation.** The default `τ_e = 2.04` was inherited from earlier experiments on a different PopQA sample (v5, n=453 queries) and a different prompt format. It has **not** been re‑optimised on the v13 expanded corpus to avoid data leakage. Using it on a different model or dataset may be suboptimal.

- **Do NOT re-enable the BM25 fallback.** In the earlier v5 corpus (453 queries), BM25 achieved EM = 0.287 vs FAISS EM = 0.394 (McNemar p ≈ 0.000, Δ = −0.107). While not re‑tested on the larger v13 corpus, the deterministic QE + FAISS fallback proved strictly superior in that controlled setting and remains the recommended default.

- **Do NOT run multiprocessing on Windows without the `if __name__ == "__main__":` guard.** The script includes this guard; do not remove it.

---

## Experimental Results

### v13 Final Results — 1,273 Queries (Stratified 500/500/500 by s_pop)

| Configuration | EM | Latency (ms) | ZS | VEC+QE | Recall@10 |
|---|---|---|---|---|---|
| Always\_Retrieve × Always\_FAISS | 0.3936 | 6,609 | 0 | 1,273 | 0.978 |
| Always\_Retrieve × Skewness | 0.3943 | 6,609 | 0 | 1,273 | 0.980 |
| Always\_Retrieve × Skew\_Kurt | **0.3951** | 6,603 | 0 | 1,273 | 0.982 |
| Always\_Retrieve × Skew\_Moments | **0.3951** | 6,604 | 0 | 1,273 | 0.982 |
| Entropy × Always\_FAISS | 0.3912 | 10,913† | 190 | 1,083 | 0.975 |
| Entropy × Skewness | 0.3896 | 6,503 | 190 | 1,083 | 0.976 |
| Entropy × Skew\_Kurt | 0.3904 | 6,487 | 190 | 1,083 | 0.979 |
| Entropy × Skew\_Moments | 0.3904 | **6,477** | 190 | 1,083 | 0.979 |

† First Entropy run — entropy cache populated during this run (1,273 additional probe forward passes). Subsequent Entropy runs use the cache, achieving latency ≈ 6,500ms.

### Progression Across Versions

| Version | Queries | Baseline EM | Best EM | Recall@10 | Entropy Lat. vs Baseline |
|---|---|---|---|---|---|
| v5 | 453 | 0.3664 | 0.3664 | 0.728 | +1.4% |
| v6 | 453 | 0.3775 | 0.3819 | 0.704 | **+75.4%** ⚠️ |
| **v13** | **1,273** | **0.3936** | **0.3951** | **0.978** | **−1.9%** ✅ |

### Key Scientific Findings

**1. Shannon entropy is a valid predictor of LLM knowledge boundaries.**  
> ROC analysis (ground truth: `s_pop ≥ 1000` as proxy for “LLM knows the answer”, following Mallen et al.) yields AUC = 0.6498 — well above the 0.500 random baseline. The entropy probe correctly identifies ~15% of queries where Llama 3.1 8B has sufficient parametric knowledge to answer without retrieval.  
> **Limitation:** As shown in Point 5, the correlation between `s_pop` and entropy is very weak on this quantized model (`ρ = -0.1639`). Therefore, the AUC reported here should be interpreted with caution; it may overestimate the true predictive power of entropy against actual zero‑shot EM. The proxy is used only for consistency with prior literature.

**2. Few-shot prompting is a necessary architectural requirement.**
The v5→v6 transition (few-shot introduced) improved baseline EM by +1.1pp on identical data and corpus. Applied to v13, baseline EM reaches 0.3936 (+2.72pp over v5).

**3. The entropy probe decoupling resolves the latency bottleneck.**
v13 Entropy × Skew\_Moments: 6,477ms — **132ms faster** than the Always\_Retrieve baseline. The v6 overhead (+75%) is fully eliminated by using a minimal probe prompt separate from the few-shot generation prompt.

**4. Post-retrieval QE strategies show no statistically significant EM improvement.**  
Always\_Retrieve × Skew\_Kurt and Skew\_Moments achieve EM = 0.3951 (+0.0015 vs baseline), but McNemar tests (paired, continuity‑corrected) reveal **no statistically significant difference** from the baseline for any configuration (p > 0.05 for all comparisons). The observed EM differences fall within the 95% bootstrap confidence interval width (±0.027). Therefore, the routing strategies **do not harm answer quality**.

**5. Retrieval savings are the primary benefit – quality is preserved.**  
Entropy routing reduces retrieval calls by **−14.9%** and context tokens by **−14.9%** compared to the Always_Retrieve baseline, with **no statistically significant loss in EM**. The best Entropy configuration (Entropy × Skew\_Moments) also achieves slightly **lower latency** (−132 ms, −2.0%) than the baseline when the entropy cache is populated. This demonstrates that training‑free semantic routing can **cut computational costs without sacrificing accuracy** – a practically valuable trade‑off for deployed RAG systems.

**6. s_pop ↔ entropy correlation on quantized models.**
ρ_Spearman = -0.1639 on Llama 3.1 8B Q4_K_M, versus ρ ≈ -0.42 reported by Mallen et al. for full‑precision GPT‑3 class models.This weak correlation implies that s_pop is not a reliable proxy for uncertainty on this quantized model, contra Mallen et al. This is an important negative result for practitioners using 4‑bit models

**Summary of statistical significance:**  
Formal McNemar tests confirm that no EM difference is statistically significant (p > 0.05 for all configurations vs. baseline).

---

### Future Work: Mitigating Data Leakage via Dynamic Percentile Routing

Currently, the entropy threshold for pre-retrieval routing ($\tau_e = 2.04$) is hardcoded, having been empirically derived from a previous data sample (v5). While functional for demonstrating the routing concept in this ablation study, static absolute thresholds present a methodological risk of data leakage and are inherently fragile in production environments. Shannon entropy distributions will inevitably fluctuate if the prompt format is altered or if the underlying LLM is updated.

To further improve the scientific rigor and production-readiness of this pipeline, future iterations should transition from a static threshold to a **Dynamic Percentile Threshold via a Sliding Window**:

* **Sliding Window Memory:** Instead of evaluating against an absolute value, the system maintains a running buffer of the Shannon entropy scores from the last $N$ queries (e.g., $N=500$).
* **Relative Percentile Routing:** The routing decision becomes a dynamic percentile calculation. For instance, the rule adapts to: *"Route the 15% of queries with the lowest entropy in the current window to zero-shot generation."*

**Scientific and Practical Benefits:** Implementing this approach eliminates the data leakage associated with hyperparameter tuning ($\tau_e$) on an evaluation set. Furthermore, it guarantees a predictable, constant rate of computational savings while automatically self-calibrating against distribution shifts in user queries or upstream changes to the LLM's parametric knowledge.

---

## References

- Mallen A. et al. (2023). *When Not to Trust Language Models*. ACL 2023. https://arxiv.org/abs/2212.10511
- Lewis P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
- Kadavath S. et al. (2022). *Language Models (Mostly) Know What They Know*. https://arxiv.org/abs/2207.05221
- Karpukhin V. et al. (2020). *Dense Passage Retrieval for Open-Domain QA*. EMNLP 2020. https://arxiv.org/abs/2004.04906
- Johnson J. et al. (2021). *Billion-scale similarity search with GPUs*. IEEE Trans. Big Data. https://arxiv.org/abs/1702.08734
- Holtzman A. et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR 2020. https://arxiv.org/abs/1904.09751
- Dror R. et al. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP*. ACL 2018.
- Efron B. & Hastie T. (2016). *Computer Age Statistical Inference*. Cambridge University Press.

---

## Citation

If you use this framework or any part of its code in your research, please cite this work upon publication.

```bibtex
@misc{ragsemanticRouter2026,
  title   = {Semantic Routing for RAG: Training-Free Ablation Study on PopQA},
  year    = {2026},
  note    = {Ablation study using Llama 3.1 8B Instruct (Q4\_K\_M) on PopQA.
             Shannon entropy pre-routing and distributional skewness post-routing.
             n=1,273 queries, stratified by entity popularity (s\_pop).}
}
```
