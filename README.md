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
The `LLMGenerator` implements **separate few-shot example sets** for zero-shot and RAG generation contexts (aligned with the presence or absence of retrieved context). This forces Llama 3.1 8B to produce concise entity-only answers, eliminating verbosity-induced false negatives in the EM metric — a requirement empirically validated during earlier ablation versions.

### 3. Entropy Probe with Zero Context Overhead
The **Shannon entropy** used for pre-retrieval routing is computed via a dedicated minimal probe prompt (`Question: {q}\nAnswer:`) that is **categorically decoupled** from the full few-shot generation prompt. This design prevents the prefill cost explosion observed with `logits_all=True` in `llama-cpp-python` when long prompts are used — reducing probe latency by ~90% at the cost of a minor distributional shift (AUC ~0.68 vs ~0.705 ideal), a documented and justified engineering trade-off.

### 4. Deterministic Query Expansion (QE) Fallback
When the post-retrieval statistical signal (skewness, kurtosis, or dispersion) indicates that the retrieved document distribution is ambiguous, the system applies **rule-based keyword extraction** (stop-word removal + length-sorted top-3 terms) to construct an expanded query. FAISS is then re-queried with `top_k_expanded=20`. This approach is:
- Fully deterministic and reproducible.
- Zero additional LLM cost.
- Strictly superior to BM25 fallback (demonstrated empirically across ablation versions).

### 5. Entropy Caching
Each query's entropy value is computed once and cached for the entire ablation run. Since `temperature=0.0` guarantees deterministic outputs, this eliminates redundant forward passes across the 8-combination matrix (2 PRE × 4 POST), reducing total probe calls by a factor of 4.

### 6. Skewness Bias Monitoring
The framework tracks EM stratified by `s_pop` bucket (HIGH ≥ 1000, MEDIUM 100–999, LOW < 100), enabling analysis of **routing fairness** across entity popularity classes and verification of the s_pop ↔ entropy correlation (ρ_Spearman ≈ −0.33 on Llama 3.1 8B Q4\_K\_M).

---

## Architecture

```
Query
  │
  ▼  PRE-RETRIEVAL  ──────────────────────────────────────────────────
  │  Shannon Entropy probe (minimal prompt, logprobs, temperature=0)
  │
  ├── H(X) < τ_e ──→  ZERO-SHOT generation (few-shot prompt, no context)
  │
  └── H(X) ≥ τ_e ──→  FAISS Search (top-10, IVFFlat, normalized L2)
                          │
                          ▼  POST-RETRIEVAL ────────────────────────────
                          │  Distributional analysis on top-10 distances
                          │    · Skewness  γ₁  (Fisher-Pearson)
                          │    · Kurtosis  κ   (Fisher excess)
                          │    · Dispersion σ  (std of L2 distances)
                          │
                          ├── Good distribution ──→  RAG generation
                          │                          (few-shot + FAISS top-3 context)
                          │
                          └── Ambiguous distribution ──→  Query Expansion
                                                           FAISS re-query (top-20)
                                                           RAG generation
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

> **Note for Windows users**: If you encounter issues building `llama-cpp-python` from source on Windows, use the pre-built wheels available at [llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases) and select the correct CUDA version.

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
│   ├── cache/                          # Wikipedia API download cache
│   │   └── wikipedia_cache.json
│   └── processed/                      # Intermediate processed files
│       ├── popqa_vector_data.json      # Raw stratified sample (Step 0)
│       ├── popqa_vector_data_expanded.json  # Corpus with hard negatives (Step 1)
│       ├── expansion_stats.json        # Expansion statistics
│       └── contamination_audit.json   # Full contamination rejection log
│
├── db_indexes/                         # Generated by Step 2
│   ├── popqa_expanded.faiss            # FAISS IVFFlat vector index
│   ├── popqa_expanded_metadata.json    # Per-passage metadata (is_gold, s_pop, etc.)
│   └── validation_summary.md          # Recall@10, MRR@10, corpus coverage report
│
├── models/
│   └── Llama-3.1-8B-Instruct.Q4_K_M.gguf   # ← DOWNLOAD MANUALLY
│
├── results/
│   ├── statistical_tests/             # McNemar, Bootstrap CI outputs
│   ├── fig_em_by_bucket.png
│   ├── fig_roc_entropy.png
│   ├── fig_spop_entropy.png
│   ├── fig_pareto.png
│   └── ablation_summary.csv
│
├── scripts/
│   ├── p1c_expand_corpus.py           # Step 1 — Corpus expansion
│   ├── p2_rebuild_indexes.py          # Step 2 — FAISS index construction
│   └── p3_router_popqa.py             # Step 3 — Ablation study router
│
├── results_popqa_v13.csv              # Final ablation results
├── router_popqa_v13.log               # Full execution log
└── README.md
```

---

## Execution Pipeline (Step-by-Step)

> **Important**: The three steps must be executed **in strict order**. Each step depends on the output of the previous one.

---

### Step 1 — Corpus Expansion (`p1c_expand_corpus.py`)

Downloads hard negatives from Wikipedia for each query subject using two complementary API signals:
- `morelike:` search — semantically similar pages (topic-level hard negatives)
- `linkshere` property — pages that cite the subject (contextual hard negatives)

Applies anti-contamination filtering with boundary-aware regex before adding any passage to the corpus.

```bash
python scripts/p1c_expand_corpus.py
```

**Outputs:**
- `data/processed/popqa_vector_data_expanded.json` — expanded corpus (~80 distractors/query)
- `data/processed/contamination_audit.json` — rejected passages with rejection reasons
- `data/processed/expansion_stats.json` — statistics (unique passages, avg context, rejection rate)
- `data/cache/wikipedia_cache.json` — persistent cache for Wikipedia API calls

**Expected runtime:** ~60–120 minutes (Wikipedia API rate-limited at 0.10s/request)

---

### Step 2 — Index Construction (`p2_rebuild_indexes.py`)

Encodes all unique passages with `all-MiniLM-L6-v2` (SentenceTransformers) and builds a FAISS `IndexIVFFlat` index with L2-normalized embeddings.

```bash
python scripts/p2_rebuild_indexes.py
```

**Outputs:**
- `db_indexes/popqa_expanded.faiss` — vector index
- `db_indexes/popqa_expanded_metadata.json` — per-vector metadata
- `db_indexes/validation_summary.md` — empirical Recall@10, MRR@10

**FAISS configuration:**
| Parameter | Value | Rationale |
|---|---|---|
| `nlist` | 256 | ≈ √60,000 (corpus size) |
| `nprobe` | 10 | 3.9% cluster search, good recall/speed trade-off |
| Normalization | L2 = 1 | Enables exact cos\_sim equivalence |

**Expected runtime:** ~5–15 minutes (CPU encoding); ~1–2 minutes with GPU.

> After this step, verify `validation_summary.md`. Corpus coverage should be ≥ 90%.

---

### Step 3 — Ablation Study (`p3_router_popqa.py`)

Loads Llama 3.1 8B, runs 2 PRE × 4 POST = 8 routing configurations over all 1500 queries.

```bash
python scripts/p3_router_popqa.py
```

**Outputs:**
- `results_popqa_v13.csv` — full per-query results with all metrics
- `router_popqa_v13.log` — complete execution log with per-run statistics

**CSV columns:**

| Column | Description |
|---|---|
| `em` | Exact Match (normalized: lower, no punctuation, no articles) |
| `f1` | Token-overlap F1 (max over possible\_answers) |
| `entropy` | Shannon H(X) in bits — computed on minimal probe prompt |
| `adaptive_tau` | s\_pop-based adaptive threshold annotation (not used for routing) |
| `skewness` | Fisher-Pearson γ₁ on top-10 similarity scores |
| `kurtosis` | Fisher excess kurtosis κ on top-10 |
| `dispersion` | σ of L2 distances (semantic crowding detector) |
| `retrieval_calls` | 1 = single FAISS call, 2 = QE fallback triggered |
| `gold_in_top_k` | Whether the gold passage appeared in the retrieved top-k |
| `route_taken` | `zero_shot` / `vector` / `query_expansion_fallback` / `error` |

**Expected runtime:** ~8–12 hours (8 runs × 1500 queries on RTX 4050)

---

## DOs and DON'Ts

### ✅ DOs

- **Check RAM/VRAM before starting Step 3.** With `n_gpu_layers=26` and `logits_all=True`, the model occupies approximately 5.0–5.3 GB of VRAM. Ensure at least 6 GB is available.
- **Use the persistent Wikipedia cache.** If Step 1 is interrupted, re-running it will automatically resume from the cache (`wikipedia_cache.json`), skipping already-downloaded passages.
- **Read `validation_summary.md` after Step 2.** If Recall@10 is below 0.70, the corpus is too sparse for meaningful skewness analysis. Consider re-running Step 1 with more distractors.
- **Verify `popqa_vector_data_expanded.json` exists before Step 2.** The `preflight_check` in Step 3 will exit cleanly if any required file is missing.

### ❌ DON'Ts

- **Do NOT add few-shot examples to the entropy probe prompt.** The probe uses `Question: {q}\nAnswer:` intentionally. Adding the system message or few-shot examples increases prefill cost by ~180 tokens, causing up to +5000ms latency overhead per query on `llama-cpp-python` with `logits_all=True`. This was empirically confirmed during v6 experiments.

- **Do NOT recalibrate `entropy_threshold` across datasets without a new CV run.** The value `τ_e = 2.04` is calibrated specifically for Llama 3.1 8B Q4\_K\_M on PopQA via 5-fold cross-validation. Applying it to a different model or dataset constitutes cross-dataset leakage.

- **Do NOT re-enable the BM25 fallback.** The BM25 baseline produced EM = 0.26 versus FAISS EM = 0.38 (McNemar p ≈ 0.000, Δ = −0.117). On datasets with semantic hard negatives, BM25 is strictly inferior to the deterministic QE + FAISS fallback.

- **Do NOT shuffle or reorder queries between runs.** The entropy cache (`_entropy_cache`) uses `query_id` as key. Reordering does not affect correctness but does invalidate cross-run latency comparisons logged in the `.log` file.

- **Do NOT run Step 2 multiprocessing on Windows without the `if __name__ == "__main__":` guard.** The script already includes this guard, but any external modification that removes it will cause infinite subprocess spawning on Windows.

### Key Scientific Findings (from v5/v6)

1. **Shannon entropy is a valid predictor of LLM knowledge boundaries** (AUC = 0.705, well above the 0.500 random baseline).
2. **Few-shot prompting is a necessary implementation requirement** for Llama 3.1 8B — it improved EM by +1.1pp over zero-shot prompting on the same corpus.
3. **BM25 is significantly worse than FAISS** on hard-negative corpora (McNemar χ², p < 0.0001).
4. **s\_pop ↔ entropy correlation** (ρ = −0.33) is weaker than Mallen et al. report for GPT-3 class models (ρ ≈ −0.42), consistent with the calibration degradation expected from Q4\_K\_M quantization.
5. **The retrieval savings from entropy routing are real** (−23.8% retrieval calls, −19% context tokens) but are masked by `llama-cpp-python` prefill overhead on consumer hardware — a hardware implementation constraint, not a theoretical failure of the method.

---

## References

- Mallen A. et al. (2023). *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories*. ACL 2023. https://arxiv.org/abs/2212.10511
- Lewis P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
- Kadavath S. et al. (2022). *Language Models (Mostly) Know What They Know*. https://arxiv.org/abs/2207.05221
- Karpukhin V. et al. (2020). *Dense Passage Retrieval for Open-Domain QA*. EMNLP 2020. https://arxiv.org/abs/2004.04906
- Johnson J. et al. (2021). *Billion-scale similarity search with GPUs*. IEEE Trans. Big Data. https://arxiv.org/abs/1702.08734
- Holtzman A. et al. (2020). *The Curious Case of Neural Text Degeneration*. ICLR 2020. https://arxiv.org/abs/1904.09751
- Dror R. et al. (2018). *The Hitchhiker's Guide to Testing Statistical Significance in NLP*. ACL 2018.
- Efron B. & Hastie T. (2016). *Computer Age Statistical Inference*. Cambridge University Press.

---

## Citation

If you use this framework or any part of its code in your research, please cite this work upon publication. The ablation study is structured for **full methodological reproducibility**.

```bibtex
@misc{ragsemanticRouter2026,
  title   = {Semantic Routing for RAG: Training-Free Ablation Study on PopQA},
  year    = {2026},
  note    = {Ablation study using Llama 3.1 8B Instruct (Q4\_K\_M) on PopQA.
             Shannon entropy pre-routing and distributional skewness post-routing.}
}
```
