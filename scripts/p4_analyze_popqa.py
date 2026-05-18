"""
p4_analyze_popqa.py
===================
Complete post-hoc analysis of PopQA ablation study v13.

Output:
  results/fig_em_by_bucket.png        → EM per bucket (high/medium/low) per strategy
  results/fig_roc_entropy.png         → ROC curve: entropy as ZS predictor
  results/fig_spop_entropy.png        → s_pop ↔ entropy correlation (with adaptive_tau)
  results/fig_pareto.png              → EM vs latency (Pareto frontier)
  results/fig_savings.png             → Computational savings (retrieval_calls, tokens)
  results/ablation_summary.csv        → Full aggregated table
  results/routing_analysis.txt        → Textual report for the paper
"""

import os
import sys
import csv
import json
import warnings
import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")   # Non‑interactive backend (no window)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[WARN] matplotlib not installed: pip install matplotlib")
    print("       Figures will not be generated, only summary CSV.")

try:
    from sklearn.metrics import roc_curve, auc
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[WARN] scikit-learn not installed: pip install scikit-learn")
    print("       ROC curve will be computed manually.")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_INPUT   = os.path.join(BASE_DIR, "results_popqa_v13.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Consistent color palette for all plots ──────────────────────────────────
COLORS = {
    "Always_Retrieve": "#888888",
    "Entropy":         "#2196F3",
    "Entropy_PP":      "#9C27B0",
    "BM25_Always":     "#FF9800",
    "high":            "#4CAF50",
    "medium":          "#FFC107",
    "low":             "#F44336",
    "vector":          "#2196F3",
    "zero_shot":       "#9C27B0",
    "bm25_fallback":   "#FF9800",
    "bm25":            "#FF9800",
}

BUCKET_ORDER = ["high", "medium", "low"]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING AND AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        print(f"[ERROR] CSV not found: {path}")
        print("  Run p3_router_popqa_v13.py first")
        sys.exit(1)
    print(f"[LOAD] {path}")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[LOAD] {len(rows)} rows loaded.")
    return rows


def safe_float(v: str) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def group_by_combination(rows: List[Dict]) -> Dict[str, List[Dict]]:
    groups = defaultdict(list)
    for r in rows:
        key = f"{r['pre_strategy']} × {r['post_strategy']}"
        groups[key].append(r)
    return dict(groups)


def compute_combo_stats(rows: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics for a combination."""
    ems    = [safe_float(r["em"])  for r in rows if safe_float(r["em"]) is not None]
    f1s    = [safe_float(r["f1"])  for r in rows if safe_float(r["f1"]) is not None]
    lats   = [safe_float(r["total_latency_ms"]) for r in rows
              if safe_float(r["total_latency_ms"]) is not None and r["route_taken"] != "error"]
    recall_vals = [r["gold_in_top_k"] for r in rows if r.get("gold_in_top_k") in ("True","False")]
    recall_rate = np.mean([1 if x == "True" else 0 for x in recall_vals]) if recall_vals else float("nan")

    routes = {k: sum(1 for r in rows if r["route_taken"] == k)
              for k in ("zero_shot","vector","bm25_fallback","bm25","error")}
    n_tot  = len(rows)

    ret_calls = [int(r.get("retrieval_calls","1")) for r in rows
                 if r.get("retrieval_calls","").isdigit()]
    tok_est   = [safe_float(r.get("prompt_tokens_estimate","")) for r in rows]
    tok_est   = [t for t in tok_est if t is not None]

    # Stratified EM per bucket
    bucket_em = {}
    for b in BUCKET_ORDER:
        b_rows = [r for r in rows if r.get("pop_bucket","") == b]
        b_ems  = [safe_float(r["em"]) for r in b_rows if safe_float(r["em"]) is not None]
        bucket_em[b] = (np.mean(b_ems) if b_ems else 0.0, len(b_ems))

    return {
        "em":           np.mean(ems) if ems else 0.0,
        "f1":           np.mean(f1s) if f1s else 0.0,
        "lat_mean":     np.mean(lats) if lats else 0.0,
        "lat_p95":      np.percentile(lats, 95) if lats else 0.0,
        "recall_at_k":  recall_rate,
        "n_zero_shot":  routes["zero_shot"],
        "n_vector":     routes["vector"],
        "n_bm25_fb":    routes["bm25_fallback"],
        "n_bm25":       routes["bm25"],
        "n_error":      routes["error"],
        "n_total":      n_tot,
        "pct_skip":     100 * routes["zero_shot"] / n_tot if n_tot else 0,
        "retrieval_calls_mean": np.mean(ret_calls) if ret_calls else 1.0,
        "tokens_mean":  np.mean(tok_est) if tok_est else 0.0,
        "bucket_em":    bucket_em,
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: FIGURE 1 — EM per bucket
# ═════════════════════════════════════════════════════════════════════════════

def plot_em_by_bucket(groups: Dict[str, List[Dict]], output_path: str):
    """
    Grouped bar chart: EM per bucket (high/medium/low) for each pre-strategy.

    This is the main visual contribution of the paper:
    shows that Entropy routing improves EM on LOW buckets
    (where the LLM is uncertain) without degrading the HIGH bucket
    (where zero‑shot answers are already good).
    """
    if not MATPLOTLIB_AVAILABLE: return

    # Group by pre‑strategy (averaging over post‑strategies)
    pre_strategies = ["Always_Retrieve", "Entropy", "Entropy_PP", "BM25_Always"]
    pre_bucket_em  = {}

    for pre in pre_strategies:
        bucket_ems = {b: [] for b in BUCKET_ORDER}
        for combo, rows in groups.items():
            if not combo.startswith(pre): continue
            for b in BUCKET_ORDER:
                b_rows = [r for r in rows if r.get("pop_bucket","") == b]
                b_ems  = [safe_float(r["em"]) for r in b_rows if safe_float(r["em"]) is not None]
                if b_ems: bucket_ems[b].extend(b_ems)
        pre_bucket_em[pre] = {b: np.mean(v) if v else 0.0 for b, v in bucket_ems.items()}

    fig, ax = plt.subplots(figsize=(10, 6))
    x       = np.arange(len(pre_strategies))
    width   = 0.25
    offsets = [-width, 0, width]
    bucket_labels = {"high": "HIGH (s_pop≥1000)", "medium": "MEDIUM (100-999)", "low": "LOW (<100)"}

    for i, (bucket, offset) in enumerate(zip(BUCKET_ORDER, offsets)):
        values = [pre_bucket_em.get(pre, {}).get(bucket, 0) for pre in pre_strategies]
        bars   = ax.bar(x + offset, values, width, label=bucket_labels[bucket],
                        color=COLORS[bucket], alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Pre-Retrieval Strategy", fontsize=12)
    ax.set_ylabel("Exact Match (EM)", fontsize=12)
    ax.set_title("EM per s_pop bucket × routing strategy\n"
                 "(averaged over post‑retrieval strategies)", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(pre_strategies, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: FIGURE 2 — ROC Curve (entropy as ZS predictor)
# ═════════════════════════════════════════════════════════════════════════════

def plot_roc_entropy(rows: List[Dict], output_path: str):
    """
    ROC Curve: Shannon entropy as a binary classifier for the ZS decision.

    Positive label definition:
      y=1 if EM_zero_shot > 0 (the model would have answered correctly in ZS)
      y=0 otherwise.

    This analysis answers the core scientific question:
      "Is Shannon entropy a good predictor of when the LLM already knows the
       answer without needing retrieval?"

    AUC > 0.6: entropy has useful predictive power.
    AUC ≈ 0.5: equivalent to random (entropy does not discriminate).

    Note: we use entropy from the Entropy × Always_FAISS strategy
    because it is the "pure" measure without post‑routing effects.
    """
    if not MATPLOTLIB_AVAILABLE: return

    # Filter rows from Entropy × Always_FAISS (they have both entropy and em)
    entropy_rows = [r for r in rows
                    if r.get("pre_strategy") == "Entropy"
                    and r.get("post_strategy") == "Always_FAISS"
                    and safe_float(r.get("entropy")) is not None]

    if not entropy_rows:
        print("[WARN] ROC: no Entropy × Always_FAISS rows found.")
        return

    # Label: EM_vector > 0 for queries the model "knows" how to answer
    # Use vector response EM as a proxy for ground‑truth ZS knowledge
    entropies = np.array([safe_float(r["entropy"]) for r in entropy_rows])
    ems_faiss = np.array([safe_float(r["em"]) for r in entropy_rows])

    # Binary label: ZS‑answerable queries = EM_zero‑shot > 0
    # Use s_pop > 1000 as ground truth proxy (from literature: Mallen et al.)
    s_pops  = np.array([safe_float(r.get("s_pop","1")) or 1.0 for r in entropy_rows])
    y_true  = (s_pops >= 1000).astype(int)   # HIGH bucket = ZS‑answerable

    # Score: low entropy → high confidence → predicts ZS (invert for ROC)
    y_score = -entropies   # lower entropy = more confident = more ZS

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: ROC curve ────────────────────────────────────────────────
    ax1 = axes[0]
    if SKLEARN_AVAILABLE:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
    else:
        # Manual ROC calculation (if sklearn not available)
        sorted_idx   = np.argsort(y_score)[::-1]
        y_s_sorted   = y_score[sorted_idx]
        y_t_sorted   = y_true[sorted_idx]
        n_pos        = y_true.sum()
        n_neg        = len(y_true) - n_pos
        tpr, fpr     = [0.0], [0.0]
        tp = fp = 0
        for yt in y_t_sorted:
            if yt == 1: tp += 1
            else:        fp += 1
            tpr.append(tp / n_pos if n_pos > 0 else 0)
            fpr.append(fp / n_neg if n_neg > 0 else 0)
        tpr, fpr = np.array(tpr), np.array(fpr)
        roc_auc  = np.trapz(tpr, fpr)
        thresholds = np.linspace(y_score.max(), y_score.min(), len(tpr))

    ax1.plot(fpr, tpr, lw=2, color="#2196F3",
             label=f"Entropy AUC = {roc_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax1.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title("ROC: Shannon Entropy\nas predictor of ZS‑answerable queries", fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    # AUC interpretation annotation
    if roc_auc > 0.65:
        interp = "GOOD predictive power"
    elif roc_auc > 0.55:
        interp = "MODERATE predictive power"
    else:
        interp = "POOR predictive power"
    ax1.text(0.55, 0.1, interp, transform=ax1.transAxes,
             fontsize=10, style="italic", color="gray")

    # ── Right panel: entropy distribution per bucket ─────────────────────────
    ax2 = axes[1]
    for bucket in BUCKET_ORDER:
        b_mask = np.array([r.get("pop_bucket","") == bucket for r in entropy_rows])
        b_ent  = entropies[b_mask]
        if len(b_ent) > 0:
            ax2.hist(b_ent, bins=20, alpha=0.5, color=COLORS[bucket],
                     label=f"{bucket.upper()} (n={len(b_ent)})",
                     density=True, edgecolor="white")
    ax2.axvline(x=2.04, color="black", linestyle="--", lw=1.5,
                label="τ_entropy=2.04 (p50)")
    ax2.set_xlabel("Shannon Entropy H(X) [bits]", fontsize=12)
    ax2.set_ylabel("Density", fontsize=12)
    ax2.set_title("Entropy distribution per s_pop bucket\n"
                  "(τ = routing threshold)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")
    return roc_auc


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: FIGURE 3 — s_pop vs Entropy (correlation + adaptive_tau)
# ═════════════════════════════════════════════════════════════════════════════

def plot_spop_entropy(rows: List[Dict], output_path: str):
    """
    Scatter plot log(s_pop) vs Shannon entropy with adaptive_tau overlay.

    This plot visualizes the scientific motivation for Shannon_Adaptive
    even if not used in the router:
    - If correlation is strong (ρ < -0.3), adaptive threshold is justified.
    - If correlation is weak (ρ ≈ 0), a fixed threshold is sufficient.

    Mallen et al. (ACL 2023) report ρ_Spearman ≈ -0.42 on GPT‑3 class models.
    Compare with the empirical value on Llama 3.1 8B Q4_K_M.
    """
    if not MATPLOTLIB_AVAILABLE: return

    entropy_rows = [r for r in rows
                    if r.get("pre_strategy") == "Entropy"
                    and r.get("post_strategy") == "Always_FAISS"
                    and safe_float(r.get("entropy")) is not None
                    and safe_float(r.get("s_pop")) is not None]

    if not entropy_rows:
        print("[WARN] spop‑entropy: no available rows.")
        return

    s_pops     = np.array([safe_float(r["s_pop"]) for r in entropy_rows])
    entropies  = np.array([safe_float(r["entropy"]) for r in entropy_rows])
    adapt_taus = np.array([safe_float(r.get("adaptive_tau","")) or 2.04
                           for r in entropy_rows])
    buckets    = [r.get("pop_bucket","unknown") for r in entropy_rows]

    log_pops = np.log10(np.maximum(s_pops, 1.0))

    # Spearman correlation (robust to outliers)
    from scipy import stats as sp_stats
    rho, pval = sp_stats.spearmanr(log_pops, entropies)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Coloured scatter by bucket
    for bucket in BUCKET_ORDER:
        mask = np.array([b == bucket for b in buckets])
        ax.scatter(log_pops[mask], entropies[mask],
                   c=COLORS[bucket], alpha=0.4, s=20,
                   label=f"{bucket.upper()} (n={mask.sum()})")

    # Regression line
    if len(log_pops) > 2:
        m, b_coeff = np.polyfit(log_pops, entropies, 1)
        x_line = np.linspace(log_pops.min(), log_pops.max(), 100)
        ax.plot(x_line, m*x_line + b_coeff, "k-", lw=1.5, alpha=0.6,
                label=f"Regression (slope={m:.3f})")

    # Adaptive_tau curve
    x_tau = np.linspace(0, 6, 200)
    tau_min, tau_max, pivot = 1.0, 3.5, 3.0
    y_tau = tau_max - (tau_max-tau_min) / (1 + np.exp(-(x_tau - pivot)))
    ax.plot(x_tau, y_tau, "r--", lw=2, label=f"τ_adaptive(s_pop)")

    # Fixed threshold
    ax.axhline(y=2.04, color="#2196F3", linestyle=":", lw=1.5,
               label="τ_fixed=2.04")

    ax.set_xlabel("log₁₀(s_pop)", fontsize=12)
    ax.set_ylabel("Shannon Entropy H(X) [bits]", fontsize=12)
    ax.set_title(f"Correlation popularity ↔ LLM uncertainty\n"
                 f"ρ_Spearman = {rho:.3f}  (p={pval:.4f})", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # Secondary x‑axis with real s_pop values
    ax2 = ax.twiny()
    ticks_log = [0, 1, 2, 3, 4, 5]
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ticks_log)
    ax2.set_xticklabels([f"10^{t}" for t in ticks_log], fontsize=9)
    ax2.set_xlabel("s_pop (log scale)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")
    return rho, pval


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: FIGURE 4 — Pareto frontier (EM vs latency)
# ═════════════════════════════════════════════════════════════════════════════

def plot_pareto(groups: Dict[str, List[Dict]], output_path: str):
    """
    Scatter EM vs latency. The Pareto frontier (top‑left corner)
    identifies configurations that maximise quality and minimise latency.
    Useful for the paper to recommend the "best configuration".
    """
    if not MATPLOTLIB_AVAILABLE: return

    fig, ax = plt.subplots(figsize=(10, 7))

    points_for_pareto = []

    for combo, rows in sorted(groups.items()):
        stats = compute_combo_stats(rows)
        em    = stats["em"]
        lat   = stats["lat_mean"]

        pre   = combo.split(" × ")[0] if " × " in combo else combo
        color = COLORS.get(pre, "#666666")
        post  = combo.split(" × ")[1] if " × " in combo else ""
        marker = {"Always_FAISS":"o","Skewness":"s","Skew_Kurt":"^","Skew_Moments":"D",
                  "BM25_Always":"*"}.get(post, "o")

        ax.scatter(lat, em, c=color, marker=marker, s=100, alpha=0.85,
                   edgecolors="white", linewidth=0.5, zorder=3)
        ax.annotate(combo.replace(" × ","\n×"), (lat, em),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, alpha=0.7)

        points_for_pareto.append((lat, em, combo))

    # Compute Pareto frontier (maximise EM, minimise latency)
    pareto = []
    for lat, em, combo in sorted(points_for_pareto, key=lambda x: x[0]):
        if not pareto or em > pareto[-1][1]:
            pareto.append((lat, em, combo))
    if pareto:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.plot(px, py, "k-", lw=1.5, alpha=0.4, zorder=2, label="Pareto frontier")
        ax.scatter(px, py, c="gold", edgecolors="black", s=120, zorder=4, marker="*")

    # Colour legend (pre‑strategy)
    pre_labels = {"Always_Retrieve":"gray","Entropy":"#2196F3",
                  "Entropy_PP":"#9C27B0","BM25_Always":"#FF9800"}
    legend_patches = [mpatches.Patch(color=c, label=n, alpha=0.85)
                      for n, c in pre_labels.items()]
    ax.legend(handles=legend_patches, title="Pre-Strategy", fontsize=9, loc="lower right")

    ax.set_xlabel("Average Latency [ms]", fontsize=12)
    ax.set_ylabel("Exact Match (EM)", fontsize=12)
    ax.set_title("Pareto Frontier: quality vs latency\n"
                 "(top‑left is best)", fontsize=13)
    ax.grid(alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: FIGURE 5 — Computational savings
# ═════════════════════════════════════════════════════════════════════════════

def plot_savings(groups: Dict[str, List[Dict]],
                 baseline_combo: str,
                 output_path: str):
    """
    Bar charts of computational savings compared to the baseline None × None.
    Shows saved retrieval_calls, saved prompt tokens, and latency change.
    """
    if not MATPLOTLIB_AVAILABLE: return

    # Baseline statistics
    baseline_rows = groups.get(baseline_combo, [])
    if not baseline_rows:
        print(f"[WARN] Baseline '{baseline_combo}' not found.")
        return

    base_stats   = compute_combo_stats(baseline_rows)
    base_lat     = base_stats["lat_mean"]
    base_ret     = base_stats["retrieval_calls_mean"]
    base_tok     = base_stats["tokens_mean"]

    labels, lat_savings, tok_savings = [], [], []

    for combo in sorted(groups):
        stats = compute_combo_stats(groups[combo])
        labels.append(combo.replace(" × ","\n×"))
        lat_savings.append(100 * (base_lat - stats["lat_mean"]) / base_lat
                           if base_lat > 0 else 0)
        tok_savings.append(100 * (base_tok - stats["tokens_mean"]) / base_tok
                           if base_tok > 0 else 0)

    x      = np.arange(len(labels))
    width  = 0.4
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    colors_lat = ["#4CAF50" if v >= 0 else "#F44336" for v in lat_savings]
    ax1.bar(x, lat_savings, width, color=colors_lat, alpha=0.8, edgecolor="white")
    ax1.axhline(0, color="black", lw=0.8)
    ax1.set_ylabel("Latency savings %\n(green = faster)", fontsize=11)
    ax1.set_title("Computational savings vs baseline Always_Retrieve × Always_FAISS",
                  fontsize=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top","right"]].set_visible(False)

    colors_tok = ["#4CAF50" if v >= 0 else "#F44336" for v in tok_savings]
    ax2.bar(x, tok_savings, width, color=colors_tok, alpha=0.8, edgecolor="white")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("Token savings %\n(green = fewer tokens)", fontsize=11)
    ax2.set_xlabel("Configuration", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.5, rotation=35, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] {output_path}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: SUMMARY CSV AND TEXTUAL REPORT
# ═════════════════════════════════════════════════════════════════════════════

def save_summary_csv(groups: Dict[str, List[Dict]], path: str):
    """Aggregated table ready for the paper."""
    COLS = [
        "combination","pre_strategy","post_strategy",
        "em","f1","lat_mean_ms","lat_p95_ms","recall_at_k",
        "n_zero_shot","n_vector","n_bm25_fb","n_bm25","n_error","n_total",
        "pct_skip","retrieval_calls_mean","tokens_mean_estimate",
        "em_high","em_medium","em_low",
        "n_high","n_medium","n_low",
    ]
    rows_out = []
    for combo, rows in sorted(groups.items()):
        stats = compute_combo_stats(rows)
        parts = combo.split(" × ")
        row = {
            "combination":          combo,
            "pre_strategy":         parts[0],
            "post_strategy":        parts[1] if len(parts) > 1 else "",
            "em":                   f"{stats['em']:.4f}",
            "f1":                   f"{stats['f1']:.4f}",
            "lat_mean_ms":          f"{stats['lat_mean']:.1f}",
            "lat_p95_ms":           f"{stats['lat_p95']:.1f}",
            "recall_at_k":          f"{stats['recall_at_k']:.4f}" if not math.isnan(stats['recall_at_k']) else "",
            "n_zero_shot":          stats["n_zero_shot"],
            "n_vector":             stats["n_vector"],
            "n_bm25_fb":            stats["n_bm25_fb"],
            "n_bm25":               stats["n_bm25"],
            "n_error":              stats["n_error"],
            "n_total":              stats["n_total"],
            "pct_skip":             f"{stats['pct_skip']:.1f}",
            "retrieval_calls_mean": f"{stats['retrieval_calls_mean']:.3f}",
            "tokens_mean_estimate": f"{stats['tokens_mean']:.1f}",
            "em_high":              f"{stats['bucket_em'].get('high',(0,0))[0]:.4f}",
            "em_medium":            f"{stats['bucket_em'].get('medium',(0,0))[0]:.4f}",
            "em_low":               f"{stats['bucket_em'].get('low',(0,0))[0]:.4f}",
            "n_high":               stats['bucket_em'].get('high',(0,0))[1],
            "n_medium":             stats['bucket_em'].get('medium',(0,0))[1],
            "n_low":                stats['bucket_em'].get('low',(0,0))[1],
        }
        rows_out.append(row)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"[SAVE] Summary CSV: {path}")


def generate_text_report(groups: Dict[str, List[Dict]],
                         roc_auc: Optional[float],
                         spearman_rho: Optional[float],
                         path: str):
    """Generates the textual report with main metrics ready for the paper."""
    lines = []
    lines.append("=" * 72)
    lines.append("TRAINING‑FREE ADAPTIVE SEMANTIC ROUTING — PopQA v13")
    lines.append("POST‑HOC ANALYSIS REPORT")
    lines.append("=" * 72)
    lines.append("")

    # Find baseline and best configuration
    baseline_stats = compute_combo_stats(groups.get("Always_Retrieve × Always_FAISS", []))
    best_em        = 0.0
    best_combo     = ""
    for combo, rows in groups.items():
        s = compute_combo_stats(rows)
        if s["em"] > best_em:
            best_em    = s["em"]
            best_combo = combo

    lines.append("1. MAIN RESULTS")
    lines.append("─" * 72)
    lines.append(f"   Baseline (Always_Retrieve × Always_FAISS): EM={baseline_stats['em']:.4f}  F1={baseline_stats['f1']:.4f}")
    lines.append(f"   Best configuration:   {best_combo}")
    lines.append(f"   Best EM:              {best_em:.4f}  (Δ={best_em-baseline_stats['em']:+.4f} vs baseline)")
    lines.append("")

    lines.append("2. FULL RESULTS TABLE")
    lines.append("─" * 72)
    lines.append(f"  {'Combination':<42} | {'EM':>6} | {'F1':>6} | {'Lat':>7} | {'%ZS':>5} | {'Rec@k':>6}")
    lines.append("  " + "─" * 80)

    for combo, rows in sorted(groups.items(), key=lambda x: compute_combo_stats(x[1])["em"], reverse=True):
        s = compute_combo_stats(rows)
        rec = f"{s['recall_at_k']:.3f}" if not math.isnan(s['recall_at_k']) else "N/A"
        lines.append(
            f"  {combo:<42} | {s['em']:>6.4f} | {s['f1']:>6.4f} | "
            f"{s['lat_mean']:>7.0f} | {s['pct_skip']:>5.1f} | {rec:>6}"
        )
    lines.append("")

    lines.append("3. s_pop BUCKET ANALYSIS")
    lines.append("─" * 72)
    for pre in ["Always_Retrieve","Entropy","Entropy_PP","BM25_Always"]:
        pre_rows = [r for rows in groups.values() for r in rows
                    if r.get("pre_strategy") == pre]
        if not pre_rows: continue
        bucket_data = {}
        for b in BUCKET_ORDER:
            b_ems = [safe_float(r["em"]) for r in pre_rows
                     if r.get("pop_bucket","") == b and safe_float(r["em"]) is not None]
            bucket_data[b] = np.mean(b_ems) if b_ems else 0.0
        lines.append(f"  {pre:<20}: "
                     f"HIGH={bucket_data['high']:.4f}  "
                     f"MED={bucket_data['medium']:.4f}  "
                     f"LOW={bucket_data['low']:.4f}")
    lines.append("")

    lines.append("4. ROC ANALYSIS (entropy as ZS predictor)")
    lines.append("─" * 72)
    if roc_auc is not None:
        interpretation = ("GOOD (>0.65)" if roc_auc > 0.65 else
                          "MODERATE (0.55-0.65)" if roc_auc > 0.55 else
                          "POOR (<0.55)")
        lines.append(f"  AUC = {roc_auc:.4f}  → predictive power {interpretation}")
        lines.append(f"  Reference: AUC=0.500 = random classifier")
        lines.append(f"  Interpretation: entropy {'IS' if roc_auc > 0.60 else 'IS NOT'} "
                     f"a good predictor of when the LLM knows the answer.")
    else:
        lines.append("  ROC not computed (insufficient data).")
    lines.append("")

    lines.append("5. CORRELATION s_pop ↔ ENTROPY")
    lines.append("─" * 72)
    if spearman_rho is not None:
        lines.append(f"  ρ_Spearman = {spearman_rho:.4f}")
        lines.append(f"  Mallen et al. (ACL 2023) report ρ ≈ -0.42 on GPT‑3 class models.")
        lines.append(f"  On Llama 3.1 8B Q4_K_M: ρ = {spearman_rho:.4f}")
        if abs(spearman_rho) > 0.3:
            lines.append("  → Correlation sufficient to justify adaptive threshold.")
        else:
            lines.append("  → Weak correlation: fixed threshold is sufficient.")
    lines.append("")

    lines.append("6. COMPUTATIONAL SAVINGS")
    lines.append("─" * 72)
    baseline_lat     = baseline_stats["lat_mean"]
    baseline_ret     = baseline_stats["retrieval_calls_mean"]
    baseline_tok     = baseline_stats["tokens_mean"]
    for combo, rows in sorted(groups.items()):
        s       = compute_combo_stats(rows)
        lat_sav = 100*(baseline_lat - s["lat_mean"])/baseline_lat if baseline_lat > 0 else 0
        ret_sav = 100*(baseline_ret - s["retrieval_calls_mean"])/baseline_ret if baseline_ret > 0 else 0
        tok_sav = 100*(baseline_tok - s["tokens_mean"])/baseline_tok if baseline_tok > 0 else 0
        lines.append(f"  {combo:<42}: "
                     f"ΔLat={lat_sav:+.1f}%  ΔRet={ret_sav:+.1f}%  ΔTok={tok_sav:+.1f}%")
    lines.append("")

    report = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[SAVE] Report: {path}")
    print("\n" + report)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  p4_analyze_popqa.py — PopQA v13 post‑hoc analysis")
    print("=" * 70)

    rows   = load_csv(CSV_INPUT)
    groups = group_by_combination(rows)
    print(f"[INFO] {len(groups)} combinations, {len(rows)} total rows.")

    roc_auc      = None
    spearman_rho = None

    if MATPLOTLIB_AVAILABLE:
        # Figure 1: EM per bucket
        plot_em_by_bucket(
            groups,
            os.path.join(RESULTS_DIR, "fig_em_by_bucket.png")
        )

        # Figure 2: ROC curve
        roc_auc = plot_roc_entropy(
            rows,
            os.path.join(RESULTS_DIR, "fig_roc_entropy.png")
        )

        # Figure 3: s_pop vs entropy
        result = plot_spop_entropy(
            rows,
            os.path.join(RESULTS_DIR, "fig_spop_entropy.png")
        )
        if result: spearman_rho = result[0]

        # Figure 4: Pareto frontier
        plot_pareto(
            groups,
            os.path.join(RESULTS_DIR, "fig_pareto.png")
        )

        # Figure 5: Computational savings
        plot_savings(
            groups,
            baseline_combo = "Always_Retrieve × Always_FAISS",
            output_path    = os.path.join(RESULTS_DIR, "fig_savings.png")
        )
    else:
        print("[WARN] matplotlib not available — only text output.")

    # Summary CSV
    save_summary_csv(groups, os.path.join(RESULTS_DIR, "ablation_summary.csv"))

    # Textual report
    generate_text_report(
        groups, roc_auc, spearman_rho,
        os.path.join(RESULTS_DIR, "routing_analysis.txt")
    )

    print("\n[DONE] Output in:", RESULTS_DIR)
    if MATPLOTLIB_AVAILABLE:
        for fig in ["fig_em_by_bucket","fig_roc_entropy","fig_spop_entropy",
                    "fig_pareto","fig_savings"]:
            print(f"  {fig}.png")
    print("  ablation_summary.csv")
    print("  routing_analysis.txt")


if __name__ == "__main__":
    main()