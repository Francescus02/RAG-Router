"""
p2_rebuild_indexes.py
=====================
Rebuilding FAISS indexes (IVFFlat/HNSW/IVFPQ) and BM25 (v7).

v7 IMPROVEMENTS (Ultimate Academic Edition):
  - Integrity check on embeddings cache (.npy).
  - --log_level parameter for automated environments.
  - Automatic saving of a Markdown report (validation_summary.md).
  - (MAP@k metric coincides with MRR since there is only one gold passage per query).
  - FIX: Removed multi-line f-string for IDE syntax highlighter compatibility.
  - FIX: Updated NumPy 2.0 syntax (replaced np.asfarray with np.asarray) for NDCG metric.
"""

import os
import sys
import json
import pickle
import numpy as np
import faiss
import hashlib
import argparse
import random
import string
import logging
import csv
import multiprocessing
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "processed", "popqa_vector_data_expanded.json")
INDEX_DIR  = os.path.join(BASE_DIR, "db_indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

FAISS_INDEX    = os.path.join(INDEX_DIR, "popqa_expanded.faiss")
FAISS_METADATA = os.path.join(INDEX_DIR, "popqa_expanded_metadata.json")
BM25_PKL       = os.path.join(INDEX_DIR, "popqa_expanded_bm25.pkl")
BM25_CORPUS    = os.path.join(INDEX_DIR, "popqa_expanded_bm25_corpus.json")
VALIDATION_OUT = os.path.join(INDEX_DIR, "validation_results.json")
DETAILED_OUT   = os.path.join(INDEX_DIR, "detailed_query_results.json")
VALIDATION_CSV = os.path.join(INDEX_DIR, "validation_results.csv")
REPORT_MD      = os.path.join(INDEX_DIR, "validation_summary.md")
CONFIG_OUT     = os.path.join(INDEX_DIR, "run_config.json")
EMBEDDINGS_OUT = os.path.join(INDEX_DIR, "embeddings_cache.npy")
DISTANCES_OUT  = os.path.join(INDEX_DIR, "distances_matrix.npy")

# ── Initial Logging Configuration ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("p2_indexes")

def text_hash(t: str) -> str:
    return hashlib.md5(t.encode("utf-8")).hexdigest()

def dcg_at_k(r: List[int], k: int) -> float:
    # FIX for NumPy 2.0: asfarray has been removed, use asarray with float dtype
    r = np.asarray(r, dtype=float)[:k]
    if r.size: return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def load_data(path: str) -> List[Dict]:
    logger.info(f"Loading data from {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"{len(data)} queries loaded from the expanded corpus.")
    return data

def build_flat_corpus(data: List[Dict]) -> Tuple[List[str], List[Dict]]:
    seen_texts: Dict[str, int] = {}
    passages: List[str] = []
    metadata: List[Dict] = []
    
    for item in data:
        qid    = item["id"]
        bucket = item.get("pop_bucket", "unknown")
        s_pop  = item.get("s_pop", 0)
        gold_text = item.get("gold_passage", "")
        
        if not gold_text:
            continue

        for ctx in item.get("contexts", []):
            h = text_hash(ctx)
            if h not in seen_texts:
                seen_texts[h] = len(passages)
                passages.append(ctx)
                metadata.append({
                    "text": ctx, 
                    "is_gold": (ctx == gold_text),
                    "query_id": qid,
                    "pop_bucket": bucket,
                    "s_pop": s_pop
                })
                
    logger.info(f"{len(passages):,} unique passages extracted.")
    
    if len(passages) > 0:
        idx = random.randint(0, len(passages)-1)
        if text_hash(passages[idx]) != text_hash(metadata[idx]["text"]):
            logger.error("Deduplication Consistency Check failed!")
            sys.exit(1)
            
    return passages, metadata

def encode_with_oom_fallback(embedder, passages, start_batch=128):
    batch_size = start_batch
    while batch_size >= 8:
        try:
            return embedder.encode(passages, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM detected. Reducing batch size: {batch_size} -> {batch_size//2}")
                batch_size //= 2
            else:
                raise e
    raise RuntimeError("Unable to encode: insufficient memory even with batch_size=8.")

def generate_markdown_report(args, faiss_results: Dict, bm25_recall: float, num_passages: int):
    """Generates a Markdown summary file of validation results."""
    lines = [
        "# Validation Summary: PopQA Indexes",
        "",
        "## Configuration",
        f"- **Embedding Model**: `{args.embedding_model}`",
        f"- **Index Type**: `{args.index_type.upper()}`",
        f"- **Metric**: `{args.metric.upper()}`",
        f"- **Unique Passages**: {num_passages:,}",
        f"- **Top-K Evaluation**: {args.top_k}",
        "",
        "## Corpus Coverage",
        f"- **Coverage (Theoretical Max Recall)**: {faiss_results['coverage'] * 100:.2f}%",
        "",
        "## FAISS Results (Dense Retrieval)",
        "| Metric | Value |",
        "|---|---|",
        f"| **Recall@{args.top_k} (Macro Avg)** | {faiss_results['macro_recall']:.4f} |",
        f"| **Recall@{args.top_k} (Micro Avg)** | {faiss_results['micro_recall']:.4f} |",
        f"| **MRR@{args.top_k} (Avg)** | {faiss_results['mrr']:.4f} |",
        f"| **Skew Bias (High vs Low)** | {faiss_results['bias']:.4f} |",
        "",
        "### FAISS Recall per Bucket",
        "| Bucket | Recall@{args.top_k} |",
        "|---|---|",
        f"| High | {faiss_results['recall_by_bucket'].get('high', 0.0):.4f} |",
        f"| Medium | {faiss_results['recall_by_bucket'].get('medium', 0.0):.4f} |",
        f"| Low | {faiss_results['recall_by_bucket'].get('low', 0.0):.4f} |",
        "",
        "## BM25 Results (Lexical Retrieval)",
        f"- **Recall@{args.top_k} (Micro)**: {bm25_recall:.4f}"
    ]
    md_content = "\n".join(lines)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Markdown report saved: {REPORT_MD}")

def build_faiss_index(passages: List[str], metadata: List[Dict], data: List[Dict], args) -> Tuple[float, float, List[Dict], Dict]:
    embedder = SentenceTransformer(args.embedding_model, device="cuda" if args.use_gpu else "cpu")
    
    embeddings = None
    if args.cache_embeddings and os.path.exists(EMBEDDINGS_OUT):
        try:
            temp_embeddings = np.load(EMBEDDINGS_OUT)
            if len(temp_embeddings.shape) == 2 and temp_embeddings.shape[0] == len(passages):
                embeddings = temp_embeddings
                logger.info(f"Embeddings loaded from cache: {EMBEDDINGS_OUT} (shape: {embeddings.shape})")
            else:
                logger.warning(f"Incompatible embeddings cache. Recomputing...")
        except Exception as e:
            logger.warning(f"Error loading embeddings cache ({e}). Recomputing...")

    if embeddings is None:
        logger.info(f"Encoding {len(passages):,} passages (Model: {args.embedding_model})...")
        embeddings = encode_with_oom_fallback(embedder, passages)
        embeddings = np.array(embeddings, dtype="float32")
        if (args.save_embeddings or args.cache_embeddings) and not args.dry_run:
            np.save(EMBEDDINGS_OUT, embeddings)
            logger.info(f"Embeddings saved to {EMBEDDINGS_OUT}")
    
    dim = embeddings.shape[1]
    metric = faiss.METRIC_INNER_PRODUCT if args.metric == "ip" else faiss.METRIC_L2
    
    logger.info(f"Creating FAISS index type: {args.index_type.upper()}")
    if args.index_type == "ivfflat":
        quantizer = faiss.IndexFlatIP(dim) if args.metric == "ip" else faiss.IndexFlatL2(dim)
        nlist = args.nlist
        if nlist > len(passages):
            nlist = max(1, int(np.sqrt(len(passages))))
            logger.warning(f"Original NLIST too high. Dynamically reduced to {nlist}.")
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric)
    elif args.index_type == "ivfpq":
        quantizer = faiss.IndexFlatIP(dim) if args.metric == "ip" else faiss.IndexFlatL2(dim)
        nlist = args.nlist
        if nlist > len(passages):
            nlist = max(1, int(np.sqrt(len(passages))))
        
        if args.pq_m > dim:
            logger.warning(f"pq_m ({args.pq_m}) > dim ({dim}). Setting pq_m = {dim}")
            args.pq_m = dim
        if dim % args.pq_m != 0:
            logger.warning(f"dim ({dim}) is not a multiple of pq_m ({args.pq_m}). FAISS may raise an exception.")
            
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, args.pq_m, args.pq_nbits, metric)
    elif args.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, metric)
    else:
        index = faiss.IndexFlatIP(dim) if args.metric == "ip" else faiss.IndexFlatL2(dim)
    
    if args.use_gpu and args.index_type not in ["hnsw", "ivfpq"]:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("GPU enabled for FAISS.")
        except AttributeError:
            logger.warning("faiss-gpu not available. Falling back to CPU.")
        except Exception as e:
            logger.warning(f"Unable to move index to GPU: {e}. Falling back to CPU.")

    if args.index_type in ["ivfflat", "ivfpq"]:
        logger.info(f"Training k-means (NLIST={nlist})...")
        try:
            index.train(embeddings)
        except Exception as e:
            logger.error(f"FAISS training failed: {e}. Falling back to IndexFlat.")
            index = faiss.IndexFlatIP(dim) if args.metric == "ip" else faiss.IndexFlatL2(dim)

    logger.info("Adding vectors...")
    index.add(embeddings)
    if hasattr(index, 'nprobe'):
        index.nprobe = args.nprobe

    index_size = index.ntotal * (dim * 4) / (1024**2)
    logger.info(f"Estimated raw vectors size: {index_size:.1f} MB")

    if args.use_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
        try:
            index_to_save = faiss.index_gpu_to_cpu(index)
        except Exception:
            index_to_save = index
    else:
        index_to_save = index

    # --- Validation ---
    logger.info(f"Computing diagnostic metrics (Recall@{args.top_k}, NDCG@{args.top_k})...")
    bucket_hits_10, bucket_hits_1, bucket_ndcg = defaultdict(int), defaultdict(int), defaultdict(float)
    bucket_totals = defaultdict(int)
    gold_positions = []
    detailed_metrics = []
    
    text_to_idx = {text_hash(p): i for i, p in enumerate(passages)}
    
    valid_queries = [q for q in data if q.get("gold_passage")]
    coverage_hits = sum(1 for q in valid_queries if text_hash(q["gold_passage"]) in text_to_idx)
    coverage = coverage_hits / len(valid_queries) if valid_queries else 0.0
    logger.info(f"Corpus Coverage: {coverage:.4f} ({coverage_hits}/{len(valid_queries)} gold passages found after deduplication)")
    if coverage < 0.90:
        logger.warning("Coverage < 90%. Optimal Recall results will be structurally limited.")
    
    test_samples = []
    
    if args.sampling == "strat_equal":
        for b in ["high", "medium", "low"]:
            b_queries = [q for q in valid_queries if q.get("pop_bucket") == b]
            test_samples.extend(random.sample(b_queries, min(len(b_queries), args.samples)))
    else:
        total_valid = len(valid_queries)
        total_samples = args.samples * 3
        for b in ["high", "medium", "low"]:
            b_queries = [q for q in valid_queries if q.get("pop_bucket") == b]
            prop_samples = int(total_samples * (len(b_queries) / total_valid))
            test_samples.extend(random.sample(b_queries, min(len(b_queries), prop_samples)))

    all_dists = []
    
    for q in test_samples:
        gold_text = q.get("gold_passage", "")
        gold_h = text_hash(gold_text)
        if gold_h not in text_to_idx: continue
            
        bucket = q["pop_bucket"]
        bucket_totals[bucket] += 1
        
        q_vec = embedder.encode([q["question"]], normalize_embeddings=True, show_progress_bar=False)
        D, I = index.search(np.array(q_vec, dtype="float32"), args.top_k)
        
        if args.save_distances: all_dists.append(D[0])
        
        retrieved_hashes = [text_hash(passages[i]) for i in I[0] if i >= 0]
        relevance_vector = [1 if h == gold_h else 0 for h in retrieved_hashes]
        
        idcg = dcg_at_k([1] + [0]*(args.top_k-1), args.top_k)
        ndcg = dcg_at_k(relevance_vector, args.top_k) / idcg if idcg > 0 else 0
        bucket_ndcg[bucket] += ndcg
        
        hit_10 = 0
        hit_1 = 0
        pos = -1
        
        if gold_h in retrieved_hashes:
            hit_10 = 1
            bucket_hits_10[bucket] += 1
            pos = retrieved_hashes.index(gold_h) + 1
            gold_positions.append(pos)
            if pos == 1:
                hit_1 = 1
                bucket_hits_1[bucket] += 1
                
        if args.save_detailed:
            detailed_metrics.append({
                "query_id": q["id"],
                "bucket": bucket,
                f"hit_at_{args.top_k}": hit_10,
                "hit_at_1": hit_1,
                "rank_pos": pos,
                f"ndcg_{args.top_k}": ndcg
            })

    total_recall = 0
    bucket_recalls = {}
    logger.info("-" * 45)
    for b in ["high", "medium", "low"]:
        if bucket_totals[b] == 0: continue
        rec = bucket_hits_10[b] / bucket_totals[b]
        ndcg_val = bucket_ndcg[b] / bucket_totals[b]
        s1 = bucket_hits_1[b] / bucket_totals[b]
        total_recall += rec
        bucket_recalls[b] = rec
        logger.info(f"[{b:6s}] n={bucket_totals[b]}, Rec={rec:.3f}, NDCG={ndcg_val:.3f}, S@1={s1:.3f}")
        
    avg_recall = total_recall / 3
    total_hits = sum(bucket_hits_10.values())
    total_queries = sum(bucket_totals.values())
    micro_recall = total_hits / total_queries if total_queries else 0.0
    
    mrr = np.mean([1.0/p for p in gold_positions]) if gold_positions else 0.0
    
    bias_score = 0.0
    if "high" in bucket_recalls and "low" in bucket_recalls:
        rh, rl = bucket_recalls["high"], bucket_recalls["low"]
        bias_score = (rh - rl) / max(rh + rl, 1e-9)
        
    logger.info("-" * 45)
    logger.info(f"Recall@{args.top_k} MACRO: {avg_recall:.4f}")
    logger.info(f"MRR@{args.top_k} AVG:    {mrr:.4f}")
    logger.info(f"Skew Bias (High vs Low): {bias_score:.4f} (0=Fair, >0=Bias vs Low)")
    
    if avg_recall >= 0.99:
        logger.warning("Recall@10 ≈ 1.0 – The corpus may not stress post-retrieval.")

    results = {
        "coverage": coverage, 
        "recall_by_bucket": bucket_recalls, 
        "macro_recall": avg_recall, 
        "micro_recall": micro_recall, 
        "mrr": mrr, 
        "bias": bias_score
    }

    if not args.dry_run:
        faiss.write_index(index_to_save, FAISS_INDEX)
        with open(FAISS_METADATA, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)
        with open(VALIDATION_OUT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if args.save_detailed and detailed_metrics:
            with open(DETAILED_OUT, "w", encoding="utf-8") as f:
                json.dump(detailed_metrics, f, indent=2)
            logger.info(f"Detailed metrics saved to {DETAILED_OUT}")
        if args.save_distances and all_dists:
            np.save(DISTANCES_OUT, np.array(all_dists))
            logger.info(f"Distance matrix saved: {DISTANCES_OUT}")

    return avg_recall, mrr, test_samples, results

def tokenize_bm25(text: str) -> List[str]:
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

def build_bm25_index(passages: List[str], test_samples: List[Dict], args, faiss_results: Dict):
    if not BM25_AVAILABLE:
        logger.warning("rank_bm25 not installed: skipping BM25.")
        return 0.0

    logger.info(f"Tokenizing {len(passages):,} passages for BM25...")
    if args.parallel_bm25:
        n_workers = args.bm25_workers if args.bm25_workers else max(1, multiprocessing.cpu_count() // 2)
        logger.info(f"Using Multiprocessing ({n_workers} workers) for BM25 tokenization...")
        with multiprocessing.Pool(processes=n_workers) as pool:
            tokenized = pool.map(tokenize_bm25, passages)
    else:
        tokenized = [tokenize_bm25(p) for p in passages]

    logger.info("Building Okapi index...")
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)

    if not args.dry_run:
        with open(BM25_PKL, "wb") as f: pickle.dump(bm25, f)
        with open(BM25_CORPUS, "w", encoding="utf-8") as f: json.dump(passages, f, ensure_ascii=False)

    bm25_hits = 0
    for q in test_samples:
        gold_text = q.get("gold_passage", "")
        if not gold_text: continue
        
        tokenized_q = tokenize_bm25(q["question"])
        scores = bm25.get_scores(tokenized_q)
        top_idx = np.argsort(scores)[::-1][:args.top_k]
        retrieved_hashes = [text_hash(passages[i]) for i in top_idx]
        
        if text_hash(gold_text) in retrieved_hashes:
            bm25_hits += 1

    bm25_recall = bm25_hits / len(test_samples) if test_samples else 0.0
    logger.info(f"Recall@{args.top_k} BM25 (MICRO): {bm25_recall:.4f}")
    
    if args.export_csv and not args.dry_run:
        with open(VALIDATION_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "Recall_Macro", "Recall_Micro", "MRR", "Bias", "BM25_Recall", "Coverage"])
            writer.writerow([args.embedding_model, f"{faiss_results['macro_recall']:.4f}", 
                             f"{faiss_results['micro_recall']:.4f}", f"{faiss_results['mrr']:.4f}",
                             f"{faiss_results['bias']:.4f}", f"{bm25_recall:.4f}", f"{faiss_results['coverage']:.4f}"])
        logger.info(f"CSV export: {VALIDATION_CSV}")
        
    return bm25_recall

def main():
    parser = argparse.ArgumentParser(description="Advanced RAG Index Rebuilding.")
    parser.add_argument("--nlist", type=int, default=256, help="Number of clusters (IVFFlat)")
    parser.add_argument("--nprobe", type=int, default=20, help="Clusters to explore")
    parser.add_argument("--top_k", type=int, default=10, help="Top-K for Recall evaluation")
    parser.add_argument("--samples", type=int, default=50, help="Test samples per bucket")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random and numpy")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for FAISS")
    parser.add_argument("--save_embeddings", action="store_true", help="Save raw vectors to .npy")
    parser.add_argument("--cache_embeddings", action="store_true", help="Reuse vectors if available")
    parser.add_argument("--metric", type=str, choices=["l2", "ip"], default="l2", help="FAISS distance")
    parser.add_argument("--index_type", type=str, choices=["ivfflat", "hnsw", "flat", "ivfpq"], default="ivfflat", help="FAISS index type")
    parser.add_argument("--pq_m", type=int, default=8, help="Parameter M for IVFPQ (subquantizers)")
    parser.add_argument("--pq_nbits", type=int, default=8, help="Bits per subquantizer for IVFPQ")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformers model")
    parser.add_argument("--parallel_bm25", action="store_true", help="Multiprocessing for tokenization")
    parser.add_argument("--bm25_workers", type=int, default=None, help="Number of workers for BM25 (default: cpu_count // 2)")
    parser.add_argument("--sampling", type=str, choices=["strat_equal", "strat_prop"], default="strat_equal", help="Validation sampling method")
    parser.add_argument("--export_csv", action="store_true", help="Export validation summary to CSV")
    parser.add_argument("--save_distances", action="store_true", help="Save validation distance matrix (.npy)")
    parser.add_argument("--save_detailed", action="store_true", help="Save per-query detailed metrics to JSON")
    parser.add_argument("--dry_run", action="store_true", help="Do not save any files to disk")
    parser.add_argument("--log_level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level))

    random.seed(args.seed)
    np.random.seed(args.seed)

    logger.info("=" * 70)
    logger.info("  p2_rebuild_indexes.py — Advanced FAISS + BM25 Builder")
    try:
        logger.info(f"  FAISS version: {faiss.__version__}")
    except AttributeError:
        logger.info("  FAISS version: Unknown")
    logger.info("=" * 70)
    
    if args.dry_run: logger.warning("DRY RUN: No files will be modified on disk.")
    
    if not os.path.isfile(INPUT_FILE):
        logger.error(f"File not found: {INPUT_FILE}")
        sys.exit(1)

    data = load_data(INPUT_FILE)
    passages, metadata = build_flat_corpus(data)
    
    recall, mrr, test_samples, faiss_results = build_faiss_index(passages, metadata, data, args)
    bm25_recall = build_bm25_index(passages, test_samples, args, faiss_results)

    if not args.dry_run:
        with open(CONFIG_OUT, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2)
        logger.info(f"Configuration saved to {CONFIG_OUT}")
        generate_markdown_report(args, faiss_results, bm25_recall, len(passages))

    logger.info("=" * 70)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 70)
    if not args.dry_run:
        logger.info(f"  FAISS Index: {FAISS_INDEX}")
        logger.info(f"  BM25 Pkl:    {BM25_PKL}")
        logger.info(f"  Report MD:   {REPORT_MD}")
        logger.info("  NEXT STEP: python scripts/p3_router_popqa.py")
    else:
        logger.info("  Finished successfully (Dry Run).")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()