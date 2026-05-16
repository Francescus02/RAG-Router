"""
p1c_expand_corpus.py
====================
Expansion of the PopQA corpus with Hard Negatives and Anti-Contamination Filter (v7).

DEFINITIVE SCIENTIFIC SAFETY LOGIC:
  1. CONSERVATIVE ANSWER FILTER: Handles multi-word answers and punctuation
     via non-destructive regex boundary check.
  2. COMPLETE AUDIT TRAIL: Every rejection is logged with its original source.
  3. DYNAMIC SAFETY LIMITS: Number of attempts in the global pool calibrated
     on the available cache size.
  4. EFFORT MONITORING: avg_global_attempts metric to quantify the difficulty
     of retrieving uncontaminated distractors.

v7 IMPROVEMENTS:
  - dynamic max_attempts (safe handling of small caches).
  - Enhanced academic comments on the nature of the metrics.
  - Final optimization of the sampling loop.
  - Added cache_size metric to contextualize attempts.
"""

import os
import sys
import json
import time
import random
import requests
import urllib.parse
import re
import string
from typing import Dict, List, Optional, Set, Tuple, Any

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE      = os.path.join(BASE_DIR, "data", "processed", "popqa_vector_data.json")
OUTPUT_FILE     = os.path.join(BASE_DIR, "data", "processed", "popqa_vector_data_expanded.json")
STATS_FILE      = os.path.join(BASE_DIR, "data", "processed", "expansion_stats.json")
AUDIT_FILE      = os.path.join(BASE_DIR, "data", "processed", "contamination_audit.json")
CACHE_FILE      = os.path.join(BASE_DIR, "data", "cache", "wikipedia_cache.json")

# ── Parameters ────────────────────────────────────────────────────
MORELIKE_PER_SUBJECT  = 80    
LINKSHERE_PER_SUBJECT = 20    
MAX_DISTRACTORS       = 100   
BATCH_FETCH_DELAY     = 0.10  
MAX_RETRIES           = 3
RANDOM_SEED           = 42

random.seed(RANDOM_SEED)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "PopQA-RAG-Research/3.7"})

# Load Wikipedia cache
wiki_cache: Dict[str, str] = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        wiki_cache = json.load(f)

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(wiki_cache, f, ensure_ascii=False)

def _normalize(s: str) -> str:
    """Minimal normalization for accurate regex matching."""
    if not s: return ""
    s = s.lower().strip()
    return re.sub(r'\s+', ' ', s)

def is_contaminated(passage_norm: str, normalized_answers: List[str]) -> bool:
    """Check whether the passage contains the gold answer via boundary check."""
    for ans in normalized_answers:
        if not ans: continue
        # Flexible boundary check: ensures the answer is not part of a larger token
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(ans) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, passage_norm):
            return True
    return False

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: WIKIPEDIA API
# ═════════════════════════════════════════════════════════════════════════════

def search_morelike(title: str, limit: int = 80) -> List[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "list": "search", "srsearch": f"morelike:{title}",
        "srlimit": limit, "srnamespace": 0, "format": "json"
    }
    try:
        r = SESSION.get(url, params=params, timeout=10)
        if r.status_code == 200:
            results = r.json().get("query", {}).get("search", [])
            return [item["title"] for item in results if item.get("title") != title]
    except Exception: pass
    return []

def search_what_links_here(title: str, limit: int = 20) -> List[str]:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "linkshere", "titles": title,
        "lhnamespace": 0, "lhshow": "!redirect", "lhlimit": limit, "format": "json"
    }
    try:
        r = SESSION.get(url, params=params, timeout=10)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for pid, info in pages.items():
                if "linkshere" in info:
                    return [item["title"] for item in info["linkshere"] if item.get("title") != title]
    except Exception: pass
    return []

def fetch_wikipedia_summary(title: str) -> Optional[str]:
    if title in wiki_cache: return wiki_cache[title]
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title.replace(' ', '_'))}"
    try:
        r = SESSION.get(url, timeout=10)
        if r.status_code == 200:
            extract = r.json().get("extract", "").strip()
            if extract and len(extract) > 50:
                wiki_cache[title] = extract
                return extract
    except Exception: pass
    return None

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: EXPANSION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 85)
    print(f"  p1c_expand_corpus.py — Expansion v7")
    print("=" * 85)

    if not os.path.isfile(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        sys.exit(1)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        original_queries: List[Dict] = json.load(f)

    # PHASE 1: TITLE COLLECTION
    all_candidates: Dict[str, List[str]] = {}
    print(f"[PHASE 1] Searching hard negatives for {len(original_queries)} subjects...")
    est_min = (len(original_queries) * BATCH_FETCH_DELAY) / 60
    print(f"  Estimated API search time: ~{est_min:.1f} minutes")

    for i, q in enumerate(original_queries):
        subj = q["subj"]
        similar = search_morelike(subj, MORELIKE_PER_SUBJECT)
        links = search_what_links_here(subj, LINKSHERE_PER_SUBJECT)
        all_candidates[q["id"]] = list(set(similar + links))
        
        if (i+1) % 100 == 0: 
            print(f"  [{i+1}/{len(original_queries)}] Subjects processed...")
        time.sleep(BATCH_FETCH_DELAY)

    # PHASE 2: FETCHING SUMMARIES
    unique_titles = set(t for titles in all_candidates.values() for t in titles)
    new_titles = [t for t in unique_titles if t not in wiki_cache]
    print(f"\n[PHASE 2] Fetching summaries for {len(unique_titles)} titles...")
    print(f"  Cache hits: {len(unique_titles) - len(new_titles)} | To download: {len(new_titles)}")
    
    new_downloads = 0
    for i, title in enumerate(new_titles):
        if fetch_wikipedia_summary(title): 
            new_downloads += 1
            time.sleep(BATCH_FETCH_DELAY)
        
        if (i+1) % 500 == 0: 
            print(f"  [{i+1}/{len(new_titles)}] Progress: {100*(i+1)/len(new_titles):.1f}%...")
            save_cache()
    save_cache()

    # PHASE 3: GERARCHICAL DATASET RECONSTRUCTION
    print(f"\n[PHASE 3] Reconstructing dataset with hierarchy and attempt monitoring...")
    
    prop_to_gold: Dict[str, List[Tuple[str, str]]] = {} 
    for q in original_queries:
        p, g, qid = q.get("prop", "unknown"), q.get("gold_passage", ""), q.get("id", "")
        if g: prop_to_gold.setdefault(p, []).append((qid, g))

    processed = []
    contamination_audit = {} 
    stats = {
        "total_passages": 0, "n_queries": len(original_queries),
        "rejected_contamination": 0, "gold_not_found": 0,
        "gold_integrity_failed": 0, "total_global_attempts": 0,
        "cache_size": len(wiki_cache)
    }

    all_cached_titles = list(wiki_cache.keys())

    for q in original_queries:
        qid, gold, prop = q["id"], q.get("gold_passage", ""), q.get("prop", "unknown")
        possible_answers = q.get("possible_answers", [])
        if not gold: 
            stats["gold_not_found"] += 1
            continue

        # Pre-normalize gold and answers for contamination checks
        norm_answers = [_normalize(str(a)) for a in possible_answers]
        gold_norm = _normalize(gold)

        if not is_contaminated(gold_norm, norm_answers):
            stats["gold_integrity_failed"] += 1
            if stats["gold_integrity_failed"] <= 3:
                print(f"  [INFO] Gold passage {qid}: textual match not found (alias?)")

        distractor_pool: List[str] = []
        rejected_for_this_q = []

        def filter_and_add(candidates: List[Tuple[str, str, str]]):
            """candidates: list of (source, identifier, text)"""
            for src, ident, txt in candidates:
                if len(distractor_pool) >= MAX_DISTRACTORS: break
                if txt == gold: continue
                
                txt_norm = _normalize(txt)
                if is_contaminated(txt_norm, norm_answers):
                    stats["rejected_contamination"] += 1
                    rejected_for_this_q.append(f"{src}:{ident}")
                    continue
                
                if txt not in distractor_pool:
                    distractor_pool.append(txt)

        # Priority 1: Specific hard negatives (morelike/linkshere)
        specific = [("WikiTitle", t, f"Title: {t}. Content: {wiki_cache[t]}") 
                    for t in all_candidates.get(qid, []) if t in wiki_cache]
        filter_and_add(specific)

        # Priority 2: Other gold passages with the same property
        if len(distractor_pool) < MAX_DISTRACTORS:
            same_prop = [("OtherGold", donor_id, g_txt) 
                         for donor_id, g_txt in prop_to_gold.get(prop, [])]
            random.shuffle(same_prop)
            filter_and_add(same_prop)

        # Priority 3: Global pool (random titles from cache)
        if len(distractor_pool) < MAX_DISTRACTORS:
            random.shuffle(all_cached_titles)
            attempts = 0
            # Dynamic limit to prevent infinite loops in small caches
            max_attempts = min(MAX_DISTRACTORS * 10, len(all_cached_titles))
            
            for t in all_cached_titles:
                if len(distractor_pool) >= MAX_DISTRACTORS or attempts >= max_attempts:
                    break
                attempts += 1
                stats["total_global_attempts"] += 1
                
                txt = f"Title: {t}. Content: {wiki_cache[t]}"
                if txt == gold or txt in distractor_pool: continue
                
                txt_norm = _normalize(txt)
                if is_contaminated(txt_norm, norm_answers):
                    stats["rejected_contamination"] += 1
                    rejected_for_this_q.append(f"GlobalPool:{t}")
                else:
                    distractor_pool.append(txt)

        if rejected_for_this_q:
            contamination_audit[qid] = rejected_for_this_q

        # Query finalization
        distractors = distractor_pool[:MAX_DISTRACTORS]
        all_ctx = [gold] + distractors
        random.shuffle(all_ctx)
        
        q_expanded = dict(q)
        q_expanded["contexts"] = all_ctx
        processed.append(q_expanded)
        stats["total_passages"] += len(all_ctx)

    # Final statistics
    n_proc = len(processed)
    stats["avg_ctx"] = stats["total_passages"] / n_proc if n_proc else 0
    stats["unique_passages"] = len(set(c for q in processed for c in q["contexts"]))
    stats["avg_global_attempts"] = stats["total_global_attempts"] / n_proc if n_proc else 0

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(contamination_audit, f, indent=2, ensure_ascii=False)
    
    print(f"\n[DONE] Expanded corpus saved: {OUTPUT_FILE}")
    print(f"  Final Scientific Statistics:")
    print(f"    - Passages rejected for contamination: {stats['rejected_contamination']}")
    print(f"    - Average global scans per query:    {stats['avg_global_attempts']:.1f} (on cache size {stats['cache_size']})")
    print(f"    - Total unique passages in index:    {stats['unique_passages']:,}")
    print(f"    - Audit trail (rejections) saved to: {AUDIT_FILE}")
    print(f"  Next step: p2_rebuild_indexes.py")

if __name__ == "__main__":
    main()