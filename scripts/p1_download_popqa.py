"""
p1_download_popqa.py
====================
Download the PopQA dataset and build the retrieval corpus.

v7 IMPROVEMENTS (Robustness and Data Quality):
  1. Automatic verification of gold answer presence in the passage.
  2. Persistent caching (JSON) for Wikipedia API calls.
  3. Distractors sampled by predicate similarity (prop) to increase
     retrieval difficulty (hard negatives).
  4. Analysis and tagging of answer format (entity, year, number, phrase).
  5. API redirect for low-pop malformed entities.
  6. Saving gold passage length.
  7. Traceability of seed and script version for reproducibility.

Output:
  data/raw/popqa_sample.json          → 1500 queries with s_pop and metadata
  data/processed/popqa_vector_data.json → format ready for FAISS
"""

import os
import sys
import re
import json
import time
import random
import requests
import urllib.parse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# ─── Dependencies ──────────────────────────────────────────────────────
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[WARN] 'datasets' not installed. Attempting direct TSV download.")

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
CACHE_DIR     = os.path.join(BASE_DIR, "data", "cache")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Output and Cache files ───────────────────────────────────────────────────────
SAMPLE_OUTPUT = os.path.join(RAW_DIR,       "popqa_sample.json")
VECTOR_OUTPUT = os.path.join(PROCESSED_DIR, "popqa_vector_data.json")
CACHE_FILE    = os.path.join(CACHE_DIR,     "wikipedia_cache.json")

# ── Global parameters ──────────────────────────────────────────────────────────
SCRIPT_VERSION          = "v7"
RANDOM_SEED             = 42
BUCKET_CONFIG = {
    "high":   {"min": 1000,  "max": float("inf"), "n": 500},
    "medium": {"min": 100,   "max": 1000,          "n": 500},
    "low":    {"min": 0,     "max": 100,            "n": 500},
}
N_DISTRACTORS_PER_QUERY = 4
WIKIPEDIA_API_DELAY     = 0.10
MAX_RETRIES             = 3

# Load Wikipedia cache
wiki_cache: Dict[str, str] = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        wiki_cache = json.load(f)

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(wiki_cache, f, ensure_ascii=False)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: HELPER FUNCTIONS (Answer analysis)
# ═════════════════════════════════════════════════════════════════════════════

def determine_answer_format(possible_answers: List[Any]) -> str:
    """Classifies the expected answer format for stratified analysis."""
    if not possible_answers: return "unknown"
    ans = str(possible_answers[0]).strip()
    if re.match(r'^\d{4}$', ans): return "year"
    if re.match(r'^-?\d+(?:\.\d+)?$', ans): return "number"
    if len(ans.split()) >= 3: return "phrase"
    return "entity"

def passage_contains_answer(passage: str, possible_answers: List[Any]) -> bool:
    """Verifies that the gold answer is present in the retrieved document."""
    if not passage or not possible_answers: return False
    p_low = passage.lower()
    for ans in possible_answers:
        if str(ans).lower() in p_low:
            return True
    return False

def resolve_redirect(title: str) -> str:
    """Queries the API to resolve disambiguations and redirects for low-pop entities."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "titles": title, "redirects": 1, "format": "json"}
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            redirects = data.get("query", {}).get("redirects", [])
            if redirects:
                return redirects[0]["to"]
    except Exception:
        pass
    return title


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: DOWNLOAD AND SAMPLING
# ═════════════════════════════════════════════════════════════════════════════

def load_popqa_huggingface() -> List[Dict]:
    print("[HF] Download PopQA from Hugging Face...")
    ds = load_dataset("akariasai/PopQA", split="test")
    records = []
    for row in ds:
        records.append({
            "id":               str(row["id"]),
            "question":         row["question"],
            "subj":             row["subj"],
            "obj":              row["obj"],
            "prop":             row["prop"],
            "s_pop":            float(row.get("s_pop", 1)),
            "o_pop":            float(row.get("o_pop", 1)),
            "possible_answers": row.get("possible_answers", [row["obj"]]),
            "s_aliases":        row.get("s_aliases", []),
        })
    print(f"[HF] Uploaded {len(records)} record.")
    return records


def load_popqa_tsv_fallback() -> List[Dict]:
    url = "https://raw.githubusercontent.com/AlexMallen/PopQA/main/data/popqa_longitudinal.tsv"
    print(f"[TSV] Download from: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    lines = r.text.strip().split("\n")
    header = lines[0].split("\t")
    records = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < len(header): continue
        row = dict(zip(header, parts))
        try:
            records.append({
                "id":               row.get("id", ""),
                "question":         row.get("question", ""),
                "subj":             row.get("subj", ""),
                "obj":              row.get("obj", ""),
                "prop":             row.get("prop", ""),
                "s_pop":            float(row.get("s_pop", 1)),
                "o_pop":            float(row.get("o_pop", 1)),
                "possible_answers": json.loads(row.get("possible_answers", f'["{row.get("obj","")}"]')),
                "s_aliases":        json.loads(row.get("s_aliases", "[]")),
            })
        except (json.JSONDecodeError, ValueError):
            continue

    print(f"[TSV] Uploaded {len(records)} record.")
    return records


def stratified_sample(records: List[Dict], config: Dict, seed: int = 42) -> List[Dict]:
    random.seed(seed)
    sampled = []

    for bucket_name, params in config.items():
        bucket = [r for r in records if params["min"] <= r["s_pop"] < params["max"]]
        n_available = len(bucket)
        n_sample    = min(params["n"], n_available)

        if n_sample < params["n"]:
            print(f"[WARN] Bucket '{bucket_name}': requested {params['n']}, available {n_available}.")

        chosen = random.sample(bucket, n_sample)
        for r in chosen:
            r["pop_bucket"] = bucket_name
            # Metadata for reproducibility added in-place
            r["sampling_seed"] = seed
            r["script_version"] = SCRIPT_VERSION
            # Metadata for analysis
            r["answer_format"] = determine_answer_format(r.get("possible_answers", []))
            
        sampled.extend(chosen)
        max_val_str = "inf" if params["max"] == float("inf") else str(int(params["max"]))
        print(f"[SAMPLE] {bucket_name:6s} (s_pop {params['min']:>5}–{max_val_str:>6}): "
              f"{n_sample} query sampled.")

    random.shuffle(sampled)
    print(f"[SAMPLE] Total sample: {len(sampled)} queries.")
    return sampled


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: FETCH WIKIPEDIA PASSAGES
# ═════════════════════════════════════════════════════════════════════════════

def fetch_wikipedia_summary(title: str, retries: int = MAX_RETRIES) -> Optional[str]:
    # 1. Check in Cache
    if title in wiki_cache:
        return wiki_cache[title]
        
    title_encoded = urllib.parse.quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_encoded}"

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=10, headers={"User-Agent": "PopQA-RAG-Research/2.0"})
            if r.status_code == 200:
                data = r.json()
                extract = data.get("extract", "").strip()
                if extract and len(extract) > 30:
                    wiki_cache[title] = extract
                    return extract
                return None
            elif r.status_code == 404:
                # If the first attempt fails with 404, try to resolve potential redirects
                if attempt == 0:
                    redirected = resolve_redirect(title)
                    if redirected != title:
                        return fetch_wikipedia_summary(redirected, retries - 1)
                return None
            elif r.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                return None
        except requests.Timeout:
            if attempt < retries - 1:
                time.sleep(1)
        except Exception as e:
            return None

    return None


def build_passage(query: Dict) -> Optional[str]:
    title = query.get("subj", "")
    text  = fetch_wikipedia_summary(title)

    if text is None and query.get("s_aliases"):
        aliases = query["s_aliases"]
        if isinstance(aliases, str):
            try: aliases = json.loads(aliases)
            except Exception: aliases = []
            
        for alias in aliases[:2]:
            text = fetch_wikipedia_summary(str(alias))
            if text:
                title = str(alias)
                break

    if text is None:
        return None

    return f"Title: {title}. Content: {text}"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: CONSTRUCTION OF CORPUS AND HARD NEGATIVES
# ═════════════════════════════════════════════════════════════════════════════

def build_corpus(sampled: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    print(f"\n[CORPUS] Fetch Wikipedia for {len(sampled)} entities...")

    passage_map: Dict[str, str] = {}
    prop_map: Dict[str, List[str]] = defaultdict(list)
    failed: List[Dict] = []
    cached_hits = 0

    for i, query in enumerate(sampled):
        subj = query.get("subj", "")
        if subj in wiki_cache:
            cached_hits += 1
            
        passage = build_passage(query)
        if passage:
            passage_map[query["id"]] = passage
            prop_map[query.get("prop", "unknown")].append(query["id"])
        else:
            failed.append(query)

        if subj not in wiki_cache:  # Time delay only for non-cached calls
            time.sleep(WIKIPEDIA_API_DELAY)

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(sampled)}] Fetch completed (Cache hits: {cached_hits}).")
            save_cache() # Partial save to prevent data loss in long runs

    save_cache() # Final save after all fetches
    print(f"\n[CORPUS] Fetch completed: {len(passage_map)} successful, {len(failed)} failed.")

    successful_ids = list(passage_map.keys())
    processed      = []

    for query in sampled:
        qid = query["id"]
        if qid not in passage_map:
            continue

        gold_passage = passage_map[qid]
        prop = query.get("prop", "unknown")
        distractor_ids = []

        # -- HARD NEGATIVES SELECTION BASED ON PREDICATE (PROP) --
        # Attempt to take half of the distractors from queries sharing the same property
        same_prop_candidates = [oid for oid in prop_map.get(prop, []) if oid != qid]
        n_same = min(N_DISTRACTORS_PER_QUERY // 2, len(same_prop_candidates))
        
        if n_same > 0:
            distractor_ids.extend(random.sample(same_prop_candidates, n_same))
            
        # Fill the rest with random entities from the pool
        other_candidates = [oid for oid in successful_ids if oid != qid and oid not in distractor_ids]
        n_rest = min(N_DISTRACTORS_PER_QUERY - len(distractor_ids), len(other_candidates))
        
        if n_rest > 0:
            distractor_ids.extend(random.sample(other_candidates, n_rest))

        distractor_passages = [passage_map[did] for did in distractor_ids]
        all_passages = [gold_passage] + distractor_passages
        random.shuffle(all_passages)

        # -- METADATA FOR GOLD PASSAGE --
        possible_answers = query.get("possible_answers", [query.get("obj", "")])
        has_answer = passage_contains_answer(gold_passage, possible_answers)

        processed.append({
            "id":                   qid,
            "question":             query["question"],
            "s_pop":                query["s_pop"],
            "pop_bucket":           query["pop_bucket"],
            "subj":                 query["subj"],
            "prop":                 query["prop"],
            "possible_answers":     possible_answers,
            "answer_format":        query.get("answer_format", "unknown"),
            "gold_passage":         gold_passage,
            "gold_passage_length":  len(gold_passage),
            "gold_answer_present":  has_answer,
            "contexts":             all_passages,
        })

    print(f"[CORPUS] {len(processed)} queries with retrieval corpus built.")
    return processed, failed


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print(f"  p1_download_popqa.py — Download and preprocessing PopQA ({SCRIPT_VERSION})")
    print("=" * 70)

    if HF_AVAILABLE:
        records = load_popqa_huggingface()
    else:
        records = load_popqa_tsv_fallback()

    if not records:
        print("[ERROR] No records loaded. Check your connection.")
        sys.exit(1)

    sampled = stratified_sample(records, BUCKET_CONFIG, seed=RANDOM_SEED)

    with open(SAMPLE_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Raw sample: {SAMPLE_OUTPUT}")

    processed, failed = build_corpus(sampled)

    if not processed:
        print("[ERROR] No queries processed. Check access to Wikipedia API.")
        sys.exit(1)

    with open(VECTOR_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=2, ensure_ascii=False)

    buckets = {"high": 0, "medium": 0, "low": 0}
    gold_missing = 0
    for q in processed:
        buckets[q["pop_bucket"]] += 1
        if not q.get("gold_answer_present", True):
            gold_missing += 1

    print(f"\n[DONE] Dataset PopQA ready.")
    print(f"  Total queries:          {len(processed)}")
    print(f"  Bucket HIGH (≥1000):   {buckets['high']}")
    print(f"  Bucket MEDIUM (100-999):{buckets['medium']}")
    print(f"  Bucket LOW (<100):     {buckets['low']}")
    print(f"  Failed fetches:         {len(failed)}")
    print(f"  Gold answer missing:   {gold_missing} queries (monitor in analysis)")
    print(f"  Output: {VECTOR_OUTPUT}")


if __name__ == "__main__":
    main()