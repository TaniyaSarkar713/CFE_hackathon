"""
check_embeddings.py
===================
Checks which of your papers exist in Semantic Scholar and have SPECTER2 embeddings.
Auto-reads paper IDs from ../Data/Paper_summaries/ — no hardcoding needed.

Run from the same folder as semantic_scholar_module.py:
  python check_embeddings.py
  python check_embeddings.py --api-key YOUR_KEY
"""

import os
import glob
import time
import argparse
import requests

# ---------------------------------------------------------------------------
# Paths — Data/ is one level up from this file
# ---------------------------------------------------------------------------
_HERE         = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.dirname(_HERE)
SUMMARIES_DIR = os.path.join(_ROOT, "Data", "Paper_summaries")


def get_paper_ids(summaries_dir: str):
    """Auto-detect paper IDs from the Paper_summaries folder."""
    pattern = os.path.join(summaries_dir, "*_future_directions.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"No JSON files found in: {summaries_dir}")
        print("Check that the path is correct.")
        return []
    ids = [os.path.basename(f).replace("_future_directions.json", "") for f in files]
    return ids


def check_paper(pid: str, headers: dict, delay: float = 3.0):
    """Query S2 for one paper and report title + embedding availability."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/CorpusID:{pid}"
    try:
        r = requests.get(url, params={"fields": "title,year,citationCount,embedding"},
                         headers=headers, timeout=15)
        if r.status_code == 200:
            data      = r.json()
            emb_field = data.get("embedding") or {}
            has_emb   = bool(emb_field.get("vector"))
            title     = (data.get("title") or "?")[:60]
            year      = data.get("year", "?")
            cites     = data.get("citationCount", "?")
            status    = "EMB=YES" if has_emb else "EMB=NO "
            print(f"  FOUND   [{status}] {year} | cites={cites:<5} | {pid} | {title}")
            return {"found": True, "has_embedding": has_emb, "title": title}
        elif r.status_code == 404:
            print(f"  NOT FOUND                              | {pid}")
            return {"found": False, "has_embedding": False}
        elif r.status_code == 429:
            print(f"  RATE LIMITED — increase delay          | {pid}")
            time.sleep(delay * 3)
            return None
        else:
            print(f"  HTTP {r.status_code}                  | {pid}")
            return None
    except requests.RequestException as e:
        print(f"  ERROR: {e}  | {pid}")
        return None
    finally:
        time.sleep(delay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries-dir", default=SUMMARIES_DIR,
                   help=f"Path to Paper_summaries folder (default: {SUMMARIES_DIR})")
    p.add_argument("--api-key", default=None,
                   help="S2 API key (optional but reduces rate-limit risk)")
    p.add_argument("--delay",   type=float, default=3.0,
                   help="Seconds between requests (default 3.0, increase if rate-limited)")
    args = p.parse_args()

    headers = {"x-api-key": args.api_key} if args.api_key else {}

    paper_ids = get_paper_ids(args.summaries_dir)
    if not paper_ids:
        return

    print(f"\nChecking {len(paper_ids)} papers against Semantic Scholar...")
    print(f"Summaries dir : {args.summaries_dir}")
    print(f"Delay         : {args.delay}s between requests")
    print(f"API key       : {'yes' if args.api_key else 'no (unauthenticated)'}")
    print("-" * 70)

    found, has_emb, not_found = [], [], []

    for pid in paper_ids:
        result = check_paper(pid, headers, delay=args.delay)
        if result is None:
            continue
        if result["found"]:
            found.append(pid)
            if result["has_embedding"]:
                has_emb.append(pid)
        else:
            not_found.append(pid)

    print("-" * 70)
    print(f"\nSummary:")
    print(f"  Total checked   : {len(paper_ids)}")
    print(f"  Found in S2     : {len(found)}")
    print(f"  Has embedding   : {len(has_emb)}  <- graph signals will work for these")
    print(f"  Not found in S2 : {len(not_found)}")

    if has_emb:
        print(f"\nPapers with embeddings (use these for --paper-id demo):")
        for pid in has_emb:
            print(f"  {pid}")

    if not_found:
        print(f"\nPapers not found in S2 (CorpusIDs may not match S2's index):")
        for pid in not_found:
            print(f"  {pid}")
        print("\nTip: if most papers are not found, your IDs may be from a different")
        print("system. Try searching by paper title instead via --search-by-title.")


if __name__ == "__main__":
    main()