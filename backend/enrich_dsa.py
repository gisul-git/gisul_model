"""
enrich_dsa.py
=============
Processes all problems in merged_problems.json one by one.
Calls POST /api/v1/enrich-dsa on the local backend.
Saves progress after every problem — safe to stop and resume anytime.

Usage:
    python3 enrich_dsa.py

Output:
    dsa_enriched.json  — enriched problems (grows as script runs)
"""

import json
import requests
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE    = "merged_problems.json"
OUTPUT_FILE   = "dsa_enriched.json"
API_URL       = "http://localhost:9000/api/v1/enrich-dsa"
DELAY_SECONDS = 2      # gap between requests (gives GPU time to cool)
TIMEOUT       = 120    # seconds to wait for each response
# ──────────────────────────────────────────────────────────────────────────────

# Load all problems
with open(INPUT_FILE) as f:
    raw = json.load(f)

# Handle both formats: list OR {"questions": [...]}
if isinstance(raw, dict) and "questions" in raw:
    problems = raw["questions"]
elif isinstance(raw, list):
    problems = raw
else:
    raise ValueError("Unexpected JSON format in merged_problems.json")

print(f"📚 Total problems: {len(problems)}")

# Load already enriched (resume support)
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        enriched = json.load(f)
    done_ids = {str(p["problem_id"]) for p in enriched}
    print(f"▶️  Resuming — {len(enriched)} already done, {len(problems) - len(enriched)} remaining")
else:
    enriched  = []
    done_ids  = set()
    print("🆕 Starting fresh")

# Track stats
success_count = 0
fail_count    = 0
skip_count    = 0

print("\n" + "="*60)

for i, problem in enumerate(problems):
    pid   = str(problem.get("problem_id", i))
    title = problem.get("title", f"Problem {i}")

    # Skip already done
    if pid in done_ids:
        skip_count += 1
        continue

    # Skip problems with no python3 starter (can't extract signature)
    if not problem.get("code_snippets", {}).get("python3", "").strip():
        print(f"[{i+1}/{len(problems)}] ⏭️  Skipped (no python3 starter): {title}")
        skip_count += 1
        continue

    print(f"[{i+1}/{len(problems)}] ⚙️  Enriching: {title} ({problem.get('difficulty', '?')})...")

    try:
        resp = requests.post(API_URL, json=problem, timeout=TIMEOUT)

        if resp.status_code == 200:
            result = resp.json()

            # Merge enriched fields into original problem
            enriched_problem = dict(problem)
            enriched_problem["function_signature"] = result.get("function_signature", {})
            enriched_problem["public_testcases"]   = result.get("public_testcases", [])
            enriched_problem["hidden_testcases"]   = result.get("hidden_testcases", [])
            enriched_problem["starter_code"]       = result.get("starter_code", {})

            enriched.append(enriched_problem)
            done_ids.add(pid)
            success_count += 1

            # Save after every problem — crash-safe
            with open(OUTPUT_FILE, "w") as f:
                json.dump(enriched, f, indent=2)

            sig_name = result.get("function_signature", {}).get("name", "?")
            n_hidden = len(result.get("hidden_testcases", []))
            n_public = len(result.get("public_testcases", []))
            n_langs  = len(result.get("starter_code", {}))
            print(f"         ✅ fn={sig_name}, public={n_public}, hidden={n_hidden}, langs={n_langs}")

        else:
            fail_count += 1
            print(f"         ❌ HTTP {resp.status_code}: {resp.text[:200]}")

    except requests.exceptions.Timeout:
        fail_count += 1
        print(f"         ⏱️  Timeout after {TIMEOUT}s — skipping")

    except Exception as e:
        fail_count += 1
        print(f"         ❌ Error: {e}")

    # Progress summary every 50 problems
    if (i + 1) % 50 == 0:
        print(f"\n── Progress: {len(enriched)}/{len(problems)} done | ✅ {success_count} | ❌ {fail_count} | ⏭️ {skip_count} ──\n")

    time.sleep(DELAY_SECONDS)

print("\n" + "="*60)
print(f"✅ Enrichment complete!")
print(f"   Total enriched : {len(enriched)}")
print(f"   Success        : {success_count}")
print(f"   Failed         : {fail_count}")
print(f"   Skipped        : {skip_count}")
print(f"   Output file    : {OUTPUT_FILE}")