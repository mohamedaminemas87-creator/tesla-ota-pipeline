#!/usr/bin/env python3
"""
Tesla OTA Update Scraping Pipeline v4
Bachelor Thesis - Amine | fixes the v3 content-extraction bug
============================================================
v3 grabbed NotATeslaApp's short "Overview" feature-name list instead of
the real per-feature description. The real content lives in:

    <section class="mod-update-feature" data-feature-type="official"
              data-feature="...">
        <h2>Feature Name</h2>
        ...
        <div class="mod-update-feature-requirement-name requirement-name-model
                     requirement-name-model-available">3</div>   (repeated per model)
        <div class="mod-update-feature-description">
            <p>Real description text...</p>
        </div>
    </section>

This version extracts, per feature block:
    heading, body, text_origin (official/editorial, from data-feature-type),
    models_available (e.g. S,3,X,Y,CT), build it came from.

It keeps v3's already-validated date logic unchanged.

INPUT:
    scrape_candidates.xlsx

OUTPUT:
    tesla_scraped_v4_branches.xlsx/.csv  -- one row per branch (dates, availability)
    tesla_scraped_v4_functions.xlsx/.csv -- one row per individual feature (for P2/P3)
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

BASE_URL = "https://www.notateslaapp.com"
URL_TEMPLATE = BASE_URL + "/software-updates/version/{build}/release-notes"
INPUT_CANDIDATES = "scrape_candidates.xlsx"
OUT_BRANCHES_XLSX = "tesla_scraped_v4_branches.xlsx"
OUT_BRANCHES_CSV = "tesla_scraped_v4_branches.csv"
OUT_FUNCTIONS_XLSX = "tesla_scraped_v4_functions.xlsx"
OUT_FUNCTIONS_CSV = "tesla_scraped_v4_functions.csv"
CHECKPOINT_CSV = "tesla_scraped_v4_checkpoint.csv"
DELAY = 1.5

BOILERPLATE_PATTERNS = [r"this release contains minor bug fixes and improvements\.?"]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Connection': 'keep-alive',
}

# ============================================================
# Candidate loading (unchanged from v3)
# ============================================================
def load_candidates():
    print("Loading scrape_candidates.xlsx ...")
    df = pd.read_excel(INPUT_CANDIDATES, sheet_name="SCRAPE_CANDIDATES")
    branches = []
    for _, row in df.iterrows():
        branch = str(row["branch (YYYY.WW)"]).strip()
        raw = row["candidate_builds (largest\u2192smallest)"]
        cands = [c.strip() for c in str(raw).split(",")] if pd.notna(raw) and raw else []
        branches.append({"branch": branch, "candidates": cands})
    print(f"  {len(branches)} branches loaded, "
          f"{sum(len(b['candidates']) for b in branches)} total candidate builds")
    return branches

# ============================================================
# Date extraction (unchanged, already validated)
# ============================================================
def extract_date(soup):
    h3 = soup.find("h3", string=re.compile("Release Date", re.I))
    if not h3:
        return None
    div = h3.find_next("div", class_=re.compile("mod-update-overview-feature-description"))
    if not div:
        return None
    return div.get_text(strip=True)

def parse_date(date_str):
    if not date_str:
        return None
    for fmt in ["%B %d, %Y", "%d %B %Y", "%Y-%m-%d", "%b %d, %Y"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

# ============================================================
# NEW: real per-feature extraction
# ============================================================
def extract_features(soup):
    """Returns a list of dicts, one per real feature block on the page."""
    features = []
    sections = soup.find_all("section", class_="mod-update-feature")
    for sec in sections:
        feature_type = sec.get("data-feature-type")
        if not feature_type:
            continue  # skip generic/intro wrapper sections without this attribute

        h2 = sec.find("h2")
        if not h2:
            continue
        heading = h2.get_text(strip=True)

        desc_div = sec.find("div", class_="mod-update-feature-description")
        body = desc_div.get_text(separator="\n", strip=True) if desc_div else ""

        model_divs = sec.find_all(
            "div",
            class_=re.compile(r"requirement-name-model-available")
        )
        models = sorted({d.get_text(strip=True) for d in model_divs if d.get_text(strip=True)})

        hw_divs = sec.find_all(
            "div",
            class_=re.compile(r"requirement-name-feature(?!s)")
        )
        hw_reqs = sorted({d.get_text(strip=True) for d in hw_divs if d.get_text(strip=True)})

        if not body or len(body) < 10:
            continue  # skip empty/placeholder sections

        features.append({
            "heading": heading,
            "body": body,
            "text_origin": feature_type,          # "official" / "editorial" / etc.
            "models_available": ", ".join(models) if models else None,
            "hardware_requirement": ", ".join(hw_reqs) if hw_reqs else None,
        })
    return features

def is_boilerplate_page(soup):
    text = soup.get_text(" ", strip=True).lower()
    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, text) and len(soup.find_all("section", class_="mod-update-feature")) == 0:
            return True
    return False

# ============================================================
# Fetch one build
# ============================================================
def fetch_build(build):
    url = URL_TEMPLATE.format(build=build)
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
    except requests.exceptions.RequestException:
        return None, None, None
    if r.status_code != 200:
        return None, None, None
    soup = BeautifulSoup(r.text, "html.parser")
    date_str = extract_date(soup)
    if is_boilerplate_page(soup):
        return [], date_str, True
    features = extract_features(soup)
    return features, date_str, False

# ============================================================
# Process one branch
# ============================================================
def process_branch(branch, candidates):
    all_seen_dates = []
    contributing_builds = []
    merged = {}  # normalized_heading -> feature_dict (+ source build)

    for build in candidates:
        features, date_str, seen_but_empty = fetch_build(build)
        time.sleep(DELAY)
        if features is None:
            continue  # page unreachable
        if date_str:
            all_seen_dates.append((build, date_str))
        if not features:
            continue

        contributed = False
        for feat in features:
            key = feat["heading"].strip().lower()
            if key not in merged or len(feat["body"]) > len(merged[key]["body"]):
                feat_with_src = dict(feat)
                feat_with_src["source_build"] = build
                merged[key] = feat_with_src
            contributed = True
        if contributed:
            contributing_builds.append((build, date_str))

    branch_first_seen = None
    if all_seen_dates:
        parsed = [parse_date(d) for _, d in all_seen_dates if parse_date(d)]
        if parsed:
            branch_first_seen = min(parsed)

    build_date = None
    builds_merged_list = sorted({b for b, _ in contributing_builds})
    if contributing_builds:
        dates_parsed = [parse_date(d) for _, d in contributing_builds if parse_date(d)]
        if dates_parsed:
            build_date = max(dates_parsed)

    n_features = len(merged)
    if n_features >= 1:
        data_availability = "FULL" if n_features >= 2 else "MINIMAL"
    else:
        data_availability = "NO_NOTES"

    branch_row = {
        "branch": branch,
        "branch_first_seen": branch_first_seen.strftime("%Y-%m-%d") if branch_first_seen else None,
        "build_date": build_date.strftime("%Y-%m-%d") if build_date else None,
        "year": branch_first_seen.year if branch_first_seen else None,
        "quarter": f"Q{(branch_first_seen.month - 1)//3 + 1}" if branch_first_seen else None,
        "data_availability": data_availability,
        "builds_tried": len(candidates),
        "builds_seen": len(all_seen_dates),
        "builds_merged": ", ".join(builds_merged_list) if builds_merged_list else None,
        "n_features": n_features,
    }

    function_rows = []
    for i, (key, feat) in enumerate(merged.items(), 1):
        function_rows.append({
            "function_id": f"{branch}_F{i:03d}",
            "branch": branch,
            "heading": feat["heading"],
            "body": feat["body"],
            "heading_body": f"{feat['heading']}. {feat['body']}",
            "text_origin": feat["text_origin"],
            "models_available": feat["models_available"],
            "hardware_requirement": feat["hardware_requirement"],
            "source_build": feat["source_build"],
        })

    return branch_row, function_rows

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    branches = load_candidates()
    branch_results = []
    function_results = []

    for i, b in enumerate(branches, 1):
        print(f"[{i}/{len(branches)}] {b['branch']} ({len(b['candidates'])} candidates)...", end=" ")
        branch_row, function_rows = process_branch(b["branch"], b["candidates"])
        branch_results.append(branch_row)
        function_results.extend(function_rows)
        print(f"-> {branch_row['data_availability']} "
              f"({branch_row['n_features']} features, first_seen={branch_row['branch_first_seen']})")

        if i % 10 == 0:
            pd.DataFrame(branch_results).to_csv(CHECKPOINT_CSV, index=False)

    bdf = pd.DataFrame(branch_results)
    bdf.to_excel(OUT_BRANCHES_XLSX, index=False)
    bdf.to_csv(OUT_BRANCHES_CSV, index=False, encoding="utf-8-sig")

    fdf = pd.DataFrame(function_results)
    fdf.to_excel(OUT_FUNCTIONS_XLSX, index=False)
    fdf.to_csv(OUT_FUNCTIONS_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(bdf["data_availability"].value_counts())
    print(f"\nTotal individual features extracted: {len(fdf)}")
    print(f"text_origin distribution:\n{fdf['text_origin'].value_counts()}")
    print(f"\n\u2705 Saved {OUT_BRANCHES_XLSX}, {OUT_BRANCHES_CSV}")
    print(f"\u2705 Saved {OUT_FUNCTIONS_XLSX}, {OUT_FUNCTIONS_CSV}")
