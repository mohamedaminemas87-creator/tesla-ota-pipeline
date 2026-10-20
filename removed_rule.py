#!/usr/bin/env python3
"""
P4 - Handling of the REMOVED function type
==========================================
Only 2 of the 185 hand-coded examples are REMOVED, too few to train a classifier. REMOVED is therefore
assigned by a keyword rule plus manual verification:

  1. CANDIDATES : every function whose heading + body_clean contains "removed" or "discontinued"
                  (whole word, case-insensitive) -> 43 candidates in the 1710-function corpus.
  2. MANUAL CHECK: each candidate was read; 12 are genuine whole-feature removals (CONFIRMED_REMOVED).
                  The other 31 were judged not to be whole-feature removals.
  3. ASSIGNMENT : the 12 confirmed functions get function_type_pred = REMOVED
                  and function_type_pred_rule_applied = True.

Usage:   python removed_rule.py tesla_ota_classified.xlsx candidates.xlsx
         -> writes the 43 candidates (with a column `confirmed`) for inspection
Verify:  python removed_rule.py --verify tesla_ota_classified_v3_FINAL.xlsx
"""
import re, sys
import pandas as pd

KEYWORDS = re.compile(r"\b(removed|discontinued)\b", re.I)

# Result of the manual verification (function_id of the 12 genuine removals)
CONFIRMED_REMOVED = [
    "2022.40_F008",  # Dashcam Viewer removed due to local regulations
    "2024.20_F012", "2024.21_F013",  # Tesla Atari games removed
    "2025.26_F004",  # Battery Health Test removed
    "2025.44_F009", "2025.45_F031", "2025.46_F027", "2025.47_F009",  # Parking Brake (display control) removed
    "2025.44_F014", "2025.45_F036", "2025.46_F032", "2025.47_F014",  # Boomerang Fu game removed
]

def candidates(df, body_col="body_clean"):
    text = df["heading"].fillna("") + ". " + df[body_col].fillna("")
    return df[text.str.contains(KEYWORDS)]

def apply_rule(df):
    df = df.copy()
    hit = df["function_id"].isin(CONFIRMED_REMOVED)
    df["function_type_pred_rule_applied"] = hit
    df.loc[hit, "function_type_pred"] = "REMOVED"
    return df

if __name__ == "__main__":
    if sys.argv[1] == "--verify":
        f = pd.read_excel(sys.argv[2], dtype=str)
        cand = candidates(f)
        print("candidates:", len(cand), "(README: 43)")
        print("confirmed inside candidates:", cand["function_id"].isin(CONFIRMED_REMOVED).sum(), "(README: 12)")
        stored = set(f.loc[f["function_type_pred_rule_applied"] == "True", "function_id"])
        print("same 12 functions as in the committed FINAL file:", stored == set(CONFIRMED_REMOVED))
        print("function_type_pred == REMOVED for all 12:", (f[f["function_id"].isin(CONFIRMED_REMOVED)]["function_type_pred"] == "REMOVED").all())
    else:
        df = pd.read_excel(sys.argv[1], dtype=str)
        c = candidates(df).copy()
        c["confirmed"] = c["function_id"].isin(CONFIRMED_REMOVED)
        c[["function_id", "branch", "heading", "body_clean", "confirmed"]].to_excel(sys.argv[2], index=False)
        print(len(c), "candidates,", int(c["confirmed"].sum()), "confirmed")
