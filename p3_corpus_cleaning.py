#!/usr/bin/env python3
"""
P3 - Content-quality filtering (reproduces tesla_ota_corpus_p3_ready.xlsx)
=========================================================================
Input : functions_final_p2.xlsx  (2002 scraped functions; columns function_id, branch, heading, body,
        heading_body, text_origin, models_available, hardware_requirement, source_build,
        carried_forward, duplicate_of)
Output: tesla_ota_corpus_p3_ready.xlsx (1710 functions)

Step 1  Editorial commentary: the site inserts blocks starting with "Editor's note:" /
        "Editorial note:" into the body. Everything from that marker on is moved to
        `editorial_note`; the text before it is `body_clean`. The classifiers only see
        heading + body_clean (`heading_body_clean`).
Step 2  Model applicability from the site's own `models_available` field:
        APPLIES   : Model 3 is listed ("3", "New3", "1st Gen3", "Model 3")
        AMBIGUOUS : field empty or not model-specific ("Select 2021 Models", "Features Depend On Model")
                    -> kept (40 functions)
        EXCLUDED  : other models only -> removed from the corpus (292 functions)
Step 3  One manual correction (2022.16_F009, a generic "Note:" that is editorial) - see MANUAL_EDITORIAL.

Usage:   python p3_corpus_cleaning.py functions_final_p2.xlsx tesla_ota_corpus_p3_ready.xlsx
Verify:  python p3_corpus_cleaning.py --verify tesla_ota_corpus_p3_ready.xlsx
         (re-derives the new columns from the original ones and compares them with the file)
"""
import re, sys
import pandas as pd

EDITOR_MARK = re.compile(r"\n\s*Edit(?:or|orial)(?:’|')?s?\s+[Nn]ote\s*:\s*")
AMBIGUOUS_VALUES = {"Select 2021 Models", "Features Depend On Model"}
MODEL3 = re.compile(r"(?<![A-Za-z0-9])3(?![0-9])|Gen3|New3")

# function_id -> text marker after which the body is editorial (found by manual sampling of the 164 generic "Note:" cases)
MANUAL_EDITORIAL = {"2022.16_F009": "\nNote: This feature appears to be in testing"}

def split_editorial(function_id, body):
    b = body if isinstance(body, str) else ""
    if function_id in MANUAL_EDITORIAL:
        k = b.find(MANUAL_EDITORIAL[function_id])
        if k >= 0:
            note = b[k:].split("\n")
            return (b[:k] + "\n" + "\n".join(note[2:])).strip(), note[1].strip()
    m = EDITOR_MARK.search(b)
    if m:
        return b[:m.start()].strip(), b[m.end():].strip()
    return b.strip(), None

def model_applicability(models_available):
    if not isinstance(models_available, str) or not models_available.strip():
        return "AMBIGUOUS"
    if models_available.strip() in AMBIGUOUS_VALUES:
        return "AMBIGUOUS"
    return "APPLIES" if MODEL3.search(models_available) else "EXCLUDED"

def add_columns(df):
    parts = [split_editorial(i, b) for i, b in zip(df["function_id"], df["body"])]
    df["body_clean"] = [p[0] for p in parts]
    df["editorial_note"] = [p[1] for p in parts]
    df["heading_body_clean"] = df["heading"].fillna("").astype(str) + ". " + df["body_clean"].fillna("")
    df["model_applicability"] = df["models_available"].map(model_applicability)
    return df

def verify(path):
    ref = pd.read_excel(path, dtype=str)
    new = add_columns(ref[[c for c in ref.columns if c not in ("body_clean", "editorial_note", "heading_body_clean", "model_applicability")]].copy())
    for col in ("body_clean", "editorial_note", "heading_body_clean", "model_applicability"):
        a = new[col].fillna("").astype(str).str.strip(); b = ref[col].fillna("").astype(str).str.strip()
        print(f"{col:22s} identical rows: {(a == b).sum()} / {len(ref)}")

if __name__ == "__main__":
    if sys.argv[1] == "--verify":
        verify(sys.argv[2])
    else:
        df = pd.read_excel(sys.argv[1], dtype={"branch": str})
        n0 = len(df)
        df = add_columns(df)
        print("editorial notes stripped:", df["editorial_note"].notna().sum())
        print(df["model_applicability"].value_counts().to_dict())
        df = df[df["model_applicability"] != "EXCLUDED"].reset_index(drop=True)
        print(f"{n0} -> {len(df)} functions")
        df.to_excel(sys.argv[2], index=False)
