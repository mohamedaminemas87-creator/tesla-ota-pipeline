#!/usr/bin/env python3
"""
Tesla OTA Update Classifier Pipeline v3
Bachelor Thesis - Amine
============================================================
Standalone classification script. Loads the 3 already-trained
heading+body BERT models and classifies a functions file
(same schema as functions_final_p2.xlsx: branch, heading, body...)
WITHOUT retraining. No GPU required for inference (slower on CPU,
but works).

INPUT (place in the same folder before running):
    functions_final_p2.xlsx (or any file with the same columns)
    domain_headingbody_model.pt / domain_headingbody_encoder.pkl
    type_headingbody_model.pt   / type_headingbody_encoder.pkl
    coord_headingbody_model.pt  / coord_headingbody_encoder.pkl

OUTPUT:
    tesla_ota_classified_v3_headingbody.xlsx / .csv
"""

import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_FUNCTIONS = "functions_final_p2.xlsx"
OUTPUT_EXCEL = "tesla_ota_classified_v3_headingbody.xlsx"
OUTPUT_CSV = "tesla_ota_classified_v3_headingbody.csv"
MAX_LEN = 256  # heading+body needs more tokens than heading-only

# Confidence threshold for flagging uncertain predictions (NEEDS_REVIEW).
# 0.6 is above chance level for both the binary tasks (50% chance) and the
# 5-class domain task (20% chance); see thesis Methodology for rationale.
CONFIDENCE_THRESHOLD = 0.6

TASKS = {
    "domain": ("domain_headingbody_model.pt", "domain_headingbody_encoder.pkl", "domain_primary_pred"),
    "type": ("type_headingbody_model.pt", "type_headingbody_encoder.pkl", "function_type_pred"),
    "coord": ("coord_headingbody_model.pt", "coord_headingbody_encoder.pkl", "standalone_or_coord_pred"),
}

# ============================================================
# STEP 1: Load models
# ============================================================
def load_models():
    print("Loading BERT models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    loaded = {}
    for task_name, (model_file, encoder_file, out_col) in TASKS.items():
        with open(encoder_file, "rb") as f:
            le = pickle.load(f)
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=len(le.classes_)
        )
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device).eval()
        loaded[task_name] = {"model": model, "label_encoder": le, "out_col": out_col}
        print(f"  {task_name}: classes={list(le.classes_)}")

    print(f"  Device: {device}")
    return loaded, tokenizer, device

# ============================================================
# STEP 2: Predict (returns label + confidence = max softmax probability)
# ============================================================
def predict(model, le, text, tokenizer, device, max_len=MAX_LEN):
    encoding = tokenizer(
        str(text), max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
    return le.inverse_transform([pred.item()])[0], confidence.item()

# ============================================================
# STEP 3: Classify a functions dataframe
# ============================================================
def classify_functions(df, loaded, tokenizer, device):
    if "heading_body" not in df.columns:
        df["heading_body"] = df["heading"].astype(str) + ". " + df["body"].astype(str)

    needs_review_cols = []
    for task_name, task_data in loaded.items():
        print(f"Classifying: {task_name} ...")
        results = df["heading_body"].apply(
            lambda t: predict(task_data["model"], task_data["label_encoder"], t, tokenizer, device)
        )
        df[task_data["out_col"]] = results.apply(lambda x: x[0])
        conf_col = task_data["out_col"].replace("_pred", "_confidence")
        df[conf_col] = results.apply(lambda x: x[1])
        needs_review_cols.append(conf_col)

    # Flag a row for manual review if ANY of the 3 classifiers was uncertain
    df["needs_review"] = (df[needs_review_cols] < CONFIDENCE_THRESHOLD).any(axis=1)
    n_review = df["needs_review"].sum()
    print(f"\nFlagged for manual review (confidence < {CONFIDENCE_THRESHOLD:.0%} on at least one task): "
          f"{n_review} / {len(df)} ({n_review/len(df):.1%})")
    return df

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Tesla OTA Classifier Pipeline v3 (heading+body)")
    print("=" * 60)

    loaded, tokenizer, device = load_models()

    print(f"\nLoading {INPUT_FUNCTIONS} ...")
    df = pd.read_excel(INPUT_FUNCTIONS, dtype={"branch": str})
    print(f"  {len(df)} functions to classify")

    df = classify_functions(df, loaded, tokenizer, device)

    df.to_excel(OUTPUT_EXCEL, index=False)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nDomain:\n{df['domain_primary_pred'].value_counts()}")
    print(f"\nFunction type:\n{df['function_type_pred'].value_counts()}")
    print(f"\nCoordination:\n{df['standalone_or_coord_pred'].value_counts()}")
    print(f"\n\u2705 Saved {OUTPUT_EXCEL} and {OUTPUT_CSV}")
