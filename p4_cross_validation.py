#!/usr/bin/env python3
"""
P4 - Grouped 5-fold cross-validation with majority-class baseline
=================================================================
Produces p4_cv_results.xlsx and p4_confusion_matrices.xlsx (same layout as in the repository).

Design (as described in the README):
  * labelled data = hand-coded pilot functions matched to their scraped release-note text
  * input text   = heading + body_clean   (heading+body model)
  * GroupKFold(5), group = normalized heading, so the same feature never appears in train AND test
  * baseline     = always predict the most frequent class of the TRAINING fold, evaluated on the same test folds
  * metrics      = accuracy, macro-F1, confusion matrix (pooled over the 5 test folds)
  * REMOVED is excluded from training (2/185 examples); function type is therefore binary (NEW / MODIFIED)

Step 1  python p4_cross_validation.py build  Pilot_Study_Draft.xlsx tesla_ota_corpus_p3_ready.xlsx ground_truth_matched.xlsx
        Matches hand-coded functions to scraped functions (same branch year.week, fuzzy heading similarity >= 0.7).
        NOTE: this matching is a reconstruction. The original thesis run used 135 matched examples; if you still have
        that file, use it as ground_truth_matched.xlsx instead (columns: heading, heading_body_clean, domain, type, coord).
Step 2  python p4_cross_validation.py cv ground_truth_matched.xlsx [bert|tfidf]
        bert  : fine-tunes bert-base-uncased in every fold (GPU needed, run in Colab). 10 epochs, lr 2e-5,
                max_len 256 (values from the training notebook; BATCH_SIZE below must be checked against it).
        tfidf : fast sanity check without GPU (TF-IDF + logistic regression), NOT the thesis model.
"""
import re, sys, difflib
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

EPOCHS, LR, MAX_LEN, BATCH_SIZE, SEED = 10, 2e-5, 256, 8, 42   # BATCH_SIZE: check against the training notebook
TASKS = {"domain": "domain", "type": "type", "coord": "coord"}
DOMAIN_FIX = {"HMI, MULTIMEDIA, AND TELEMATICS": "HMI/Multimedia/Telematics"}

norm = lambda s: re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
clean = lambda s: str(s).replace("\t", "").strip()

def build(pilot_xlsx, corpus_xlsx, out_xlsx, threshold=0.7):
    g = pd.read_excel(pilot_xlsx, sheet_name="INTERMEDIARY_FUNCTIONAL_CODING", dtype=str)
    g = g[g["function_id"].notna()].copy()
    c = pd.read_excel(corpus_xlsx, dtype=str)
    g["br"] = g["update_name"].str.extract(r"^(\d{4}\.\d{1,2})")[0]
    c["hn"] = c["heading"].map(norm)
    rows = []
    for _, r in g.iterrows():
        cand = c[c["branch"] == r["br"]]
        if cand.empty: continue
        hn = norm(r["function_name"])
        scores = cand["hn"].map(lambda h: difflib.SequenceMatcher(None, hn, h).ratio())
        if scores.max() < threshold: continue
        m = cand.loc[scores.idxmax()]
        dom = clean(r["domain_primary"]); dom = DOMAIN_FIX.get(dom, dom)
        rows.append({"function_id": r["function_id"], "heading": m["heading"],
                     "heading_body_clean": m["heading_body_clean"], "domain": dom,
                     "type": clean(r["function_type"]), "coord": clean(r["standalone_or_coord"])})
    out = pd.DataFrame(rows).drop_duplicates("function_id")
    out.to_excel(out_xlsx, index=False)
    print(f"{len(out)} of {len(g)} hand-coded functions matched (original thesis run: 135)")

def tfidf_fold(Xtr, ytr, Xte):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    v = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    return LogisticRegression(max_iter=2000).fit(v.fit_transform(Xtr), ytr).predict(v.transform(Xte))

def bert_fold(Xtr, ytr, Xte):
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    from sklearn.preprocessing import LabelEncoder
    torch.manual_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BertTokenizer.from_pretrained("bert-base-uncased")
    le = LabelEncoder().fit(ytr)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(le.classes_)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    enc = lambda t: tok(list(t), max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
    tr, y = enc(Xtr), torch.tensor(le.transform(ytr))
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            out = model(input_ids=tr["input_ids"][idx].to(dev), attention_mask=tr["attention_mask"][idx].to(dev), labels=y[idx].to(dev))
            out.loss.backward(); opt.step(); opt.zero_grad()
    model.eval(); te = enc(Xte); preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), 16):
            lg = model(input_ids=te["input_ids"][i:i + 16].to(dev), attention_mask=te["attention_mask"][i:i + 16].to(dev)).logits
            preds += lg.argmax(1).cpu().tolist()
    return le.inverse_transform(preds)

def cv(gt_xlsx, model="bert"):
    fold_fn = bert_fold if model == "bert" else tfidf_fold
    d = pd.read_excel(gt_xlsx, dtype=str)
    rows, mats = [], {}
    for task, col in TASKS.items():
        t = d[d[col].notna() & (d[col] != "REMOVED")].reset_index(drop=True)
        X, y, groups = t["heading_body_clean"].values, t[col].values, t["heading"].map(norm).values
        pred, base = np.empty(len(t), dtype=object), np.empty(len(t), dtype=object)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            pred[te] = fold_fn(X[tr], y[tr], X[te])
            base[te] = pd.Series(y[tr]).value_counts().idxmax()
        acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
        bacc, bf1 = accuracy_score(y, base), f1_score(y, base, average="macro")
        rows.append({"Task": task, "n": len(t), "CV Accuracy": f"{acc:.1%}", "CV Macro-F1": f"{f1:.1%}",
                     "Baseline Accuracy": f"{bacc:.1%}", "Baseline Macro-F1": f"{bf1:.1%}",
                     "Improvement over baseline (Macro-F1)": f"{(f1 - bf1) * 100:+.1f} pts"})
        labels = sorted(set(y))
        mats[task] = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)
    pd.DataFrame(rows).to_excel("p4_cv_results.xlsx", index=False)
    with pd.ExcelWriter("p4_confusion_matrices.xlsx") as w:
        for k, m in mats.items(): m.to_excel(w, sheet_name=k)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    if sys.argv[1] == "build": build(*sys.argv[2:5])
    else: cv(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "bert")
