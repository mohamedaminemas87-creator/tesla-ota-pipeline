# Tesla OTA Update Pipeline 🚗

**Bachelor Thesis in Informatics — Automated Data Collection and Machine Learning Pipeline for Empirical Research on Tesla's Software Updates**

## Overview

This repository contains the full pipeline for collecting, processing, and classifying Tesla Model 3 Over-the-Air (OTA) software update release notes. The coding scheme (vehicle domains, function types, coordination categories) follows the supervisor's dissertation specification; this repository implements the automated data collection and machine learning pipeline that applies it at scale.

## Pipeline Architecture

```
Nicole's MASTER_LIST (142 branches)
      ↓
scrape_candidates.xlsx              ← candidate build list per branch
      ↓
tesla_scraping_pipeline_v4.py       ← scraping (per-feature extraction)
      ↓
tesla_scraped_v4_branches_final.xlsx   (142 branches: dates, data_availability)
functions_final_p2.xlsx                (2002 individual functions: heading + body)
      ↓
Tesla_P2_Final_Light.ipynb          ← BERT training (Google Colab, GPU)
      ↓
3 BERT models (domain, function_type, coordination) — heading+body input
      ↓
p2_comparison_results.xlsx          ← heading-only vs heading+body comparison
      ↓
tesla_ota_corpus_p3_ready.xlsx      ← content-quality filtering (see below)
      ↓
tesla_classifier_pipeline_v3.py     ← classification with the trained models (no retraining)
      ↓
tesla_ota_classified_v3_FINAL.xlsx/.csv   ← FINAL classified dataset (1710 functions)
      ↓
hardware_cohort_tagging.py          ← hardware-cohort tagging (uses 20260923_Hardware_Cohort_Amine.xlsx)
      ↓
tesla_ota_classified_v3_hw.xlsx     ← FINAL dataset + hardware_applicability columns
```

**Content-quality filtering** (→ `tesla_ota_corpus_p3_ready.xlsx`):
- Site commentary ("Editorial note:" / "Editor's note:") detected and stripped from the text fed to the classifiers (91 functions affected), kept separately in an `editorial_note` column
- Functions that do not apply to Model 3 (other-model-only features, detected via the site's own `models_available` metadata) removed from the corpus: 292 of 2002 functions excluded, 40 kept as unresolved/ambiguous
- Final corpus: **1710 functions**

**Evaluation and final labeling** (→ `tesla_ota_classified_v3_FINAL.xlsx`):
- REMOVED function type: too few ground-truth examples (2/185) to train on; handled by keyword rule (e.g. "removed", "discontinued") over the full corpus, followed by manual verification of every flagged case (43 candidates reviewed, 12 confirmed genuine whole-feature removals)
- Model performance reported via grouped 5-fold cross-validation (same normalized heading never in both train and test, to avoid inflated scores from near-duplicate release notes) rather than a single train/test split
- Per-prediction confidence score (softmax probability) recorded for every classification; rows with confidence below 60% on any of the 3 classifiers are flagged `needs_review` (277/1710 functions, 16.2%)

## Repository Structure

```
tesla-ota-pipeline/
├── README.md
├── requirements.txt
├── scrape_candidates.xlsx                     # sampling frame: candidate builds per branch
├── tesla_scraping_pipeline_v4.py              # scraping script (per-feature extraction)
├── tesla_scraped_v4_branches_final.xlsx       # 142 branches: dates, data_availability
├── functions_final_p2.xlsx                    # 2002 scraped functions, pre-filtering (heading + body)
├── Tesla_P2_Final_Light.ipynb                 # BERT training notebook (heading-only vs heading+body)
├── p2_comparison_results.xlsx                 # heading-only vs heading+body comparison table
├── tesla_ota_corpus_p3_ready.xlsx             # post content-quality filtering, 1710 functions, pre-classification
├── p3_corpus_cleaning.py                      # P3: strips editorial notes, filters non-Model-3 functions (2002 -> 1710)
├── tesla_classifier_pipeline_v3.py            # standalone classification script (confidence score + needs_review flag, no retraining)
├── removed_rule.py                            # keyword rule + 12 manually confirmed REMOVED functions
├── p4_cross_validation.py                     # grouped 5-fold CV + majority-class baseline (produces the two p4_*.xlsx files)
├── p4_cv_results.xlsx                         # 5-fold grouped cross-validation results + majority-class baseline
├── p4_confusion_matrices.xlsx                 # confusion matrix per classifier
├── tesla_ota_classified_v3_FINAL.xlsx         # FINAL dataset: 1710 functions, classified, with confidence + needs_review
├── tesla_ota_classified_v3_FINAL.csv          # same, CSV format
├── 20260923_Hardware_Cohort_Amine.xlsx        # hardware cohorts (HW2.5 / HW3 / AI4), provided by the supervisor
├── hardware_cohort_tagging.py                 # rule-based hardware-cohort tagging (P5)
└── tesla_ota_classified_v3_hw.xlsx            # FINAL dataset + hardware_applicability columns
```

Model weights (`*.pt`, ~440MB each) and label encoders (`*.pkl`) are not committed to this repository due to size; they can be regenerated by running `Tesla_P2_Final_Light.ipynb`.

## Data Collection

- **Sampling frame:** 142 branches (YYYY.WW) from the supervisor's cross-source MASTER_LIST, not an arbitrary sample from the site index
- **Scraping rule:** every known candidate build per branch is fetched; the most complete set of feature descriptions is merged across builds (documented, deterministic rule — see `tesla_scraping_pipeline_v4.py`)
- **Two dates per branch:** `branch_first_seen` (earliest known build) and `build_date` (date of the build(s) that contributed content)
- **`data_availability`:** FULL / MINIMAL / NO_NOTES, based on real extracted content, not a default value
- Branches with no recoverable content (mostly 2017–2019, where NotATeslaApp has no live page and no other source contains body text) are explicitly labeled NO_NOTES rather than dropped

### Content Extraction

Each release page is parsed at the individual feature level (not just page-level text):

```html
<section class="mod-update-feature" data-feature-type="official" data-feature="...">
    <h2>Feature Name</h2>
    <div class="mod-update-feature-description">Real description text...</div>
    ...model/hardware requirement tags...
</section>
```

This yields, per function: `heading`, `body`, `text_origin` (official/undocumented, taken directly from the site's own markup), and `models_available` (which Tesla models the feature applies to).

## Models Performance

Three BERT classifiers were compared on the same 135 hand-coded, ground-truth-labeled examples, using heading-only (title text only) vs heading+body (title + full description) as input:

| Classifier | Accuracy (heading only) | Accuracy (heading+body) | Macro-F1 (heading only) | Macro-F1 (heading+body) |
|---|---|---|---|---|
| Domain | 70.4% | 77.8% | 65.7% | 63.3% |
| Function Type | 55.6% | 85.2% | 55.0% | 85.2% |
| Coordination (STANDALONE/COORD_ACTIVE) | 74.1% | 81.5% | 73.9% | 81.4% |

Heading+body was chosen as the final production input for all three classifiers. See `p2_comparison_results.xlsx` for the full results and the thesis methodology chapter for discussion (domain's lower macro-F1 despite higher accuracy reflects class imbalance across the 5 domains, not a regression).

### Rigorous evaluation (5-fold grouped cross-validation)

The single-split numbers above give an initial read, but can be optimistic by chance on a small dataset. The final evaluation uses grouped 5-fold cross-validation (same heading never appears in both the training and test fold) with a majority-class baseline evaluated on the identical folds, for a fair comparison:

| Task | n | CV Accuracy | CV Macro-F1 | Baseline Accuracy | Baseline Macro-F1 | Improvement (Macro-F1) |
|---|---|---|---|---|---|---|
| Domain | 135 | 74.1% | 65.7% | 48.1% | 13.0% | +52.7 pts |
| Function Type | 133 | 81.2% | 81.2% | 44.4% | 39.1% | +42.1 pts |
| Coordination | 135 | 71.1% | 71.0% | 45.2% | 40.7% | +30.3 pts |

All three classifiers outperform a naive majority-class baseline by 30–53 Macro-F1 points, indicating the models learn genuine signal rather than exploiting class imbalance. The REMOVED function type (2/185 ground-truth examples) is excluded from training as statistically infeasible and handled separately via a keyword rule with manual verification (see above). Full per-class precision/recall/F1 and confusion matrices are in `p4_cv_results.xlsx` and `p4_confusion_matrices.xlsx`.

## Hardware Cohort Tagging

`hardware_cohort_tagging.py` adds a `hardware_applicability_pred` column to every classified function, using the supervisor's cohort file (`20260923_Hardware_Cohort_Amine.xlsx`: HW2.5, HW3, AI4). It is **rule-based, not a trained model**. Rules, in order of priority:

1. **Stated requirement:** if the site's own `hardware_requirement` field names HW3, HW4/AI4 or "HW 2 or higher", it is used directly (MCU and AMD Ryzen describe the infotainment computer, not the Autopilot cohort, and are ignored).
2. **FSD version in the heading:** FSD v13 or higher → `AI4_ONLY`; a "Lite" build → `HW3`; versions up to v12.6 → `HW3+AI4` from branch 2024.01 on (before that `HW3`).
3. **Hardware named in the text:** HW3 and HW4 mentioned together → `HW3+AI4`; "only on HW4" → `AI4_ONLY`; HW3 in a negative sentence ("HW3 vehicles are not receiving ...") is flagged for review.
4. **Release date:** branches before 2019.16 can only run on `HW2.5` (HW3 first sold April 2019).
5. **Camera/Autopilot-dependent features** (FSD, Summon, Navigate on Autopilot, traffic-light control, dashcam, blind-spot camera, ...) → `HW3` (`HW3+AI4` from 2024.01 on).
6. **Default:** `ALL`.

Output columns: `hw_fleet_cohorts`, `hardware_applicability_pred`, `hw_rule`, `hw_variant_flag` (radar/ultrasonic/"Tesla Vision" mentions; V01/V02 are exclusions inside HW3, not separate cohorts), `hw_needs_review` (ambiguous text).

Result on the 1710 functions: ALL 1137 · HW3 263 · HW3+AI4 144 · HW2.5 87 · AI4_ONLY 79.

**Validation and limits:** on the 185 hand-coded pilot examples the rules reproduce the hand-coded `hardware_applicability` in 97.8% of cases (gold "HW3" is read as "HW3 or newer"). The rules were tuned on these same examples, so this is an optimistic in-sample figure; a sample of the full corpus was checked manually, and a few errors remain.

```bash
python hardware_cohort_tagging.py tesla_ota_classified_v3_FINAL.xlsx 20260923_Hardware_Cohort_Amine.xlsx tesla_ota_classified_v3_hw.xlsx
# validation on the pilot study file:
python hardware_cohort_tagging.py --evaluate <Pilot_Study_Draft.xlsx>
```

## Dataset

- 142 branches from the MASTER_LIST sampling frame
- 2002 individual functions extracted, deduplicated within branch, cross-branch duplicate content flagged (`duplicate_of`), recurring items across versions flagged (`carried_forward`)
- 1710 functions remain after content-quality filtering (site commentary stripped, 292 non-Model-3 functions excluded) — this is the final corpus
- All 1710 functions automatically classified by the final heading+body models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Scrape Tesla release notes
```bash
python tesla_scraping_pipeline_v4.py
```
Requires `scrape_candidates.xlsx` as input (built from the MASTER_LIST; see thesis methodology for how candidate builds are derived).

### 2. Train BERT models and classify (Google Colab recommended)
1. Open `Tesla_P2_Final_Light.ipynb` in Google Colab
2. Enable GPU runtime: Runtime → Change runtime type → T4 GPU
3. Upload the required Excel files when prompted and run all cells
4. Outputs: 3 model files, 3 label encoders, and the final classified dataset

### 3. Content-quality filtering (P3)
```bash
python p3_corpus_cleaning.py functions_final_p2.xlsx tesla_ota_corpus_p3_ready.xlsx
python p3_corpus_cleaning.py --verify tesla_ota_corpus_p3_ready.xlsx   # re-derives the new columns and compares with the file
```
Strips "Editor's note:" blocks into `editorial_note`, derives `model_applicability` from `models_available` and removes functions that do not apply to Model 3. One hand correction (`2022.16_F009`) is documented in the script. The re-derived columns are identical to the committed file for all 1710 functions.

### 4. REMOVED function type
```bash
python removed_rule.py tesla_ota_classified_v3_FINAL.xlsx candidates.xlsx
python removed_rule.py --verify tesla_ota_classified_v3_FINAL.xlsx
```
Keyword candidates ("removed" / "discontinued") give 43 functions; the 12 confirmed by manual reading are listed in the script and match the committed FINAL file.

### 5. Cross-validation (P4)
```bash
python p4_cross_validation.py build Pilot_Study_Draft.xlsx tesla_ota_corpus_p3_ready.xlsx ground_truth_matched.xlsx
python p4_cross_validation.py cv ground_truth_matched.xlsx bert     # GPU, e.g. Google Colab
python p4_cross_validation.py cv ground_truth_matched.xlsx tfidf    # quick CPU check, not the thesis model
```
The results in `p4_cv_results.xlsx` / `p4_confusion_matrices.xlsx` come from the thesis run on Google Colab (135 matched examples). `build` reconstructs the matching of hand-coded examples to scraped texts and finds fewer matches (116), so re-running it will not reproduce the reported numbers exactly; it documents the procedure.

### 6. Tag hardware cohorts
See the "Hardware Cohort Tagging" section above (no GPU needed).

## Requirements

See `requirements.txt`. Main dependencies:
- Python 3.8+
- transformers (HuggingFace BERT)
- torch
- pandas
- openpyxl
- beautifulsoup4
- scikit-learn

## Coding Schema

The coding scheme (domains, function types, coordination categories) follows the supervisor's dissertation specification and is implemented, not redefined, by this pipeline.

### Domain
| Domain | Subsystems |
|---|---|
| Powertrain | BMS, Energy Management, Thermal Management |
| Chassis | Braking, Steering, Stability Control, Suspension |
| Body | Climate, Lighting, Access Control, Seating |
| HMI/Multimedia/Telematics | Display, Navigation, Connectivity, Entertainment |
| Driving Automation | Autopilot, FSD, Camera System, Sentry Mode |

### Function Type
- **NEW** — feature that did not exist before this update
- **MODIFIED** — existing feature that was improved or changed
- **REMOVED** — feature removed in this update

### Coordination
- **STANDALONE** — function affects only one subsystem
- **COORD_ACTIVE** — function requires coordination between multiple subsystems

## Author

Amine — Bachelor's Thesis in Informatics
Supervised by Nicole Wenger-Wong

## License

MIT License
