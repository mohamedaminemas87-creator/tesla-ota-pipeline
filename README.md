# Tesla OTA Update Pipeline 🚗

**Bachelor Thesis in Informatics — Automated Data Collection and Machine Learning Pipeline for Empirical Research on Tesla's Software Updates**

---

## Overview

This repository contains the full pipeline for collecting, processing, and classifying Tesla Model 3 Over-the-Air (OTA) software update release notes. The pipeline was developed as part of a Bachelor's thesis examining how post-sale software updates change coordination between vehicle subsystems, using Baldwin & Clark's modularity theory as the theoretical framework.

## Pipeline Architecture

```
notateslaapp.com
      ↓
tesla_scraping_pipeline.py    ← Web scraping
      ↓
20260525_Amine_First_Scrap.xlsx   ← Raw dataset (60 versions)
      ↓
Tesla_BERT_Pipeline_v2.ipynb  ← BERT training (Google Colab)
      ↓
3 BERT Models (domain, type, coordination)
      ↓
tesla_classifier_pipeline_v2.py  ← Auto-classification
      ↓
tesla_ota_classified_v2.csv  ← Final classified dataset (765 functions)
```

## Repository Structure

```
tesla-ota-pipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── tesla_scraping_pipeline.py         # Web scraping script
├── tesla_classifier_pipeline_v2.py   # ML classification pipeline
├── Tesla_BERT_Pipeline_v2.ipynb      # BERT training notebook (Google Colab)
└── tesla_ota_classified_v2.csv       # Final classified dataset
```

## Models Performance

Three BERT-based classifiers were trained on 185 manually coded functions:

| Classifier | Task | Accuracy |
|-----------|------|----------|
| Domain Classifier | Powertrain / Chassis / Body / HMI / Driving Automation | 75.68% |
| Function Type Classifier | NEW / MODIFIED / REMOVED | 83.78% |
| Coordination Classifier | STANDALONE / COORD_ACTIVE | 72.73% |

## Dataset

- **60 Tesla Model 3 software versions** scraped from notateslaapp.com (2019-2026)
- **185 manually coded functions** used as training data
- **765 functions** automatically classified by the BERT pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Scrape Tesla release notes
```bash
python tesla_scraping_pipeline.py
```

### 2. Train BERT models (Google Colab recommended)
- Open `Tesla_BERT_Pipeline_v2.ipynb` in Google Colab
- Enable GPU runtime: Runtime → Change runtime type → T4 GPU
- Upload your Excel files and run all cells

### 3. Run classification pipeline
```bash
python tesla_classifier_pipeline_v2.py
```

Output: `tesla_ota_classified_v2.xlsx` and `tesla_ota_classified_v2.csv`

## Requirements

See `requirements.txt` for full list. Main dependencies:
- Python 3.8+
- transformers (HuggingFace BERT)
- torch
- pandas
- openpyxl
- beautifulsoup4
- scikit-learn

## Coding Schema

Functions are classified according to the following schema:

### Domain
| Domain | Subsystems |
|--------|-----------|
| Powertrain | BMS, Energy Management, Thermal Management |
| Chassis | Braking, Steering, Stability Control, Suspension |
| Body | Climate, Lighting, Access Control, Seating |
| HMI/Multimedia/Telematics | Display, Navigation, Connectivity, Entertainment |
| Driving Automation | Autopilot, FSD, Camera System, Sentry Mode |

### Function Type
- **NEW** → Feature that did not exist before this update
- **MODIFIED** → Existing feature that was improved or changed
- **REMOVED** → Feature that was removed in this update

### Coordination
- **STANDALONE** → Function affects only one subsystem
- **COORD_ACTIVE** → Function requires coordination between multiple subsystems

## Data Sources

| Source | Type | Reliability |
|--------|------|-------------|
| NotATeslaApp (S004) | Community Website | MEDIUM |
| TeslaFi (S003) | Community Website | HIGH |
| Tessie (S006) | Live Tracker | HIGH |
| ev-fw.com (S011) | Firmware Tracker | MEDIUM |

## Author

**Amine** — Bachelor's Thesis in Informatics  
Supervised by Nicole Wenger

## License

MIT License
