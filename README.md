# NirmanAI — README
# Team Aim-Nexus | IIT Madras | KAYA x IIT Hackathon 2026

## Project Structure

```
nirmanai/
├── generate_data.py        # Synthetic dataset generator (India-specific)
├── train_delay_model.py    # Delay prediction: XGBoost + conformal intervals + SHAP
├── train_wastage_model.py  # Wastage estimation: LightGBM quantile + SHAP
├── app.py                  # Streamlit dashboard (main demo)
├── setup.py                # One-click setup script
├── requirements.txt        # Dependencies
├── data/                   # Generated datasets (auto-created)
│   ├── delivery_delays.csv
│   └── material_wastage.csv
├── models/                 # Trained model files (auto-created)
└── reports/                # SHAP plots (auto-created)
```

## Quick Start

```bash
# Step 1: Setup (installs deps, generates data, trains models)
python setup.py

# Step 2: Launch dashboard
streamlit run app.py
```

Dashboard runs at: http://localhost:8501

## What Each File Does

### generate_data.py
Creates two synthetic datasets calibrated against Indian construction industry benchmarks:
- `delivery_delays.csv` — 5,000 delivery records with India-specific features
  (monsoon intensity, festival periods, state logistics scores, supplier tiers)
- `material_wastage.csv` — 3,000 project-material records with workforce and site factors

### train_delay_model.py
Trains a two-stage delay prediction system:
1. XGBoost classifier — "Will this delivery be delayed?"
2. LightGBM regressor — "How many days late?"
3. Conformal prediction — calibrated confidence intervals (80% coverage)
4. SHAP explainability — human-readable risk factors per prediction

### train_wastage_model.py
Trains wastage estimation models:
1. LightGBM regressor — predicts wastage % per material per project
2. Quantile regression (10th/90th percentile) — uncertainty range
3. LightGBM classifier — Low / Medium / High wastage category
4. SHAP explainability — key drivers of wastage

### app.py
Full Streamlit dashboard with 5 tabs:
1. Live Delivery Monitor — risk alerts, delivery table, monsoon timeline, KAYA integration preview
2. Wastage Intelligence — per-material forecast, scenario analysis
3. Predict New Order — interactive prediction with gauge chart + wastage estimate
4. Analytics — state logistics heatmap, monthly trend analysis
5. Smart Procurement Plan — AI-generated week-by-week order schedule from a Bill of Quantities

## Key Design Choices

### Why Conformal Prediction?
Construction is a high-stakes domain. A point estimate ("arrives July 15")
is dangerous. Conformal prediction gives empirically calibrated intervals:
"arrives July 15 ± 3 days with 80% confidence." This is responsible AI.

### Why SHAP?
Site managers need to act, not just know. SHAP values translate model
internals into plain English: "This delivery is at risk because the supplier
has a 38% historical delay rate and monsoon intensity is 85%." Actionable.

### Why synthetic data?
We don't have access to real procurement records. But we've encoded
domain knowledge (CIDC benchmarks, IMD monsoon patterns, festival calendars,
state logistics indices) into the data generation process. The patterns
are realistic; the specific numbers are synthetic. We're honest about this.

## Tech Stack
| Component | Technology |
|---|---|
| ML Models | XGBoost, LightGBM |
| Uncertainty | Conformal Prediction (split method) |
| Explainability | SHAP (TreeExplainer) |
| Dashboard | Streamlit + Plotly |
| Data | Pandas + NumPy |
| Serialization | joblib |

## Team
- Lead: [Your Name] — project direction, pitch, domain research
- ML Engineer 1: [M2 Name] — delay model refinement
- ML Engineer 2: [M3 Name] — wastage model refinement
- Full-Stack: [M4 Name] — dashboard, integration

IIT Madras | BS Data Science | 2026
