# ML-Driven EV Charging Optimization for Kenya's Power Grid

## Overview
A machine learning classification model that predicts EV charging priority (Low / Medium / High) to help Kenya Power (KPLC) manage grid stability and reduce peak demand costs. Built as a proof-of-concept for integration with Geviton's SafiVolt battery management system.

**Stakeholders:** Kenya Power & Lighting Company (KPLC), Geviton ,Enzi
**Model:** Calibrated Logistic Regression | **Accuracy:** 98.5% | **F1-Macro:** 0.985

---

## Business Problem
Unregulated EV charging causes peak-load spikes on Kenya's grid. Kenya's evening peak hit 2,288 MW in October 2024 against an installed capacity of 3,199 MW — leaving only 28% headroom. E-mobility consumption grew 300% YoY (EPRA FY2024/25). This model enables demand-side management by automatically deferring low-priority charging to off-peak hours while ensuring high-priority vehicles (commercial fleets, emergency) charge immediately.

---

## Results Summary
| Model | Accuracy | Notes |
|---|---|---|
| Logistic Regression (baseline) | 97.0% | Linear relationship captured well |
| Random Forest | 87.7% | Underperformed baseline |
| XGBoost | 94.3% | Strong but below LR |
| Decision Tree (tuned) | 92.0% | GridSearchCV tuned |
| **Calibrated LR (final)** | **98.5%** | Isotonic calibration, F1-macro 0.985 |

---

## Repository Structure
├── EV_Charging_Optimization_Final.ipynb  # Main analysis notebook (CRISP-DM)
├── ev_charging_data.csv                  # Dataset (Kaggle — see Data Source below)
├── presentation.pdf                      # Non-technical stakeholder slides
├── final_logistic_pipeline_calibrated.pkl # Saved deployment-ready model pipeline
├── README.md                             # This file
└── .gitignore
---

## How to Run
1. Clone the repository
```bash
   git clone <your-repo-url>
   cd <repo-folder>
```
2. Install dependencies
```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
```
3. Place `ev_charging_data.csv` in the root directory (see Data Source below)
4. Open and run `EV_Charging_Optimization_Final.ipynb` top to bottom
   - `Kernel → Restart & Run All`

---

## Data Source
Dataset: **EV Charging & Grid Optimization Data** — Kaggle  
🔗 [https://www.kaggle.com/datasets/](https://www.kaggle.com/datasets/) ← paste your actual Kaggle link here

- 2,000 rows, 11 features, 3-class target (`charging_priority`: 0=Low, 1=Medium, 2=High)
- Balanced classes: Low 33.8%, Medium 35.5%, High 30.8%
- Key features: `initial_soc`, `station_load`, `electricity_price`, `renewable_energy_ratio`, `queue_length`
- **Limitation:** Synthetic dataset — real deployment should integrate live KPLC load feeds and BasiGo fleet telemetry (see Bonus section in notebook)

---

## Links
-  Presentation: [presentation.pdf](./presentation.pdf)
-  Notebook: [EV_Charging_Optimization_Final.ipynb](./EV_Charging_Optimization_Final.ipynb)
-  Real-world data sources used in bonus section:
  - EPRA FY2024/25 Statistics: https://www.epra.go.ke/statistics-0
  - KPLC Tariff Tracker: https://www.stimatracker.com