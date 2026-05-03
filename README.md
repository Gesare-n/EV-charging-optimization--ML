# ML-Driven EV Charging Optimization for Kenya's Power Grid

> A machine learning classification model that predicts EV charging priority — **Low / Medium / High** — enabling Kenya Power (KPLC) to manage grid stability, defer low-priority loads to off-peak windows, and reduce peak demand costs. Built as a proof-of-concept for integration with Geviton's SafiVolt battery management system.

**Stakeholders:** Kenya Power & Lighting Company (KPLC) · Geviton · Enzi · BasiGo  
**Final Model:** Calibrated Logistic Regression (Isotonic) · **Accuracy: 98.5%** · **F1-Macro: 0.985**  
**Dataset:** Synthetic (Kaggle) — 2,000 rows · 11 features · 3-class target  

---

## Business Problem

Kenya's grid is under mounting pressure from rapid e-mobility growth. The evening peak hit **2,288 MW** in October 2024 against an installed capacity of **3,199 MW** — leaving only **28% headroom**. E-mobility electricity consumption grew **300% YoY** (EPRA FY2024/25), with 6,442 registered EVs as of June 2025.

Unregulated charging during peak hours risks cascading failures. This model enables **demand-side management** by:
- Deferring **Low-priority** vehicles to off-peak hours (00:00–05:00), absorbing Kenya's 668.7 GWh/yr of curtailed geothermal surplus
- Ensuring **High-priority** vehicles (commercial fleets, emergency) charge immediately
- Estimated annual savings of **~KES 19.7M** for a 76-bus fleet via KPLC's E-Mobility TOU tariff (KES 7.9 off-peak vs KES 15.8 peak)

>  **Limitation:** The dataset is synthetic and may not capture real-world Kenyan grid noise (unplanned outages, substation-level variation). Real deployment should integrate live KPLC SCADA feeds and BasiGo fleet telemetry — see the Bonus section of the notebook for grounded scenario validation using real EPRA parameters.

---

## Results Summary

| Model | Accuracy | F1-Macro | Notes |
|---|---|---|---|
| Logistic Regression (baseline) | 97.0% | 0.971 | Linear relationship captured well |
| Random Forest | 87.7% | 0.878 | Underperformed baseline |
| XGBoost | 94.3% | 0.944 | Strong but below LR |
| Decision Tree (tuned) | 92.0% | 0.921 | GridSearchCV optimised |
| **Calibrated LR (final)** | **98.5%** | **0.985** | Isotonic calibration · Nested CV F1: 0.993 |

Logistic Regression outperforms tree-based models because the relationship between `initial_soc`, `station_load`, and charging priority is largely **linear and well-separated** — complex models added variance without improving accuracy.

---

## Repository Structure

```
├── EV_Charging_Optimization_Final.ipynb   # Main analysis notebook (CRISP-DM)
├── ev_charging_data.csv                   # Dataset — see Data Source below
├── presentation.pdf                       # Non-technical stakeholder slides
├── final_logistic_pipeline_calibrated.pkl # Deployment-ready model pipeline (joblib)
├── README.md                              # This file
└── .gitignore                             # Excludes .DS_Store, __pycache__, *.ipynb_checkpoints
```

**Loading the saved model:**
```python
import joblib
model = joblib.load('final_logistic_pipeline_calibrated.pkl')
predictions = model.predict(X_new)        # Returns 0=Low, 1=Medium, 2=High
probabilities = model.predict_proba(X_new)
```

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
   ```

3. **Add the dataset** — place `ev_charging_data.csv` in the root directory (see Data Source below)

4. **Run the notebook**
   ```
   Open EV_Charging_Optimization_Final.ipynb → Kernel → Restart & Run All
   ```

---

## Data Source

**Dataset:** EV Charging & Grid Optimization Data — Kaggle  
(https://www.kaggle.com/datasets/mjawad17/ev-charging-and-grid-optimization-data) ← 

| Property | Value |
|---|---|
| Rows | 2,000 |
| Features | 11 |
| Target | `charging_priority` — 0=Low, 1=Medium, 2=High |
| Class balance | Low 33.8% · Medium 35.5% · High 30.8% |

**Key features used:**

| Feature | Role |
|---|---|
| `initial_soc` | Primary urgency signal — low SOC = higher priority |
| `station_load` | Grid stress — high load triggers deferral |
| `electricity_price` | Cost signal — charges align with off-peak windows |
| `renewable_energy_ratio` | Green charging opportunity |
| `queue_length` | Fairness / congestion signal |
| `vehicle_type` | Emergency and commercial vehicles get priority |

---

## Links

-  Presentation: [presentation.pdf](./presentation.pdf)
-  Notebook: [EV_Charging_Optimization_Final.ipynb](./EV_Charging_Optimization_Final.ipynb)

**Real-world data sources (Bonus section):**
- EPRA FY2024/25 Statistics Report: https://www.epra.go.ke/statistics-0
- KPLC Tariff Tracker: https://www.stimatracker.com
- BasiGo fleet data: https://www.basigo.africa

---

## Acknowledgements


Real-world grounding uses publicly available data from EPRA, KPLC (via stimatracker.com), and BasiGo.
