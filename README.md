# Hospital Patient Readmission Prediction

Predicts whether a diabetic patient will be readmitted to hospital **within 30 days** of discharge, using the UCI *Diabetes 130-US Hospitals (1999–2008)* dataset.

**Tech:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SQL (SQLite)

## Pipeline
1. **Cleaning:** dropped high-missing columns (weight, payer code, medical specialty…), removed patients who died or went to hospice, removed invalid rows
2. **Deduplication:** kept each patient's first encounter so the records are independent
3. **SQL analysis:** readmission rate by age, diabetes medication and prior inpatient visits
4. **EDA:** primary-diagnosis mix by age group, readmission drivers
5. **Feature selection:** Random Forest feature importance (128 encoded variables → top 20)
6. **Model:** Decision Tree with `class_weight='balanced'`, depth chosen by F2 score (weights recall higher than precision)

## Results
| Metric | Value |
|---|---|
| Raw encounters | 101,766 |
| After cleaning | 97,090 |
| Unique patients (after dedup) | 68,158 |
| Features | 128 → 20 |
| Recall (readmitted <30 days) | **55%** |
| Accuracy | 60% |

## Key insights
- Readmission rises with age: ~7% for patients under 60 vs **10.3–10.8% for ages 70–90**
- Prior inpatient visits matter most clinically: 0 visits → 8.2% readmitted, 3+ visits → **23%+**
- Patients on diabetes medication: 9.5% readmitted vs 7.6% for patients not on it
- Diabetes is the main primary diagnosis for patients under 30; circulatory disease takes over after 50
- Top predictors: number of lab procedures, number of medications, time in hospital, age, number of diagnoses

## How to run
```bash
pip install -r ../requirements.txt
# download diabetic_data.csv from https://archive.ics.uci.edu/dataset/296 into data/
jupyter notebook readmission_prediction.ipynb
```
Charts are saved to `figures/`.
