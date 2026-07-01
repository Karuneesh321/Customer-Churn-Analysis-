# Customer Churn Analysis

An end-to-end EDA + predictive modeling project on telecom customer data, built with Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

## Project structure
```
customer_churn_analysis/
├── data/
│   └── telecom_customer_churn.csv     # raw dataset (5,040 rows, intentionally messy)
├── generate_data.py                   # generates the dataset (swap for a real CSV if you have one)
├── churn_analysis.py                  # full pipeline: clean -> EDA -> model
└── outputs/
    ├── cleaned_churn_data.csv
    ├── findings_summary.md            # auto-generated write-up of results
    ├── churn_by_contract.png
    ├── tenure_by_churn.png
    ├── charges_by_churn.png
    ├── churn_by_support_internet.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── roc_curve.png
    └── confusion_matrix.png
```

## What the pipeline does
1. **Cleaning** — removes duplicate records, fixes a numeric column stored as
   messy strings, standardizes inconsistent text casing, and imputes missing
   values (median for numeric, mode for categorical).
2. **EDA** — churn rate by contract type, tenure distribution split by churn,
   monthly charges comparison, churn rate by tech support / internet service,
   and a correlation heatmap of churn against key numeric/encoded features.
3. **Modeling** — a Random Forest classifier (class-balanced) predicts churn,
   evaluated with accuracy, precision, recall, F1, and ROC-AUC, plus a feature
   importance chart, ROC curve, and confusion matrix.

## Key findings (this run)
- Overall churn rate: **29.7%**
- Month-to-month contracts churn at **37.5%** vs. **15.2%** for two-year contracts
- Churned customers have a median tenure of **12 months** vs. **20 months** for retained customers
- Strongest churn predictors: **monthly charges, tenure, total charges, contract type, internet service**
- Random Forest model: **71.9% accuracy, 0.746 ROC-AUC**

Full write-up with all numbers: `outputs/findings_summary.md`

## Using your own data instead
If you have a real dataset (e.g., the IBM Telco Customer Churn dataset from
Kaggle), just replace `data/telecom_customer_churn.csv` with your file —
as long as it has a `Churn` column (`Yes`/`No`), the rest of
`churn_analysis.py` runs unchanged. This will strengthen the project further:
you'll be able to say you analyzed a real, publicly known dataset rather than
a synthetic one.

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python generate_data.py      # or skip this and drop in your own CSV
python churn_analysis.py
```
