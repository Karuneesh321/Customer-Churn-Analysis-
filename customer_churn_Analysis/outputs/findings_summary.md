# Customer Churn Analysis — Findings Summary

- Raw records loaded: **5040** rows, **15** columns
- Duplicate rows removed: **40**
- Columns with missing values before imputation: **3** (gender, Partner, TotalCharges)
- Final cleaned shape: **5000** rows, **15** columns
- Missing values after cleaning: **0**
- **Overall churn rate: 29.7%**
- Highest-risk contract type: **Month-To-Month** (37.5% churn) vs **Two Year** (15.2% churn)
- Median tenure — churned customers: **12 months**, retained customers: **20 months**
- Top correlated factors with churn: **MonthlyCharges, Fiber_optic, tenure**

### Predictive Model (Random Forest)
- Accuracy: **71.9%**
- Precision: **52.2%**
- Recall: **64.0%**
- F1 score: **57.5%**
- ROC-AUC: **0.746**
- Top predictive features: **MonthlyCharges, tenure, TotalCharges, Contract, InternetService**
