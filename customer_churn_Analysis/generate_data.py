"""
generate_data.py
Generates a realistic synthetic telecom customer dataset (~5000 customers)
with built-in churn drivers, mimicking the structure of the well-known
IBM Telco Customer Churn dataset. Includes intentional messiness
(duplicates, missing values, inconsistent formatting) so the cleaning
step in the analysis is genuine, not decorative.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 5000

customer_id = [f"CUST-{10000+i}" for i in range(N)]
gender = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.3, 0.7])

tenure = np.random.exponential(scale=24, size=N).astype(int)
tenure = np.clip(tenure, 0, 72)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.24, 0.21]
)
payment_method = np.random.choice(
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)
internet_service = np.random.choice(
    ["Fiber optic", "DSL", "No"], N, p=[0.44, 0.34, 0.22]
)
tech_support = np.where(
    internet_service == "No", "No internet service",
    np.random.choice(["Yes", "No"], N, p=[0.29, 0.71])
)
online_security = np.where(
    internet_service == "No", "No internet service",
    np.random.choice(["Yes", "No"], N, p=[0.29, 0.71])
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])

monthly_charges = np.round(
    np.where(internet_service == "Fiber optic", np.random.normal(85, 15, N),
    np.where(internet_service == "DSL", np.random.normal(55, 12, N),
             np.random.normal(25, 8, N))), 2
)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
total_charges = np.round(monthly_charges * tenure * np.random.uniform(0.95, 1.05, N), 2)

# ---- Build churn probability from realistic, interpretable drivers ----
logit = (
    -3.0
    + 1.35 * (contract == "Month-to-month")
    + 0.55 * (contract == "One year")
    - 0.03 * tenure
    + 0.012 * monthly_charges
    + 0.65 * (internet_service == "Fiber optic")
    + 0.55 * (tech_support == "No")
    + 0.35 * (online_security == "No")
    + 0.45 * (payment_method == "Electronic check")
    + 0.25 * (paperless_billing == "Yes")
    + 0.20 * senior_citizen
    - 0.30 * (partner == "Yes")
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = np.where(np.random.uniform(0, 1, N) < prob_churn, "Yes", "No")

df = pd.DataFrame({
    "customerID": customer_id,
    "gender": gender,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "Contract": contract,
    "PaymentMethod": payment_method,
    "InternetService": internet_service,
    "TechSupport": tech_support,
    "OnlineSecurity": online_security,
    "PaperlessBilling": paperless_billing,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

# ---- Inject realistic messiness for the cleaning step ----
# 1. Missing values in TotalCharges (common real-world issue for new customers)
missing_idx = np.random.choice(df.index, size=60, replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan

# 2. Some missing values in gender / Partner
for col, n in [("gender", 25), ("Partner", 20)]:
    idx = np.random.choice(df.index, size=n, replace=False)
    df.loc[idx, col] = np.nan

# 3. Inconsistent text casing/whitespace
df.loc[df.sample(150, random_state=1).index, "Contract"] = df["Contract"].str.upper()
df["PaymentMethod"] = df["PaymentMethod"].apply(
    lambda x: f"  {x} " if np.random.rand() < 0.05 else x
)

# 4. Duplicate rows
dupes = df.sample(40, random_state=2)
df = pd.concat([df, dupes], ignore_index=True)

# 5. TotalCharges stored as string with stray spaces (mimics real dataset quirk)
df["TotalCharges"] = df["TotalCharges"].apply(
    lambda x: f" {x}" if pd.notna(x) and np.random.rand() < 0.1 else x
)

df = df.sample(frac=1, random_state=3).reset_index(drop=True)
df.to_csv("data/telecom_customer_churn.csv", index=False)
print("Generated dataset:", df.shape)
print(df["Churn"].value_counts(normalize=True))
