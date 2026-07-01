"""
Customer Churn Analysis
========================
End-to-end pipeline: data cleaning -> exploratory data analysis ->
churn-driver identification -> predictive modeling.

Dataset: telecom_customer_churn.csv (5,040 raw rows, 15 columns)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

OUT = "outputs"
# ============================================================
# 1. LOAD
# ============================================================
raw = pd.read_csv("data/telecom_customer_churn.csv")
print(f"Raw shape: {raw.shape}")

report_lines = []
report_lines.append(f"- Raw records loaded: **{raw.shape[0]}** rows, **{raw.shape[1]}** columns")

# ============================================================
# 2. CLEAN
# ============================================================
df = raw.copy()

# 2a. Strip whitespace / normalize casing on object columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()
    df.loc[df[col] == "nan", col] = np.nan

df["Contract"] = df["Contract"].str.title()

# 2b. Fix TotalCharges: stored as string with stray spaces -> numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# 2c. Drop exact duplicate rows
n_before = len(df)
df = df.drop_duplicates(subset=[c for c in df.columns if c != "customerID"])
n_dupes = n_before - len(df)
report_lines.append(f"- Duplicate rows removed: **{n_dupes}**")

# 2d. Handle missing values
missing_summary = df.isna().sum()
missing_summary = missing_summary[missing_summary > 0]
report_lines.append(f"- Columns with missing values before imputation: **{len(missing_summary)}** "
                     f"({', '.join(missing_summary.index)})")

# Numeric: median impute TotalCharges (skewed distribution)
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Categorical: mode impute
for col in ["gender", "Partner"]:
    df[col] = df[col].fillna(df[col].mode()[0])

# 2e. Sanity clip
df["tenure"] = df["tenure"].clip(lower=0)
df["MonthlyCharges"] = df["MonthlyCharges"].clip(lower=0)

report_lines.append(f"- Final cleaned shape: **{df.shape[0]}** rows, **{df.shape[1]}** columns")
report_lines.append(f"- Missing values after cleaning: **{df.isna().sum().sum()}**")

df.to_csv(f"{OUT}/cleaned_churn_data.csv", index=False)

# ============================================================
# 3. EXPLORATORY DATA ANALYSIS
# ============================================================
overall_churn_rate = (df["Churn"] == "Yes").mean() * 100
report_lines.append(f"- **Overall churn rate: {overall_churn_rate:.1f}%**")

# --- 3a. Churn rate by contract type ---
fig, ax = plt.subplots(figsize=(6, 4))
rate = df.groupby("Contract")["Churn"].apply(lambda x: (x == "Yes").mean() * 100).sort_values(ascending=False)
sns.barplot(x=rate.index, y=rate.values, hue=rate.index, palette="Reds_r", ax=ax, legend=False)
ax.set_ylabel("Churn Rate (%)")
ax.set_title("Churn Rate by Contract Type")
for i, v in enumerate(rate.values):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/churn_by_contract.png")
plt.close()

top_contract = rate.index[0]
report_lines.append(f"- Highest-risk contract type: **{top_contract}** ({rate.iloc[0]:.1f}% churn) "
                     f"vs **{rate.index[-1]}** ({rate.iloc[-1]:.1f}% churn)")

# --- 3b. Tenure distribution by churn ---
fig, ax = plt.subplots(figsize=(6, 4))
sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, common_norm=False, alpha=0.4, ax=ax)
ax.set_title("Tenure Distribution: Churned vs Retained")
plt.tight_layout()
plt.savefig(f"{OUT}/tenure_by_churn.png")
plt.close()

median_tenure_churn = df.loc[df["Churn"] == "Yes", "tenure"].median()
median_tenure_stay = df.loc[df["Churn"] == "No", "tenure"].median()
report_lines.append(f"- Median tenure — churned customers: **{median_tenure_churn:.0f} months**, "
                     f"retained customers: **{median_tenure_stay:.0f} months**")

# --- 3c. Monthly charges by churn ---
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", palette="Set2", ax=ax, legend=False)
ax.set_title("Monthly Charges by Churn Status")
plt.tight_layout()
plt.savefig(f"{OUT}/charges_by_churn.png")
plt.close()

# --- 3d. Churn rate by tech support / internet service ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col in zip(axes, ["TechSupport", "InternetService"]):
    r = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    sns.barplot(x=r.index, y=r.values, hue=r.index, palette="Blues_r", ax=ax, legend=False)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title(f"Churn Rate by {col}")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(f"{OUT}/churn_by_support_internet.png")
plt.close()

# --- 3e. Correlation heatmap (numeric + encoded key categoricals) ---
corr_df = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
corr_df["Churn"] = (df["Churn"] == "Yes").astype(int)
corr_df["Month_to_month"] = (df["Contract"] == "Month-To-Month").astype(int)
corr_df["Fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
corr_df["No_tech_support"] = (df["TechSupport"] == "No").astype(int)
corr_df["Electronic_check"] = df["PaymentMethod"].str.contains("Electronic check").astype(int)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation with Churn")
plt.tight_layout()
plt.savefig(f"{OUT}/correlation_heatmap.png")
plt.close()

top_corr = corr_df.corr()["Churn"].drop("Churn").abs().sort_values(ascending=False)
report_lines.append(f"- Top correlated factors with churn: **{', '.join(top_corr.index[:3])}**")

# ============================================================
# 4. PREDICTIVE MODEL
# ============================================================
model_df = df.copy()
le_cols = ["gender", "Partner", "Dependents", "Contract", "PaymentMethod",
           "InternetService", "TechSupport", "OnlineSecurity", "PaperlessBilling"]
encoders = {}
for col in le_cols:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

X = model_df.drop(columns=["customerID", "Churn"])
y = (model_df["Churn"] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

report_lines.append("\n### Predictive Model (Random Forest)")
report_lines.append(f"- Accuracy: **{acc*100:.1f}%**")
report_lines.append(f"- Precision: **{prec*100:.1f}%**")
report_lines.append(f"- Recall: **{rec*100:.1f}%**")
report_lines.append(f"- F1 score: **{f1*100:.1f}%**")
report_lines.append(f"- ROC-AUC: **{auc:.3f}**")

# Feature importance chart
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(8)
fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="viridis", ax=ax, legend=False)
ax.set_title("Top 8 Churn Predictors (Random Forest Feature Importance)")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png")
plt.close()

report_lines.append(f"- Top predictive features: **{', '.join(importances.index[:5])}**")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Churn Prediction Model")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curve.png")
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png")
plt.close()

# ============================================================
# 5. WRITE REPORT
# ============================================================
report = "# Customer Churn Analysis — Findings Summary\n\n" + "\n".join(report_lines) + "\n"
with open(f"{OUT}/findings_summary.md", "w") as f:
    f.write(report)

print("\n".join(report_lines))
print("\nAll outputs saved to:", OUT)
