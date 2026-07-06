"""
SupplyMind — Material Wastage Estimation Model
================================================
Predicts actual material consumption vs. blueprint estimates
for Indian construction projects.

Model: LightGBM multiclass (wastage category) + regression (wastage %)
Explainability: SHAP
Output: wastage_pct, wastage_category, cost_overrun_estimate, risk_factors
"""

import numpy as np
import pandas as pd
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, classification_report
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

MATERIAL_PRICES = {
    "TMT Steel": 62000, "OPC Cement": 380, "River Sand": 1800,
    "Coarse Aggregate": 1200, "Fly Ash Bricks": 8, "AAC Blocks": 55,
    "Structural Steel": 75000, "Electrical Cable": 2200,
    "HDPE Pipes": 450, "Vitrified Tiles": 650, "Plywood": 95, "Paint": 280,
}

def load_and_prepare(path="data/material_wastage.csv"):
    df = pd.read_csv(path)

    cat_cols = ["project_type", "state", "workforce_skill_level",
                "supervision_quality", "material_type"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    # Encode wastage category target
    cat_le = LabelEncoder()
    df["wastage_category_enc"] = cat_le.fit_transform(df["wastage_category"])
    encoders["wastage_category"] = cat_le

    FEATURES = [
        "project_type_enc", "project_size_sqft", "project_duration_months",
        "month_of_construction", "contractor_experience_yrs", "num_workers",
        "workforce_skill_level_enc", "supervision_quality_enc",
        "material_type_enc", "blueprint_quantity",
        "logistics_score", "monsoon_intensity", "monsoon_sensitivity",
    ]

    X = df[FEATURES]
    y_reg = df["wastage_pct"]
    y_clf = df["wastage_category_enc"]

    return df, X, y_reg, y_clf, FEATURES, encoders


def train_wastage_regressor(X, y_reg):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    print("\n[WASTAGE REGRESSOR] Training LightGBM wastage % model...")
    reg = lgb.LGBMRegressor(
        n_estimators=400, max_depth=7, learning_rate=0.04,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"    MAE: {mae:.2f}%")

    # Quantile models for uncertainty (80% interval)
    print("    Training quantile models for prediction intervals...")
    reg_lo = lgb.LGBMRegressor(
        objective="quantile", alpha=0.10, n_estimators=300,
        max_depth=6, learning_rate=0.05, verbose=-1
    )
    reg_hi = lgb.LGBMRegressor(
        objective="quantile", alpha=0.90, n_estimators=300,
        max_depth=6, learning_rate=0.05, verbose=-1
    )
    reg_lo.fit(X_train, y_train)
    reg_hi.fit(X_train, y_train)

    lo_pred = reg_lo.predict(X_test)
    hi_pred = reg_hi.predict(X_test)
    coverage = np.mean((y_test.values >= lo_pred) & (y_test.values <= hi_pred))
    print(f"    80% interval empirical coverage: {coverage:.1%}")

    return reg, reg_lo, reg_hi, X_test, y_test


def train_wastage_classifier(X, y_clf, encoders):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    print("\n[WASTAGE CLASSIFIER] Training LightGBM category classifier...")
    clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, subsample=0.8, random_state=42, verbose=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    labels = encoders["wastage_category"].classes_
    print(classification_report(y_test, y_pred, target_names=labels))
    return clf


def compute_shap_wastage(reg, X, feature_names):
    print("\n[SHAP] Computing wastage feature importance...")
    explainer = shap.TreeExplainer(reg)
    sample = X.iloc[:500]
    shap_values = explainer.shap_values(sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, sample, feature_names=feature_names,
                      show=False, plot_type="bar")
    plt.title("SupplyMind — Feature Importance (SHAP)\nWastage Estimation Model", fontsize=13)
    plt.tight_layout()
    plt.savefig("reports/shap_wastage.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    SHAP plot saved → reports/shap_wastage.png")
    return explainer


def predict_wastage(reg, reg_lo, reg_hi, encoders, feature_names, input_dict: dict) -> dict:
    """Single project-material wastage prediction."""
    row = input_dict.copy()
    for col in ["project_type", "state", "workforce_skill_level",
                "supervision_quality", "material_type"]:
        le = encoders[col]
        val = row.get(col, le.classes_[0])
        row[col + "_enc"] = le.transform([val])[0] if val in le.classes_ else 0

    X_input = pd.DataFrame([{f: row.get(f, 0) for f in feature_names}])

    wastage_pct    = float(np.clip(reg.predict(X_input)[0], 0.5, 60.0))
    wastage_lo     = float(np.clip(reg_lo.predict(X_input)[0], 0.0, 60.0))
    wastage_hi     = float(np.clip(reg_hi.predict(X_input)[0], wastage_pct, 80.0))
    blueprint_qty  = row.get("blueprint_quantity", 100)
    actual_qty     = blueprint_qty * (1 + wastage_pct / 100)
    price          = MATERIAL_PRICES.get(row.get("material_type", "OPC Cement"), 500)
    cost_overrun   = (actual_qty - blueprint_qty) * price

    if wastage_pct < 5:
        category, color = "Low",    "green"
    elif wastage_pct < 15:
        category, color = "Medium", "orange"
    else:
        category, color = "High",   "red"

    risk_factors = []
    if row.get("workforce_skill_level", "") in ["Unskilled", "Semi-skilled"]:
        risk_factors.append(f"Workforce is {row['workforce_skill_level'].lower()} — higher material wastage expected")
    if row.get("supervision_quality", "") in ["Poor", "Average"]:
        risk_factors.append(f"{row['supervision_quality']} site supervision increases wastage by 10–35%")
    if row.get("monsoon_intensity", 0) > 0.5 and row.get("monsoon_sensitivity", 0) > 0.5:
        risk_factors.append("Monsoon season degrades cement/sand quality, increasing wastage")
    if row.get("contractor_experience_yrs", 10) < 5:
        risk_factors.append(f"Contractor has only {row.get('contractor_experience_yrs')} yrs experience")
    if not risk_factors:
        risk_factors.append("Well-managed site — wastage within acceptable range")

    return {
        "predicted_wastage_pct": round(wastage_pct, 1),
        "wastage_range_low":     round(wastage_lo, 1),
        "wastage_range_high":    round(wastage_hi, 1),
        "actual_qty_estimate":   round(actual_qty, 2),
        "blueprint_quantity":    blueprint_qty,
        "estimated_cost_overrun_inr": round(cost_overrun, 0),
        "wastage_category":      category,
        "wastage_color":         color,
        "risk_factors":          risk_factors[:3],
    }


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SupplyMind — Training Wastage Estimation Models")
    print("="*55)

    df, X, y_reg, y_clf, feature_names, encoders = load_and_prepare()
    reg, reg_lo, reg_hi, X_test, y_test = train_wastage_regressor(X, y_reg)
    clf = train_wastage_classifier(X, y_clf, encoders)
    compute_shap_wastage(reg, X, feature_names)

    print("\n[SAVING] Saving wastage models...")
    joblib.dump(reg,          "models/wastage_regressor.pkl")
    joblib.dump(reg_lo,       "models/wastage_regressor_lo.pkl")
    joblib.dump(reg_hi,       "models/wastage_regressor_hi.pkl")
    joblib.dump(clf,          "models/wastage_classifier.pkl")
    joblib.dump(encoders,     "models/wastage_encoders.pkl")
    joblib.dump(feature_names,"models/wastage_features.pkl")
    print("    All models saved → models/")

    print("\n[TEST] Sample wastage prediction...")
    sample = {
        "project_type": "Residential Apartment", "state": "Uttar Pradesh",
        "project_size_sqft": 15000, "project_duration_months": 24,
        "month_of_construction": 7, "contractor_experience_yrs": 3,
        "num_workers": 120, "workforce_skill_level": "Unskilled",
        "supervision_quality": "Poor", "material_type": "OPC Cement",
        "blueprint_quantity": 500, "logistics_score": 0.58,
        "monsoon_intensity": 0.9, "monsoon_sensitivity": 0.7,
    }
    result = predict_wastage(reg, reg_lo, reg_hi, encoders, feature_names, sample)
    print(f"\n  Project: Residential Apartment, UP, July (Monsoon), Unskilled workforce")
    print(f"  Wastage:        {result['predicted_wastage_pct']}% (range: {result['wastage_range_low']}–{result['wastage_range_high']}%)")
    print(f"  Category:       {result['wastage_category']}")
    print(f"  Blueprint Qty:  {result['blueprint_quantity']} units")
    print(f"  Actual (est.):  {result['actual_qty_estimate']} units")
    print(f"  Cost Overrun:   ₹{result['estimated_cost_overrun_inr']:,.0f}")
    print(f"  Risk Factors:")
    for rf in result["risk_factors"]:
        print(f"    ⚠ {rf}")
    print("\n[✓] Wastage model training complete.")
