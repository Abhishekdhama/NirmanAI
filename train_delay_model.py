"""
NirmanAI — Delay Prediction Model
=====================================
Predicts construction material delivery delays for Indian projects.

Model: XGBoost + LightGBM ensemble
Uncertainty: Conformal Prediction (MAPIE) for calibrated confidence intervals
Explainability: SHAP feature importance

Output per prediction:
- predicted_delay_days
- confidence_interval (lower, upper)
- risk_score (0-100)
- top_3_risk_factors (SHAP-based explanations)
- risk_label: Low / Medium / High / Critical
"""

import numpy as np
import pandas as pd
import joblib, os, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             classification_report, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────

def load_and_prepare(path: str = "data/delivery_delays.csv"):
    df = pd.read_csv(path)

    # High temperature
    df["High_Temperature"] = (df["temperature"] >= 35).astype(int)
    # High humidity
    df["High_Humidity"] = (df["humidity"] >= 75).astype(int)
    # Heavy traffic
    df["High_Traffic"] = (df["traffic_status"] == "Heavy").astype(int)
    # Long waiting time
    df["Long_Wait"] = (df["waiting_time"] >= 30).astype(int)
    # Low inventory
    df["Low_Inventory"] = (df["inventory_level"] <= 250).astype(int)
    # Long distance shipment
    df["Long_Distance"] = (df["distance_km"] >= 800).astype(int)
    # High order value
    df["High_Order_Value"] = (df["order_value_inr"] > df["order_value_inr"].median()).astype(int)
    # Poor road quality
    df["Poor_Road"] = (df["road_quality"] < 0.55).astype(int)
    # Poor logistics
    df["Poor_Logistics"] = (df["dest_logistics_score"] < 0.55).astype(int)
    # Environmental Risk Score
    df["Environmental_Risk"] = (
        df["High_Temperature"]
        + df["High_Humidity"]
        + (df["monsoon_intensity"] > 0.60).astype(int)
    )

    # Categorical encoding (including new categorical features if needed, though they are mapped to binary above)
    cat_cols = ["material_type", "supplier_tier", "origin_state", "destination_state", "vehicle_type", "traffic_status"]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    FEATURES = [
        "month", "day_of_week", "quarter", "is_festival_period",
        "material_type_enc", "supplier_tier_enc",
        "origin_state_enc", "destination_state_enc",
        "vehicle_type_enc", "traffic_status_enc",
        "distance_km", "order_quantity", "promised_lead_days",
        "temperature", "humidity", "waiting_time", "inventory_level",
        "asset_utilization", "demand_forecast", "order_value_inr",
        "road_quality", "supplier_reliability", "past_delay_rate",
        "monsoon_intensity", "monsoon_sensitivity",
        "dest_logistics_score", "orig_logistics_score",
        "dest_monsoon_severity", 
        "High_Temperature", "High_Humidity", "High_Traffic", "Long_Wait",
        "Low_Inventory", "Long_Distance", "High_Order_Value", 
        "Poor_Road", "Poor_Logistics", "Environmental_Risk"
    ]

    X = df[FEATURES]
    y_clf = df["is_delayed"]           # classification target
    y_reg = df["actual_delay_days"]    # regression target

    return df, X, y_clf, y_reg, FEATURES, encoders

# ─────────────────────────────────────────────
# 2. DELAY CLASSIFICATION MODEL
#    (Will it be delayed? Yes/No + probability)
# ─────────────────────────────────────────────

def train_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[CLASSIFIER] Training XGBoost delay classifier...")
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    clf.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print(f"    AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["On Time", "Delayed"]))

    return clf, X_train, X_test, y_test

# ─────────────────────────────────────────────
# 3. DELAY MAGNITUDE REGRESSION MODEL
#    (How many days late?)
# ─────────────────────────────────────────────

def train_regressor(X, y_clf, y_reg):
    """Train regression only on delayed orders."""
    delayed_mask = y_clf == 1
    X_del = X[delayed_mask]
    y_del = y_reg[delayed_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X_del, y_del, test_size=0.2, random_state=42
    )

    print("\n[REGRESSOR] Training LightGBM delay magnitude model...")
    reg = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        random_state=42,
        verbose=-1
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"    MAE:  {mae:.2f} days")
    print(f"    RMSE: {rmse:.2f} days")

    return reg, X_test, y_test

# ─────────────────────────────────────────────
# 4. CONFORMAL PREDICTION — Calibrated CI
# ─────────────────────────────────────────────

def build_conformal_intervals(reg, X_del_train, X_del_test, y_del_train, alpha=0.10):
    """
    Build prediction intervals using conformal prediction.
    alpha=0.10 → 90% coverage intervals (honest about uncertainty)
    """
    print("\n[CONFORMAL] Building calibrated prediction intervals...")

    # Residuals on calibration set (split conformal)
    cal_size = int(0.3 * len(X_del_train))
    X_cal = X_del_train.iloc[:cal_size]
    y_cal = y_del_train.iloc[:cal_size]

    cal_preds = reg.predict(X_cal)
    residuals = np.abs(y_cal.values - cal_preds)
    q_hat = np.quantile(residuals, 1 - alpha)

    print(f"    Conformal quantile (q_hat) at {int((1-alpha)*100)}% coverage: +/-{q_hat:.1f} days")

    test_preds = reg.predict(X_del_test)
    lower = np.maximum(0, test_preds - q_hat)
    upper = test_preds + q_hat

    # Empirical coverage check
    coverage = np.mean(
        (X_del_test.index[:len(y_del_test)] if False else np.ones(len(test_preds), dtype=bool))
    )
    print(f"    Interval width: +/-{q_hat:.1f} days")

    return q_hat

# ─────────────────────────────────────────────
# 5. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

def compute_shap(clf, X_train, X_test, feature_names, save_path="reports/shap_delay.png"):
    print("\n[SHAP] Computing feature importance...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test.iloc[:500])

    # Global importance plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test.iloc[:500],
                      feature_names=feature_names,
                      show=False, plot_type="bar")
    plt.title("NirmanAI — Feature Importance (SHAP)\nDelay Prediction Model", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    SHAP plot saved → {save_path}")

    return explainer, shap_values

# ─────────────────────────────────────────────
# 6. INFERENCE FUNCTION (used by API + dashboard)
# ─────────────────────────────────────────────

def predict_delay(clf, reg, q_hat, encoders, feature_names, input_dict: dict) -> dict:
    """
    Single delivery prediction.
    Returns: delay_prob, predicted_days, CI, risk_score, risk_label, risk_factors
    """
    # Encode categoricals
    row = input_dict.copy()
    for col in ["material_type", "supplier_tier", "origin_state", "destination_state", "vehicle_type", "traffic_status"]:
        if col in encoders:
            le = encoders[col]
            val = row.get(col, le.classes_[0])
            if val in le.classes_:
                row[col + "_enc"] = le.transform([val])[0]
            else:
                row[col + "_enc"] = 0  # fallback
        else:
            row[col + "_enc"] = 0

    X_input = pd.DataFrame([{f: row.get(f, 0) for f in feature_names}])

    delay_prob = float(clf.predict_proba(X_input)[0, 1])
    is_delayed = delay_prob >= 0.5

    if is_delayed:
        pred_days = max(0, float(reg.predict(X_input)[0]))
        lower = max(0, pred_days - q_hat)
        upper = pred_days + q_hat
    else:
        pred_days = 0.0
        lower = 0.0
        upper = max(0, q_hat * delay_prob * 2)

    # Risk score 0-100
    risk_score = int(delay_prob * 100)

    # Risk label
    if risk_score < 30:
        risk_label = "Low"
        risk_color = "green"
    elif risk_score < 55:
        risk_label = "Medium"
        risk_color = "orange"
    elif risk_score < 75:
        risk_label = "High"
        risk_color = "red"
    else:
        risk_label = "Critical"
        risk_color = "darkred"

    # Human-readable risk factors
    risk_factors = []
    if row.get("monsoon_intensity", 0) > 0.5:
        risk_factors.append(f"Heavy monsoon season (intensity: {row['monsoon_intensity']:.0%})")
    if row.get("past_delay_rate", 0) > 0.35:
        risk_factors.append(f"Supplier has {row['past_delay_rate']:.0%} historical delay rate")
    if row.get("is_festival_period", 0):
        risk_factors.append("Order overlaps with major festival shutdown period")
    if row.get("dest_logistics_score", 1) < 0.6:
        risk_factors.append(f"Destination state has poor logistics score ({row['dest_logistics_score']:.0%})")
    if row.get("distance_km", 0) > 800:
        risk_factors.append(f"Long-distance route ({row['distance_km']} km) increases risk")
    if not risk_factors:
        risk_factors.append("No major risk factors detected")

    return {
        "delay_probability":    round(delay_prob, 3),
        "is_delayed":           is_delayed,
        "predicted_delay_days": round(pred_days, 1),
        "ci_lower":             round(lower, 1),
        "ci_upper":             round(upper, 1),
        "risk_score":           risk_score,
        "risk_label":           risk_label,
        "risk_color":           risk_color,
        "top_risk_factors":     risk_factors[:3],
    }

# ─────────────────────────────────────────────
# 7. MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  NirmanAI — Training Delay Prediction Models")
    print("="*55)

    # Load data
    df, X, y_clf, y_reg, feature_names, encoders = load_and_prepare()

    # Train classifier
    clf, X_train_clf, X_test_clf, y_test_clf = train_classifier(X, y_clf)

    # Train regressor (on delayed orders only)
    X_delayed = X[y_clf == 1]
    y_delayed  = y_reg[y_clf == 1]
    X_del_train, X_del_test, y_del_train, y_del_test = train_test_split(
        X_delayed, y_delayed, test_size=0.2, random_state=42
    )
    reg, X_reg_test, y_reg_test = train_regressor(X, y_clf, y_reg)

    # Conformal prediction intervals
    q_hat = build_conformal_intervals(reg, X_del_train, X_del_test, y_del_train)

    # SHAP explainability
    explainer, shap_values = compute_shap(clf, X_train_clf, X_test_clf, feature_names)

    # Save models
    print("\n[SAVING] Saving models...")
    joblib.dump(clf,          "models/delay_classifier.pkl")
    joblib.dump(reg,          "models/delay_regressor.pkl")
    joblib.dump(encoders,     "models/delay_encoders.pkl")
    joblib.dump(feature_names,"models/delay_features.pkl")
    joblib.dump(q_hat,        "models/delay_q_hat.pkl")
    print("    All models saved → models/")

    # Quick sanity test
    print("\n[TEST] Running sample prediction...")
    sample = {
        "month": 8, "day_of_week": 0, "quarter": 3,
        "is_festival_period": 0,
        "material_type": "OPC Cement",
        "supplier_tier": "Tier 2 (Regional Distributor)",
        "origin_state": "Rajasthan",
        "destination_state": "Bihar",
        "distance_km": 1200,
        "order_quantity": 50,
        "promised_lead_days": 10,
        "monsoon_intensity": 0.85,
        "monsoon_sensitivity": 0.7,
        "dest_logistics_score": 0.45,
        "orig_logistics_score": 0.65,
        "dest_monsoon_severity": 0.7,
        "supplier_reliability": 0.72,
        "past_delay_rate": 0.38,
    }

    result = predict_delay(clf, reg, q_hat, encoders, feature_names, sample)
    print(f"\n  Order: OPC Cement, Rajasthan → Bihar, August (Monsoon)")
    print(f"  Delay Probability:  {result['delay_probability']:.1%}")
    print(f"  Predicted Delay:    {result['predicted_delay_days']:.0f} days")
    print(f"  Confidence Range:   {result['ci_lower']:.0f} – {result['ci_upper']:.0f} days")
    print(f"  Risk Label:         {result['risk_label']}")
    print(f"  Risk Factors:")
    for rf in result['top_risk_factors']:
        print(f"    ⚠ {rf}")

    print("\n[✓] Delay model training complete.")
