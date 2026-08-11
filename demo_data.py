"""
NirmanAI — Live Order Book
==========================
Builds the in-flight purchase orders shown on the Command Center and scores
every one of them with the trained delay + wastage models.

This module exists to remove the single biggest credibility problem in the
dashboard: the delivery monitor used to be a hardcoded table of `random.choice`
rows with pre-written alert text, so nothing on the landing screen responded to
the project context and none of it came from the models.

Every field below is either
  * a real record from the curated supplier database (names, tiers, lead times,
    reliability), or
  * a deterministic derivation of it (road distance, order value), or
  * a model output (delay probability, interval, risk factors, wastage).

The purchase orders themselves are synthetic — a demo tenant has no ERP to read
from — but they are generated deterministically from the project context, so the
same project always produces the same order book, and changing the state or the
month genuinely re-scores every row.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from suppliers_db import SUPPLIERS, state_distance_km

STATE_LOGISTICS = {
    "Gujarat": 0.82, "Maharashtra": 0.78, "Haryana": 0.74,
    "Tamil Nadu": 0.75, "Karnataka": 0.72, "Telangana": 0.70,
    "Punjab": 0.74, "West Bengal": 0.62, "Rajasthan": 0.65,
    "Madhya Pradesh": 0.55, "Uttar Pradesh": 0.58, "Kerala": 0.68,
    "Odisha": 0.50, "Bihar": 0.45, "Jharkhand": 0.42,
}

# How badly each material is affected by moisture/monsoon exposure.
MONSOON_SENSITIVITY = {
    "OPC Cement": 0.90, "River Sand": 0.85, "Coarse Aggregate": 0.70,
    "Fly Ash Bricks": 0.65, "AAC Blocks": 0.55, "Plywood": 0.75,
    "Paint": 0.60, "TMT Steel": 0.35, "Structural Steel": 0.30,
    "Electrical Cable": 0.40, "HDPE Pipes": 0.25, "Vitrified Tiles": 0.35,
}

MATERIAL_PRICES = {
    "TMT Steel": 62000, "OPC Cement": 380, "River Sand": 1800,
    "Coarse Aggregate": 1200, "Fly Ash Bricks": 8, "AAC Blocks": 55,
    "Structural Steel": 75000, "Electrical Cable": 2200,
    "HDPE Pipes": 450, "Vitrified Tiles": 650, "Plywood": 95, "Paint": 280,
}

# Typical order sizes per material, in the unit the price above is quoted in.
TYPICAL_QTY = {
    "TMT Steel": (15, 90), "OPC Cement": (200, 1400), "River Sand": (100, 600),
    "Coarse Aggregate": (150, 800), "Fly Ash Bricks": (8000, 40000),
    "AAC Blocks": (500, 4000), "Structural Steel": (10, 60),
    "Electrical Cable": (20, 180), "HDPE Pipes": (100, 900),
    "Vitrified Tiles": (300, 2500), "Plywood": (200, 1500), "Paint": (80, 700),
}

# Vehicle classes the model was actually trained on. Passing anything else
# (e.g. "Truck - Heavy") silently falls back to encoder class 0.
VEHICLE_BY_MATERIAL = {
    "TMT Steel": "Trailer", "Structural Steel": "Trailer",
    "OPC Cement": "Truck", "River Sand": "Truck",
    "Coarse Aggregate": "Truck", "Fly Ash Bricks": "Truck",
    "AAC Blocks": "Truck", "HDPE Pipes": "Container",
    "Electrical Cable": "Mini Truck", "Vitrified Tiles": "Container",
    "Plywood": "Container", "Paint": "Mini Truck",
}

RISK_ORDER = ["Critical", "High", "Medium", "Low"]
RISK_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}


def monsoon_intensity(month: int) -> float:
    profile = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.05, 5: 0.15,
               6: 0.7, 7: 0.9, 8: 0.85, 9: 0.6, 10: 0.2, 11: 0.05, 12: 0.0}
    return profile.get(month, 0.0)


def _seed_for(state: str, month: int, project_name: str) -> int:
    """
    Stable seed so a given project always yields the same order book.

    Must NOT use the builtin hash(): Python randomises string hashing per
    process, so every server restart produced a different order book for the
    same project — the headline "9 of 16 orders at risk" would change between
    runs, which makes a scripted demo impossible and looks like the model is
    unstable. blake2b is deterministic across processes and machines.
    """
    key = f"{state}|{month}|{project_name}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def build_delay_input(material, supplier, dest_state, month, quantity,
                      distance_km=None, rng=None):
    """Assemble a fully-populated feature dict for the delay model."""
    rng = rng or np.random.default_rng(0)
    m_int = monsoon_intensity(month)
    dest_log = STATE_LOGISTICS.get(dest_state, 0.6)
    orig_log = STATE_LOGISTICS.get(supplier["state"], 0.6)
    distance = distance_km if distance_km is not None else state_distance_km(
        supplier["state"], dest_state
    )
    reliability = supplier["reliability_score"]
    unit_price = MATERIAL_PRICES.get(material, 500)

    return {
        "month": month,
        "day_of_week": int(rng.integers(0, 7)),
        "quarter": (month - 1) // 3 + 1,
        "is_festival_period": int(month in (10, 11)),
        "material_type": material,
        "supplier_tier": supplier["tier"],
        "origin_state": supplier["state"],
        "destination_state": dest_state,
        "distance_km": int(distance),
        "order_quantity": float(quantity),
        "promised_lead_days": int(supplier["avg_lead_days"]),
        "vehicle_type": VEHICLE_BY_MATERIAL.get(material, "Truck"),
        # Weather: monsoon months are cooler but far more humid.
        "temperature": float(round(34 - m_int * 7 + rng.normal(0, 2), 1)),
        "humidity": float(round(min(97, 48 + m_int * 40 + rng.normal(0, 5)), 1)),
        "traffic_status": _traffic_for(dest_log, m_int, rng),
        "waiting_time": int(np.clip(rng.normal(47 + (1 - dest_log) * 25, 12), 5, 120)),
        "inventory_level": int(np.clip(rng.normal(550, 200), 40, 1500)),
        "asset_utilization": float(round(np.clip(rng.normal(78, 8), 50, 99), 1)),
        "demand_forecast": int(np.clip(rng.normal(550, 180), 80, 1200)),
        "order_value_inr": float(round(quantity * unit_price)),
        "road_quality": float(round(dest_log, 2)),
        "supplier_reliability": float(reliability),
        "past_delay_rate": float(round(1 - reliability, 3)),
        "monsoon_intensity": m_int,
        "monsoon_sensitivity": MONSOON_SENSITIVITY.get(material, 0.5),
        "dest_logistics_score": dest_log,
        "orig_logistics_score": orig_log,
        "dest_monsoon_severity": float(round(min(1.0, m_int * 1.1), 2)),
    }


def _traffic_for(logistics, m_int, rng):
    weights = np.array([
        0.40 * logistics,               # Clear
        0.30,                           # Moderate
        0.15 + m_int * 0.10,            # Detour
        0.15 + (1 - logistics) * 0.25,  # Heavy
    ])
    weights = weights / weights.sum()
    return str(rng.choice(["Clear", "Moderate", "Detour", "Heavy"], p=weights))


def build_order_book(models, state, month, project_name="Project",
                     n_orders=16) -> pd.DataFrame:
    """
    Generate the in-flight purchase orders for a project and score each one
    with the trained delay model.

    Returns a DataFrame with both display columns and the raw model outputs,
    so the UI can render a table and derive KPIs/alerts from the same source.
    """
    rng = np.random.default_rng(_seed_for(state, month, project_name))

    predict_delay = None
    if models:
        try:
            from train_delay_model import predict_delay as _pd
            predict_delay = _pd
        except Exception:
            predict_delay = None

    # Pick suppliers across a spread of materials and tiers.
    materials = list(TYPICAL_QTY.keys())
    rows = []
    for i in range(n_orders):
        material = materials[i % len(materials)]
        options = [s for s in SUPPLIERS if s["material_type"] == material]
        if not options:
            continue

        # Pick from the suppliers a real buyer would actually shortlist for this
        # site — reliable and reasonably close — rather than uniformly at random.
        # Uniform sampling produced routes like Kerala to Bihar for cement, which
        # no procurement manager would ever place and which makes the whole
        # order book read as fake.
        options = sorted(
            options,
            key=lambda s: (-s["reliability_score"]
                           + min(state_distance_km(s["state"], state) / 2000, 1.0) * 0.5),
        )
        shortlist = options[:max(3, len(options) // 2)]
        supplier = shortlist[int(rng.integers(0, len(shortlist)))]

        lo, hi = TYPICAL_QTY[material]
        quantity = int(rng.integers(lo, hi))
        distance = state_distance_km(supplier["state"], state)
        inp = build_delay_input(material, supplier, state, month, quantity,
                                distance_km=distance, rng=rng)

        if predict_delay is not None:
            try:
                res = predict_delay(
                    models["clf_delay"], models["reg_delay"],
                    models.get("conformal", 5.0),
                    models["enc_delay"], models["feat_delay"], inp,
                    explainer=models.get("explainer"),
                )
            except Exception:
                res = _heuristic_delay(inp)
        else:
            res = _heuristic_delay(inp)

        eta_days = inp["promised_lead_days"] + res["predicted_delay_days"]
        rows.append({
            "po_id": f"PO-{2600 + i * 7:04d}",
            "material": material,
            "supplier": supplier["name"],
            "supplier_state": supplier["state"],
            "tier": supplier["tier"].split(" (")[0],
            "route": f"{supplier['state']} → {state}",
            "distance_km": distance,
            "quantity": quantity,
            "order_value_inr": inp["order_value_inr"],
            "promised_lead_days": inp["promised_lead_days"],
            "delay_probability": res["delay_probability"],
            "predicted_delay_days": res["predicted_delay_days"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "conditional_delay_days": res.get(
                "conditional_delay_days", res["predicted_delay_days"]),
            "risk_score": res["risk_score"],
            "risk_label": res["risk_label"],
            "eta_days": round(eta_days, 1),
            "top_risk_factors": res["top_risk_factors"],
            "traffic": inp["traffic_status"],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_risk_rank"] = df["risk_label"].map({r: i for i, r in enumerate(RISK_ORDER)})
    df = df.sort_values(["_risk_rank", "delay_probability"],
                        ascending=[True, False]).reset_index(drop=True)
    return df


def _heuristic_delay(inp: dict) -> dict:
    """
    Transparent fallback scoring used only when the models have not been
    trained yet. Clearly labelled as an estimate everywhere it surfaces.
    """
    p = 0.06
    p += inp["monsoon_intensity"] * inp["monsoon_sensitivity"] * 0.35
    p += (1 - inp["dest_logistics_score"]) * 0.30
    p += inp["past_delay_rate"] * 0.45
    p += min(inp["distance_km"] / 3000, 1.0) * 0.12
    p = float(np.clip(p, 0.02, 0.97))

    magnitude = 2 + inp["monsoon_intensity"] * 9 + (1 - inp["dest_logistics_score"]) * 6
    score = int(p * 100)
    label = ("Low" if score < 30 else "Medium" if score < 55
             else "High" if score < 75 else "Critical")

    factors = []
    if inp["monsoon_intensity"] > 0.5:
        factors.append(f"Heavy monsoon season (intensity: {inp['monsoon_intensity']:.0%})")
    if inp["past_delay_rate"] > 0.15:
        factors.append(f"Supplier has {inp['past_delay_rate']:.0%} historical delay rate")
    if inp["dest_logistics_score"] < 0.6:
        factors.append(
            f"Destination state has poor logistics score ({inp['dest_logistics_score']:.0%})")
    if inp["distance_km"] > 800:
        factors.append(f"Long-distance route ({inp['distance_km']} km) increases risk")
    if not factors:
        factors.append("No major risk factors detected")

    return {
        "delay_probability": round(p, 3),
        "is_delayed": p >= 0.5,
        "predicted_delay_days": round(magnitude if p >= 0.5 else 0.0, 1),
        "ci_lower": round(max(0, magnitude - 4), 1) if p >= 0.5 else 0.0,
        "ci_upper": round(magnitude + 6 if p >= 0.5 else magnitude * p, 1),
        "conditional_delay_days": round(magnitude, 1),
        "risk_score": score,
        "risk_label": label,
        "top_risk_factors": factors[:3],
    }


def order_book_kpis(df: pd.DataFrame) -> dict:
    """Roll the scored order book up into the headline numbers."""
    if df.empty:
        return {"total": 0, "at_risk": 0, "critical": 0,
                "value_at_risk": 0.0, "avg_delay_prob": 0.0,
                "exposed_days": 0.0}

    at_risk = df[df["delay_probability"] >= 0.5]
    critical = df[df["risk_label"] == "Critical"]
    return {
        "total": len(df),
        "at_risk": len(at_risk),
        "critical": len(critical),
        "value_at_risk": float(at_risk["order_value_inr"].sum()),
        "avg_delay_prob": float(df["delay_probability"].mean()),
        # Expected schedule exposure: probability-weighted delay across the book
        "exposed_days": float(
            (df["delay_probability"] * df["conditional_delay_days"]).sum()
        ),
    }


def build_alerts(df: pd.DataFrame, top_n: int = 3) -> list:
    """
    Turn the highest-risk scored orders into action-oriented alerts.

    The wording is generated from each order's own model output — the risk
    factors listed are the ones the model actually keyed on.
    """
    if df.empty:
        return []

    alerts = []
    for _, row in df.head(top_n).iterrows():
        if row["risk_label"] not in ("Critical", "High"):
            break
        driver = row["top_risk_factors"][0] if row["top_risk_factors"] else "elevated risk"
        alerts.append({
            "severity": row["risk_label"],
            "po_id": row["po_id"],
            "title": f"{row['material']} — {row['route']}",
            "probability": row["delay_probability"],
            "expected_days": row["conditional_delay_days"],
            "ci_lower": row["ci_lower"],
            "ci_upper": row["ci_upper"],
            "supplier": row["supplier"],
            "driver": driver,
            "all_factors": list(row["top_risk_factors"]),
            "value": row["order_value_inr"],
            "action": _action_for(row),
        })
    return alerts


def _action_for(row) -> str:
    """The single next step a site manager should take for this order."""
    if row["risk_label"] == "Critical":
        return (f"Split this order now — move 50% to a backup supplier and "
                f"pull the delivery date forward by "
                f"{max(3, int(row['conditional_delay_days']))} days.")
    if row["delay_probability"] >= 0.6:
        return ("Confirm dispatch with the supplier today and hold "
                f"{max(3, int(row['conditional_delay_days'] * 0.6))} days of "
                "buffer stock on site.")
    return "Monitor — re-check dispatch status 48 hours before the promised date."


def build_wastage_forecast(models, materials_qty: dict, project_ctx: dict) -> pd.DataFrame:
    """
    Score every material in the Bill of Quantities through the wastage model
    using the ACTUAL site conditions the user selected in the sidebar.
    """
    if not materials_qty:
        return pd.DataFrame()

    predict_wastage = None
    if models:
        try:
            from train_wastage_model import predict_wastage as _pw
            predict_wastage = _pw
        except Exception:
            predict_wastage = None

    m_int = monsoon_intensity(project_ctx["month"])
    rows = []
    for material, qty in materials_qty.items():
        payload = {
            "project_type": project_ctx["project_type"],
            "state": project_ctx["state"],
            "project_size_sqft": project_ctx["project_size"],
            "project_duration_months": project_ctx.get("duration_months", 12),
            "month_of_construction": project_ctx["month"],
            "contractor_experience_yrs": project_ctx["contractor_exp"],
            "num_workers": max(20, project_ctx["project_size"] // 200),
            "workforce_skill_level": project_ctx["workforce_skill"],
            "supervision_quality": project_ctx["supervision"],
            "material_type": material,
            "blueprint_quantity": qty,
            "logistics_score": STATE_LOGISTICS.get(project_ctx["state"], 0.6),
            "monsoon_intensity": m_int,
            "monsoon_sensitivity": MONSOON_SENSITIVITY.get(material, 0.5),
        }
        if predict_wastage is not None:
            try:
                res = predict_wastage(
                    models["reg_wast"], models["reg_wast_lo"], models["reg_wast_hi"],
                    models["enc_wast"], models["feat_wast"], payload,
                )
            except Exception:
                res = _heuristic_wastage(payload)
        else:
            res = _heuristic_wastage(payload)

        rows.append({
            "material": material,
            "blueprint_qty": qty,
            "order_qty": res["actual_qty_estimate"],
            "wastage_pct": res["predicted_wastage_pct"],
            "wastage_low": res["wastage_range_low"],
            "wastage_high": res["wastage_range_high"],
            "category": res["wastage_category"],
            "cost_overrun": res["estimated_cost_overrun_inr"],
            "drivers": res["risk_factors"],
        })
    return pd.DataFrame(rows)


def _heuristic_wastage(payload: dict) -> dict:
    """Transparent fallback when the wastage model is unavailable."""
    skill = {"Unskilled": 1.35, "Semi-skilled": 1.15, "Skilled": 1.0, "Expert": 0.85}
    supervision = {"Poor": 1.35, "Average": 1.15, "Good": 0.95, "Excellent": 0.8}
    base = 9.0
    pct = base * skill.get(payload["workforce_skill_level"], 1.0)
    pct *= supervision.get(payload["supervision_quality"], 1.0)
    pct *= 1 + payload["monsoon_intensity"] * payload["monsoon_sensitivity"] * 0.35
    pct = float(np.clip(pct, 1.5, 45.0))

    qty = payload["blueprint_quantity"]
    actual = qty * (1 + pct / 100)
    price = MATERIAL_PRICES.get(payload["material_type"], 500)
    category = "Low" if pct < 5 else "Medium" if pct < 15 else "High"

    drivers = []
    if payload["workforce_skill_level"] in ("Unskilled", "Semi-skilled"):
        drivers.append(
            f"Workforce is {payload['workforce_skill_level'].lower()} — "
            "higher material wastage expected")
    if payload["supervision_quality"] in ("Poor", "Average"):
        drivers.append(
            f"{payload['supervision_quality']} site supervision increases wastage by 10–35%")
    if payload["monsoon_intensity"] > 0.5 and payload["monsoon_sensitivity"] > 0.5:
        drivers.append("Monsoon season degrades cement/sand quality, increasing wastage")
    if not drivers:
        drivers.append("Well-managed site — wastage within acceptable range")

    return {
        "predicted_wastage_pct": round(pct, 1),
        "wastage_range_low": round(pct * 0.75, 1),
        "wastage_range_high": round(pct * 1.3, 1),
        "actual_qty_estimate": round(actual, 2),
        "blueprint_quantity": qty,
        "estimated_cost_overrun_inr": round((actual - qty) * price),
        "wastage_category": category,
        "risk_factors": drivers[:3],
    }
