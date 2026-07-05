"""
SupplyMind — Synthetic Dataset Generator
==========================================
Generates realistic Indian construction supply chain data:
- Delivery records with delay labels
- Project material consumption records with wastage labels

All patterns are calibrated against published industry benchmarks:
- CIDC (Construction Industry Development Council) reports
- McKinsey India Infrastructure Report 2024
- IMD monsoon/weather patterns
- NHAI logistics data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────
# CONSTANTS — India-specific domain knowledge
# ─────────────────────────────────────────────

MATERIALS = {
    "TMT Steel":        {"base_lead_days": 14, "weight_tons": True,  "monsoon_sensitive": 0.3, "price_per_unit": 62000},
    "OPC Cement":       {"base_lead_days": 5,  "weight_tons": True,  "monsoon_sensitive": 0.7, "price_per_unit": 380},
    "River Sand":       {"base_lead_days": 7,  "weight_tons": True,  "monsoon_sensitive": 0.9, "price_per_unit": 1800},
    "Coarse Aggregate": {"base_lead_days": 6,  "weight_tons": True,  "monsoon_sensitive": 0.5, "price_per_unit": 1200},
    "Fly Ash Bricks":   {"base_lead_days": 10, "weight_tons": False, "monsoon_sensitive": 0.6, "price_per_unit": 8},
    "AAC Blocks":       {"base_lead_days": 12, "weight_tons": False, "monsoon_sensitive": 0.2, "price_per_unit": 55},
    "Structural Steel": {"base_lead_days": 21, "weight_tons": True,  "monsoon_sensitive": 0.2, "price_per_unit": 75000},
    "Electrical Cable": {"base_lead_days": 18, "weight_tons": False, "monsoon_sensitive": 0.1, "price_per_unit": 2200},
    "HDPE Pipes":       {"base_lead_days": 16, "weight_tons": False, "monsoon_sensitive": 0.1, "price_per_unit": 450},
    "Vitrified Tiles":  {"base_lead_days": 20, "weight_tons": False, "monsoon_sensitive": 0.2, "price_per_unit": 650},
    "Plywood":          {"base_lead_days": 8,  "weight_tons": False, "monsoon_sensitive": 0.8, "price_per_unit": 95},
    "Paint":            {"base_lead_days": 5,  "weight_tons": False, "monsoon_sensitive": 0.1, "price_per_unit": 280},
}

# Major Indian states with logistics quality scores (0-1, 1=best)
STATES = {
    "Maharashtra":    {"logistics_score": 0.78, "monsoon_severity": 0.8},
    "Tamil Nadu":     {"logistics_score": 0.75, "monsoon_severity": 0.7},
    "Karnataka":      {"logistics_score": 0.72, "monsoon_severity": 0.7},
    "Gujarat":        {"logistics_score": 0.82, "monsoon_severity": 0.4},
    "Rajasthan":      {"logistics_score": 0.65, "monsoon_severity": 0.3},
    "Uttar Pradesh":  {"logistics_score": 0.58, "monsoon_severity": 0.6},
    "Bihar":          {"logistics_score": 0.45, "monsoon_severity": 0.7},
    "West Bengal":    {"logistics_score": 0.62, "monsoon_severity": 0.8},
    "Madhya Pradesh": {"logistics_score": 0.55, "monsoon_severity": 0.5},
    "Telangana":      {"logistics_score": 0.70, "monsoon_severity": 0.6},
    "Kerala":         {"logistics_score": 0.68, "monsoon_severity": 0.95},
    "Punjab":         {"logistics_score": 0.74, "monsoon_severity": 0.5},
    "Odisha":         {"logistics_score": 0.50, "monsoon_severity": 0.8},
    "Jharkhand":      {"logistics_score": 0.42, "monsoon_severity": 0.7},
    "Haryana":        {"logistics_score": 0.71, "monsoon_severity": 0.5},
}

PROJECT_TYPES = [
    "Residential Apartment",
    "Commercial Complex",
    "Industrial Warehouse",
    "Road & Highway",
    "Bridge Construction",
    "Metro Rail",
    "Data Center",
    "Hospital",
    "School/College",
]

SUPPLIER_TIERS = {
    "Tier 1 (Large Manufacturer)": {"reliability": 0.88, "delay_std": 2.0},
    "Tier 2 (Regional Distributor)": {"reliability": 0.72, "delay_std": 3.5},
    "Tier 3 (Local Supplier)":     {"reliability": 0.55, "delay_std": 6.0},
}

# Indian festivals that disrupt supply chain (month, day_start, duration_days)
FESTIVAL_DISRUPTIONS = [
    (1, 14, 3),   # Pongal / Makar Sankranti
    (3, 25, 5),   # Holi
    (4, 14, 2),   # Tamil New Year / Baisakhi
    (8, 15, 2),   # Independence Day
    (9, 15, 5),   # Ganesh Chaturthi (variable — approx)
    (10, 2, 3),   # Dussehra region
    (10, 20, 7),  # Diwali + Chhath
    (12, 25, 4),  # Christmas / Year-end
]

# Monsoon months by intensity
def monsoon_intensity(month: int) -> float:
    """Returns monsoon disruption factor 0-1 for a given month."""
    profile = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.05, 5: 0.15,
               6: 0.7,  7: 0.9, 8: 0.85, 9: 0.6,  10: 0.2,
               11: 0.05, 12: 0.0}
    return profile.get(month, 0.0)

def is_festival_period(date: datetime) -> bool:
    """Check if date falls within a major Indian festival disruption window."""
    for month, day, duration in FESTIVAL_DISRUPTIONS:
        start = datetime(date.year, month, day)
        end = start + timedelta(days=duration)
        if start <= date <= end:
            return True
    return False

# ─────────────────────────────────────────────
# DATASET 1: DELIVERY DELAY RECORDS
# ─────────────────────────────────────────────

def generate_delay_dataset(n_records: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic delivery records for Indian construction projects.
    Each row = one material delivery order.
    
    Features:
    - material_type, supplier_tier, order_quantity
    - origin_state, destination_state, distance_km
    - order_date (month, day_of_week, is_festival)
    - promised_lead_days, monsoon_intensity, logistics_score
    - supplier_reliability, past_delay_rate (supplier history)
    
    Target:
    - actual_delay_days (0 = on time, >0 = late)
    - is_delayed (binary)
    """

    records = []
    start_date = datetime(2023, 1, 1)
    end_date   = datetime(2025, 12, 31)
    date_range = (end_date - start_date).days

    state_list = list(STATES.keys())
    material_list = list(MATERIALS.keys())
    supplier_list = list(SUPPLIER_TIERS.keys())

    for _ in range(n_records):
        # Random order date
        order_date = start_date + timedelta(days=random.randint(0, date_range))
        month = order_date.month
        day_of_week = order_date.weekday()  # 0=Monday
        is_festival = is_festival_period(order_date)

        # Material
        material = random.choice(material_list)
        mat_props = MATERIALS[material]

        # Supplier
        supplier_tier = random.choice(supplier_list)
        sup_props = SUPPLIER_TIERS[supplier_tier]

        # States
        origin_state = random.choice(state_list)
        dest_state = random.choice(state_list)
        # Simulated distance
        if origin_state == dest_state:
            distance_km = np.random.randint(50, 300)
        else:
            distance_km = np.random.randint(150, 1800)

        dest_props = STATES[dest_state]
        orig_props = STATES[origin_state]

        # Order quantity
        if mat_props["weight_tons"]:
            quantity = round(np.random.lognormal(mean=3.5, sigma=0.8), 1)  # tonnes
        else:
            quantity = round(np.random.lognormal(mean=6.0, sigma=1.0))     # units

        promised_lead_days = mat_props["base_lead_days"] + max(0, int(distance_km / 200) - 1)

        # Supplier's historical delay rate (0-0.7)
        base_delay_rate = 1 - sup_props["reliability"]
        past_delay_rate = round(np.clip(
            base_delay_rate + np.random.normal(0, 0.08), 0.02, 0.70
        ), 3)

        # ── DELAY CALCULATION (causal model) ──
        m_intensity = monsoon_intensity(month)
        monsoon_factor = m_intensity * mat_props["monsoon_sensitive"] * dest_props["monsoon_severity"]
        logistics_factor = 1 - ((orig_props["logistics_score"] + dest_props["logistics_score"]) / 2)
        festival_factor = 0.4 if is_festival else 0.0
        distance_factor = min(distance_km / 2000, 0.5)
        reliability_factor = past_delay_rate * 1.5

        # Aggregate delay probability
        delay_prob = np.clip(
            0.10
            + monsoon_factor * 0.40
            + logistics_factor * 0.25
            + festival_factor * 0.25
            + distance_factor * 0.15
            + reliability_factor * 0.30
            + (0.05 if day_of_week >= 5 else 0)  # weekend orders
            + np.random.normal(0, 0.05),
            0.02, 0.95
        )

        is_delayed = np.random.random() < delay_prob

        if is_delayed:
            mean_delay = (
                sup_props["delay_std"]
                + monsoon_factor * 10
                + distance_factor * 5
                + festival_factor * 4
                + past_delay_rate * 8
            )
            actual_delay_days = max(1, int(np.random.exponential(scale=mean_delay)))
            actual_delay_days = min(actual_delay_days, 45)  # cap at 45 days
        else:
            actual_delay_days = 0

        records.append({
            "order_id":             f"ORD-{len(records)+1:05d}",
            "order_date":           order_date.strftime("%Y-%m-%d"),
            "month":                month,
            "day_of_week":          day_of_week,
            "quarter":              (month - 1) // 3 + 1,
            "is_festival_period":   int(is_festival),
            "material_type":        material,
            "supplier_tier":        supplier_tier,
            "origin_state":         origin_state,
            "destination_state":    dest_state,
            "distance_km":          distance_km,
            "order_quantity":       quantity,
            "promised_lead_days":   promised_lead_days,
            "monsoon_intensity":    round(m_intensity, 3),
            "monsoon_sensitivity":  mat_props["monsoon_sensitive"],
            "dest_logistics_score": dest_props["logistics_score"],
            "orig_logistics_score": orig_props["logistics_score"],
            "dest_monsoon_severity":dest_props["monsoon_severity"],
            "supplier_reliability": sup_props["reliability"],
            "past_delay_rate":      past_delay_rate,
            "delay_probability":    round(delay_prob, 3),
            "is_delayed":           int(is_delayed),
            "actual_delay_days":    actual_delay_days,
        })

    df = pd.DataFrame(records)
    print(f"[✓] Delay dataset: {len(df)} records")
    print(f"    Delayed orders: {df['is_delayed'].sum()} ({df['is_delayed'].mean()*100:.1f}%)")
    print(f"    Avg delay (when late): {df[df['is_delayed']==1]['actual_delay_days'].mean():.1f} days")
    return df


# ─────────────────────────────────────────────
# DATASET 2: MATERIAL WASTAGE RECORDS
# ─────────────────────────────────────────────

def generate_wastage_dataset(n_records: int = 3000) -> pd.DataFrame:
    """
    Generate synthetic project-level material consumption records.
    Each row = one material on one project.
    
    Features:
    - project_type, project_size_sqft, project_duration_months
    - material_type, blueprint_quantity
    - workforce_skill_level, num_workers
    - weather_conditions, site_supervision_quality
    - contractor_experience_years
    
    Target:
    - actual_quantity_used
    - wastage_pct  (= (actual - blueprint) / blueprint * 100)
    - wastage_category (Low <5%, Medium 5-15%, High >15%)
    """

    records = []
    project_counter = 1

    # Wastage benchmarks by material (from CIDC / industry reports)
    wastage_benchmarks = {
        "TMT Steel":        {"mean": 5.0,  "std": 2.5},
        "OPC Cement":       {"mean": 10.0, "std": 4.0},
        "River Sand":       {"mean": 15.0, "std": 5.0},
        "Coarse Aggregate": {"mean": 8.0,  "std": 3.5},
        "Fly Ash Bricks":   {"mean": 12.0, "std": 5.0},
        "AAC Blocks":       {"mean": 7.0,  "std": 3.0},
        "Structural Steel": {"mean": 4.0,  "std": 2.0},
        "Electrical Cable": {"mean": 6.0,  "std": 2.5},
        "HDPE Pipes":       {"mean": 5.0,  "std": 2.0},
        "Vitrified Tiles":  {"mean": 8.0,  "std": 4.0},
        "Plywood":          {"mean": 20.0, "std": 7.0},
        "Paint":            {"mean": 6.0,  "std": 2.5},
    }

    skill_levels = {"Unskilled": 1.4, "Semi-skilled": 1.15, "Skilled": 1.0, "Expert": 0.9}
    supervision_levels = {"Poor": 1.35, "Average": 1.10, "Good": 1.0, "Excellent": 0.85}

    for _ in range(n_records):
        project_type = random.choice(PROJECT_TYPES)
        state = random.choice(list(STATES.keys()))
        state_props = STATES[state]
        month = random.randint(1, 12)

        # Project characteristics
        if project_type in ["Metro Rail", "Bridge Construction", "Road & Highway"]:
            project_size = random.randint(5000, 50000)
            duration = random.randint(18, 60)
        elif project_type in ["Residential Apartment", "Commercial Complex"]:
            project_size = random.randint(2000, 25000)
            duration = random.randint(12, 36)
        else:
            project_size = random.randint(1000, 15000)
            duration = random.randint(8, 24)

        contractor_exp = random.randint(1, 30)
        num_workers = random.randint(20, 500)
        workforce_skill = random.choices(
            list(skill_levels.keys()),
            weights=[0.45, 0.30, 0.18, 0.07]  # India: mostly unskilled
        )[0]
        supervision = random.choices(
            list(supervision_levels.keys()),
            weights=[0.25, 0.40, 0.25, 0.10]
        )[0]

        material = random.choice(list(MATERIALS.keys()))
        mat_props = MATERIALS[material]

        if mat_props["weight_tons"]:
            blueprint_qty = round(np.random.lognormal(mean=4.0, sigma=0.9), 1)
        else:
            blueprint_qty = round(np.random.lognormal(mean=6.5, sigma=1.0))

        # ── WASTAGE CALCULATION ──
        base_mean = wastage_benchmarks[material]["mean"]
        base_std  = wastage_benchmarks[material]["std"]

        skill_factor = skill_levels[workforce_skill]
        supervision_factor = supervision_levels[supervision]
        monsoon_factor = 1 + monsoon_intensity(month) * mat_props["monsoon_sensitive"] * 0.3
        experience_factor = max(0.7, 1.0 - (contractor_exp - 5) * 0.01)
        size_factor = 1 + max(0, (project_size - 10000) / 100000) * 0.1

        adjusted_mean = (
            base_mean
            * skill_factor
            * supervision_factor
            * monsoon_factor
            * experience_factor
            * size_factor
            + np.random.normal(0, base_std * 0.3)
        )
        adjusted_mean = max(1.0, adjusted_mean)

        wastage_pct = max(0.5, np.random.normal(adjusted_mean, base_std * 0.5))
        actual_qty = round(blueprint_qty * (1 + wastage_pct / 100), 2)
        cost_overrun = round((actual_qty - blueprint_qty) * mat_props["price_per_unit"], 0)

        if wastage_pct < 5:
            wastage_category = "Low"
        elif wastage_pct < 15:
            wastage_category = "Medium"
        else:
            wastage_category = "High"

        records.append({
            "project_id":               f"PRJ-{project_counter:04d}",
            "project_type":             project_type,
            "state":                    state,
            "project_size_sqft":        project_size,
            "project_duration_months":  duration,
            "month_of_construction":    month,
            "contractor_experience_yrs":contractor_exp,
            "num_workers":              num_workers,
            "workforce_skill_level":    workforce_skill,
            "supervision_quality":      supervision,
            "material_type":            material,
            "blueprint_quantity":       blueprint_qty,
            "actual_quantity_used":     actual_qty,
            "wastage_pct":              round(wastage_pct, 2),
            "wastage_category":         wastage_category,
            "cost_overrun_inr":         cost_overrun,
            "logistics_score":          state_props["logistics_score"],
            "monsoon_intensity":        round(monsoon_intensity(month), 3),
            "monsoon_sensitivity":      mat_props["monsoon_sensitive"],
        })
        project_counter += 1

    df = pd.DataFrame(records)
    print(f"[✓] Wastage dataset: {len(df)} records")
    print(f"    Mean wastage: {df['wastage_pct'].mean():.1f}%")
    print(f"    High wastage projects: {(df['wastage_category']=='High').sum()} ({(df['wastage_category']=='High').mean()*100:.1f}%)")
    print(f"    Avg cost overrun: ₹{df['cost_overrun_inr'].mean():,.0f}")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("\n" + "="*50)
    print("  SupplyMind — Generating Synthetic Datasets")
    print("="*50 + "\n")

    print("Generating delivery delay dataset...")
    delay_df = generate_delay_dataset(n_records=5000)
    delay_df.to_csv("data/delivery_delays.csv", index=False)
    print(f"    Saved → data/delivery_delays.csv\n")

    print("Generating material wastage dataset...")
    wastage_df = generate_wastage_dataset(n_records=3000)
    wastage_df.to_csv("data/material_wastage.csv", index=False)
    print(f"    Saved → data/material_wastage.csv\n")

    print("Sample — Delay Dataset:")
    print(delay_df[["material_type", "supplier_tier", "origin_state", "destination_state",
                     "monsoon_intensity", "is_delayed", "actual_delay_days"]].head(5).to_string())

    print("\nSample — Wastage Dataset:")
    print(wastage_df[["project_type", "material_type", "workforce_skill_level",
                       "supervision_quality", "wastage_pct", "wastage_category"]].head(5).to_string())

    print("\n[✓] All datasets generated successfully.")
