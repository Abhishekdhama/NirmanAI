"""
NirmanAI — Monte Carlo Supply Chain Simulation Engine
=========================================================
Simulates an entire construction project's supply chain thousands of times
to find failure modes, cascade effects, and probability distributions.

This is NirmanAI's core differentiator: not just "what's the risk?"
but "what happens to your ENTIRE PROJECT if things go wrong?"
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import time


# ── Construction Activity Dependency Graph ────────────────────
# Each activity depends on materials and predecessor activities.
# This models the REAL sequencing of an Indian construction project.

ACTIVITY_TEMPLATES = {
    "Residential Apartment": [
        {"name": "Foundation & Excavation",   "duration_days": 30, "materials": ["River Sand", "Coarse Aggregate", "OPC Cement"], "predecessors": []},
        {"name": "Structural Steel Framing",  "duration_days": 45, "materials": ["TMT Steel", "Structural Steel"],               "predecessors": ["Foundation & Excavation"]},
        {"name": "RCC Column & Beam Work",    "duration_days": 40, "materials": ["TMT Steel", "OPC Cement", "River Sand"],       "predecessors": ["Structural Steel Framing"]},
        {"name": "Brick/Block Masonry",       "duration_days": 35, "materials": ["Fly Ash Bricks", "AAC Blocks", "OPC Cement"],  "predecessors": ["RCC Column & Beam Work"]},
        {"name": "Plumbing Rough-in",         "duration_days": 20, "materials": ["HDPE Pipes"],                                  "predecessors": ["Brick/Block Masonry"]},
        {"name": "Electrical Rough-in",       "duration_days": 20, "materials": ["Electrical Cable"],                            "predecessors": ["Brick/Block Masonry"]},
        {"name": "Plastering & Finishing",     "duration_days": 25, "materials": ["OPC Cement", "River Sand"],                    "predecessors": ["Plumbing Rough-in", "Electrical Rough-in"]},
        {"name": "Flooring & Tiling",         "duration_days": 20, "materials": ["Vitrified Tiles"],                             "predecessors": ["Plastering & Finishing"]},
        {"name": "Woodwork & Joinery",        "duration_days": 15, "materials": ["Plywood"],                                     "predecessors": ["Plastering & Finishing"]},
        {"name": "Painting & Final Finish",   "duration_days": 15, "materials": ["Paint"],                                       "predecessors": ["Flooring & Tiling", "Woodwork & Joinery"]},
    ],
    "Commercial Complex": [
        {"name": "Site Preparation & Piling", "duration_days": 35, "materials": ["River Sand", "Coarse Aggregate", "OPC Cement"], "predecessors": []},
        {"name": "Structural Steel Erection", "duration_days": 50, "materials": ["Structural Steel", "TMT Steel"],                "predecessors": ["Site Preparation & Piling"]},
        {"name": "Floor Slab Casting",        "duration_days": 40, "materials": ["TMT Steel", "OPC Cement", "River Sand"],        "predecessors": ["Structural Steel Erection"]},
        {"name": "Glass & Facade Work",       "duration_days": 30, "materials": ["Structural Steel"],                             "predecessors": ["Floor Slab Casting"]},
        {"name": "MEP Systems",               "duration_days": 35, "materials": ["Electrical Cable", "HDPE Pipes"],               "predecessors": ["Floor Slab Casting"]},
        {"name": "Interior Finishing",        "duration_days": 25, "materials": ["Vitrified Tiles", "Plywood", "Paint"],          "predecessors": ["Glass & Facade Work", "MEP Systems"]},
    ],
    "Road & Highway": [
        {"name": "Land Clearing & Grading",   "duration_days": 20, "materials": ["River Sand", "Coarse Aggregate"],               "predecessors": []},
        {"name": "Sub-base & Base Course",    "duration_days": 30, "materials": ["Coarse Aggregate", "River Sand"],               "predecessors": ["Land Clearing & Grading"]},
        {"name": "Concrete Pavement Laying",  "duration_days": 45, "materials": ["OPC Cement", "TMT Steel", "River Sand"],        "predecessors": ["Sub-base & Base Course"]},
        {"name": "Drainage & Culvert Work",   "duration_days": 25, "materials": ["HDPE Pipes", "OPC Cement"],                     "predecessors": ["Sub-base & Base Course"]},
        {"name": "Road Marking & Signage",    "duration_days": 10, "materials": ["Paint"],                                        "predecessors": ["Concrete Pavement Laying", "Drainage & Culvert Work"]},
    ],
}

# Default template for project types not explicitly listed
DEFAULT_TEMPLATE = ACTIVITY_TEMPLATES["Residential Apartment"]


def get_activity_template(project_type: str) -> list:
    """Get the activity dependency graph for a project type."""
    return ACTIVITY_TEMPLATES.get(project_type, DEFAULT_TEMPLATE)


def monsoon_intensity(month: int) -> float:
    """Monsoon intensity by month (matching our ML model)."""
    profile = {1:0.0, 2:0.0, 3:0.0, 4:0.05, 5:0.15,
               6:0.7, 7:0.9, 8:0.85, 9:0.6, 10:0.2, 11:0.05, 12:0.0}
    return profile.get(month, 0.0)


# ── State Logistics Scores ───────────────────────────────────

STATE_LOGISTICS = {
    "Gujarat": 0.82, "Maharashtra": 0.78, "Haryana": 0.74,
    "Tamil Nadu": 0.75, "Karnataka": 0.72, "Telangana": 0.70,
    "Punjab": 0.74, "West Bengal": 0.62, "Rajasthan": 0.65,
    "Madhya Pradesh": 0.55, "Uttar Pradesh": 0.58, "Kerala": 0.68,
    "Odisha": 0.50, "Bihar": 0.45, "Jharkhand": 0.42,
}


# Fraction of a material delay that a site can absorb by re-sequencing crews,
# working the float in the schedule, or partial-delivery starts. Calibrated so
# that a monsoon project in a weak-logistics state lands near the ~23% on-time
# rate implied by MoSPI's infrastructure-project overrun statistics, rather
# than the 0% a naive fully-serial model produces.
SCHEDULE_ABSORPTION = 0.45

# Baseline share of supplier failure risk that is COMMON-MODE — driven by
# events that hit every supplier on the same corridor at once (monsoon, festival
# shutdowns, fuel/trucking strikes). Dual sourcing cannot diversify this away.
# Rises with monsoon intensity; see optimize_procurement().
COMMON_MODE_CORRELATION_BASE = 0.20


def build_risk_profiles(
    materials: list,
    state: str,
    month: int,
    models: dict = None,
    distance_km: int = 800,
    supplier_reliability: float = 0.7,
) -> dict:
    """
    Turn the trained delay model into a per-material risk PROFILE:

        {material: {"p_delay": float, "lo": days, "mid": days, "hi": days}}

    This is what makes an ML-driven Monte Carlo tractable. Scoring the model
    inside the simulation loop would mean ~300,000 sklearn calls for a 10,000-run
    simulation (minutes). Scoring it once per material gives ~12 calls, and the
    loop then samples from each material's calibrated delay distribution — the
    ML model still sets every probability and magnitude, it just does so once.

    Falls back to a transparent physics-based profile when models are absent.
    """
    m_int = monsoon_intensity(month)
    logistics = STATE_LOGISTICS.get(state, 0.6)
    profiles = {}

    predict_delay = None
    if models:
        try:
            from train_delay_model import predict_delay as _pd
            predict_delay = _pd
        except Exception:
            predict_delay = None

    for material in materials:
        if predict_delay is not None:
            try:
                inp = {
                    "month": month, "day_of_week": 2,
                    "quarter": (month - 1) // 3 + 1,
                    "is_festival_period": int(month in (10, 11)),
                    "material_type": material,
                    "supplier_tier": "Tier 2 (Regional Distributor)",
                    "origin_state": "Maharashtra",
                    "destination_state": state,
                    "distance_km": distance_km,
                    "order_quantity": 250,
                    "promised_lead_days": 14,
                    "monsoon_intensity": m_int,
                    "monsoon_sensitivity": 0.6,
                    "dest_logistics_score": logistics,
                    "orig_logistics_score": 0.78,
                    "dest_monsoon_severity": min(1.0, m_int * 1.1),
                    "supplier_reliability": supplier_reliability,
                    "past_delay_rate": 1 - supplier_reliability,
                    "vehicle_type": "Truck - Heavy",
                    "temperature": 30.0,
                    "humidity": 55.0 + m_int * 35.0,
                    "traffic_status": "Moderate",
                    "waiting_time": 20,
                    "inventory_level": 400,
                    "asset_utilization": 80.0,
                    "demand_forecast": 400,
                    "order_value_inr": 250 * 1500,
                    "road_quality": logistics,
                    "supplier_capacity": 80,
                    "fuel_price_index": 102.0,
                    "driver_experience": 10,
                }
                res = predict_delay(
                    models["clf_delay"], models["reg_delay"],
                    models.get("conformal", 5.0),
                    models["enc_delay"], models["feat_delay"], inp,
                )
                # Use the CONDITIONAL magnitude ("if late, how late?"), not the
                # threshold-gated one — a material with a 30% delay probability
                # still delays by its full predicted magnitude in the 30% of
                # simulations where it goes late.
                mid = max(1.0, float(res["conditional_delay_days"]))
                profiles[material] = {
                    "p_delay": float(res["delay_probability"]),
                    "lo": max(0.0, float(res["conditional_ci_lower"])),
                    "mid": mid,
                    "hi": max(mid, float(res["conditional_ci_upper"])),
                    "source": "ml",
                }
                continue
            except Exception:
                pass

        # Physics-based profile: transparent, and the only path when the models
        # have not been trained yet. Weights are calibrated so this fallback
        # lands in the same range as the trained model (~12% delay probability
        # for Gujarat in January, ~30% for Bihar in July) — an uncalibrated
        # fallback would make the simulator's headline numbers meaningless.
        p = 0.04
        p += m_int * 0.14                       # monsoon disruption
        p += (1 - logistics) * 0.16             # state logistics quality
        p += (1 - supplier_reliability) * 0.12  # supplier track record
        p += min(distance_km / 3000, 0.05)      # route length
        p = float(np.clip(p, 0.03, 0.85))
        scale = 3 + m_int * 8 + (1 - logistics) * 5
        profiles[material] = {
            "p_delay": p,
            "lo": 1.0,
            "mid": scale,
            "hi": min(45.0, scale * 2.8),
            "source": "physics",
        }

    return profiles


def sample_material_delay(profile: dict, rng: np.random.Generator) -> float:
    """Draw one delay realisation (days) from a material's risk profile."""
    if rng.random() >= profile["p_delay"]:
        return 0.0
    # Triangular over the conformal interval: respects the calibrated bounds
    # while concentrating mass on the point estimate.
    lo, mid, hi = profile["lo"], profile["mid"], profile["hi"]
    if hi <= lo:
        return max(0.0, mid)
    mid = min(max(mid, lo), hi)
    return float(rng.triangular(lo, mid, hi))


def simulate_material_delay(
    material: str,
    state: str,
    month: int,
    supplier_reliability: float = 0.7,
    distance_km: int = 800,
    models: dict = None,
) -> float:
    """
    Simulate a single material delivery delay in days (0 = on time).

    Convenience wrapper around build_risk_profiles + sample_material_delay,
    kept for callers that want a one-off draw. The simulation loop itself uses
    the cached profiles instead.
    """
    profile = build_risk_profiles(
        [material], state, month, models,
        distance_km=distance_km, supplier_reliability=supplier_reliability,
    )[material]
    return sample_material_delay(profile, np.random.default_rng())


def _max_schedule_months(activities: list) -> int:
    """Upper bound (in months) on how far the schedule can stretch."""
    total_days = sum(a["duration_days"] for a in activities)
    return int(total_days / 30) + 2


def run_simulation(
    project_type: str,
    state: str,
    start_month: int,
    n_simulations: int = 5000,
    models: dict = None,
    seed: int = 42,
) -> dict:
    """
    Run Monte Carlo simulation of the entire project supply chain.

    Simulates the project n_simulations times, modeling:
    - Material delivery delays (using ML model or physics-based)
    - Activity dependencies (critical path)
    - Cascade effects (one delay rippling through the schedule)

    Returns a rich result dict with distributions, critical paths,
    and actionable insights.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    activities = get_activity_template(project_type)

    # ── Score the ML model ONCE per (material, calendar month) ────
    # Activities run at different points in the project, so a material's risk
    # depends on when its activity actually starts. We pre-score every month the
    # schedule can touch; ~12 model calls replaces ~300,000 in-loop calls.
    all_materials = sorted({m for a in activities for m in a["materials"]})
    months_needed = sorted({
        ((start_month - 1 + offset) % 12) + 1
        for offset in range(0, _max_schedule_months(activities) + 1)
    })
    profile_cache = {
        month: build_risk_profiles(all_materials, state, month, models)
        for month in months_needed
    }
    profile_source = next(
        (p["source"] for m in profile_cache.values() for p in m.values()),
        "physics",
    )

    # Storage for simulation results
    project_durations = np.zeros(n_simulations)
    activity_delays = {a["name"]: np.zeros(n_simulations) for a in activities}
    # Per-DELIVERY counters. Counting "was this material late anywhere in the
    # project" saturates at 100% for any material used by several activities,
    # which tells a site manager nothing. "62% of your cement deliveries land
    # late" is both non-saturating and directly actionable.
    material_late_deliveries = {m: 0 for m in all_materials}
    material_total_deliveries = {m: 0 for m in all_materials}
    material_sim_hits = {m: 0 for m in all_materials}   # simulations, not occurrences
    critical_path_counts = {}
    cascade_events = []

    for sim in range(n_simulations):
        # Track completion time for each activity
        completion_times = {}
        materials_late_this_sim = set()

        for act in activities:
            name = act["name"]
            base_duration = act["duration_days"]

            # Calculate the earliest start (max completion of all predecessors)
            if act["predecessors"]:
                earliest_start = max(
                    completion_times[pred] for pred in act["predecessors"]
                )
            else:
                earliest_start = 0

            # Simulate material delays for this activity
            max_material_delay = 0.0
            worst_material = None
            activity_month = ((start_month - 1 + int(earliest_start / 30)) % 12) + 1
            month_profiles = profile_cache.get(
                activity_month, profile_cache[months_needed[0]]
            )

            for mat in act["materials"]:
                delay = sample_material_delay(month_profiles[mat], rng)
                material_total_deliveries[mat] += 1
                if delay > max_material_delay:
                    max_material_delay = delay
                    worst_material = mat
                if delay > 0:
                    material_late_deliveries[mat] += 1
                    materials_late_this_sim.add(mat)

            # Part of a late delivery is absorbed by schedule float: crews get
            # re-sequenced, partial deliveries let work start. Only the residue
            # pushes the activity's finish date.
            schedule_impact = max_material_delay * (1 - SCHEDULE_ABSORPTION)

            duration_variation = rng.normal(0, base_duration * 0.05)
            actual_duration = base_duration + schedule_impact + duration_variation
            actual_duration = max(base_duration * 0.8, actual_duration)  # Can't be too fast

            completion_times[name] = earliest_start + actual_duration
            # Report only the MATERIAL-driven slip. Including the symmetric
            # duration noise would mark ~half of every activity's runs as
            # "delayed" by a fraction of a day, and the ranking would report
            # 100% for everything.
            activity_delays[name][sim] = schedule_impact

            # Track cascade events — a delay big enough to outrun the float
            if schedule_impact > 3 and worst_material:
                cascade_events.append({
                    "simulation": sim,
                    "activity": name,
                    "material": worst_material,
                    "delay_days": round(schedule_impact, 1),
                })

        for mat in materials_late_this_sim:
            material_sim_hits[mat] += 1

        # Total project duration
        project_durations[sim] = max(completion_times.values())

        # Identify critical path for this simulation
        last_activity = max(completion_times, key=completion_times.get)
        critical_path_counts[last_activity] = critical_path_counts.get(last_activity, 0) + 1

    # ── Compute baseline (no-delay) project duration ──────────
    baseline_completions = {}
    for act in activities:
        if act["predecessors"]:
            earliest = max(baseline_completions[p] for p in act["predecessors"])
        else:
            earliest = 0
        baseline_completions[act["name"]] = earliest + act["duration_days"]
    baseline_duration = max(baseline_completions.values())

    # ── Analyze Results ───────────────────────────────────────
    elapsed = time.time() - t0

    # Percentiles for project duration
    p10 = np.percentile(project_durations, 10)
    p50 = np.percentile(project_durations, 50)
    p90 = np.percentile(project_durations, 90)
    mean_duration = np.mean(project_durations)

    # Activity risk ranking
    activity_risk = []
    for act in activities:
        name = act["name"]
        delays = activity_delays[name]
        # A "slip" is a full day or more of material-driven delay. Anything
        # smaller is inside the noise a site absorbs without noticing.
        material_slip = delays >= 1.0
        pct_delayed = np.mean(material_slip) * 100
        avg_delay = np.mean(delays[material_slip]) if np.any(material_slip) else 0
        max_delay = np.max(delays)
        activity_risk.append({
            "activity": name,
            "pct_simulations_delayed": round(pct_delayed, 1),
            "avg_delay_when_delayed": round(avg_delay, 1),
            "max_delay_observed": round(max_delay, 1),
            "materials": act["materials"],
        })
    activity_risk.sort(key=lambda x: x["pct_simulations_delayed"], reverse=True)

    # Material risk ranking — share of that material's DELIVERIES that land late
    material_risk = []
    for mat in sorted(all_materials,
                      key=lambda m: -(material_late_deliveries[m]
                                      / max(material_total_deliveries[m], 1))):
        total = material_total_deliveries[mat]
        if not total:
            continue
        late_rate = material_late_deliveries[mat] / total
        material_risk.append({
            "material": mat,
            "delay_frequency": round(late_rate * 100, 1),
            "deliveries_per_project": round(total / n_simulations, 1),
            # Share of whole projects in which this material slips at least once
            "project_impact_pct": round(material_sim_hits[mat] / n_simulations * 100, 1),
            "is_critical": late_rate > 0.35,
        })

    # Single point of failure analysis
    spof = []
    for act in activities:
        if len(act["materials"]) == 1:
            mat = act["materials"][0]
            freq = (material_late_deliveries.get(mat, 0)
                    / max(material_total_deliveries.get(mat, 1), 1))
            if freq > 0.2:
                spof.append({
                    "activity": act["name"],
                    "single_material": mat,
                    "delay_frequency": round(freq * 100, 1),
                    "recommendation": f"Dual-source {mat} or hold buffer stock on site",
                })

    # Cascade analysis: find the biggest ripple effects
    cascade_summary = {}
    for evt in cascade_events:
        mat = evt["material"]
        if mat not in cascade_summary:
            cascade_summary[mat] = {"count": 0, "total_delay": 0, "activities_affected": set()}
        cascade_summary[mat]["count"] += 1
        cascade_summary[mat]["total_delay"] += evt["delay_days"]
        cascade_summary[mat]["activities_affected"].add(evt["activity"])

    cascade_risk = []
    for mat, data in sorted(cascade_summary.items(), key=lambda x: -x[1]["count"]):
        cascade_risk.append({
            "material": mat,
            "cascade_events": data["count"],
            "avg_cascade_delay": round(data["total_delay"] / data["count"], 1),
            "activities_affected": list(data["activities_affected"])[:5],
            "cascade_multiplier": round(len(data["activities_affected"]) * 1.5, 1),
        })

    # Duration distribution for histogram
    hist_counts, hist_edges = np.histogram(project_durations, bins=30)
    duration_distribution = {
        "bin_edges": [round(e, 1) for e in hist_edges.tolist()],
        "counts": hist_counts.tolist(),
    }

    # On-time probability (within 10% of baseline)
    on_time_threshold = baseline_duration * 1.10
    on_time_probability = np.mean(project_durations <= on_time_threshold) * 100

    # ── Build Result ──────────────────────────────────────────
    result = {
        "simulation_config": {
            "project_type": project_type,
            "state": state,
            "start_month": start_month,
            "n_simulations": n_simulations,
            "elapsed_seconds": round(elapsed, 2),
            "risk_source": profile_source,     # "ml" or "physics"
            "n_model_calls": len(profile_cache) * len(all_materials),
            "schedule_absorption": SCHEDULE_ABSORPTION,
        },
        "project_timeline": {
            "baseline_duration_days": baseline_duration,
            "simulated_mean_days": round(mean_duration, 1),
            "best_case_days": round(p10, 1),
            "most_likely_days": round(p50, 1),
            "worst_case_days": round(p90, 1),
            "delay_over_baseline_days": round(p50 - baseline_duration, 1),
            "on_time_probability_pct": round(on_time_probability, 1),
        },
        "duration_distribution": duration_distribution,
        "activity_risk_ranking": activity_risk,
        "material_risk_ranking": material_risk[:10],
        "single_points_of_failure": spof,
        "cascade_risk": cascade_risk[:5],
        "critical_path_frequency": {
            k: round(v / n_simulations * 100, 1)
            for k, v in sorted(critical_path_counts.items(), key=lambda x: -x[1])
        },
        "executive_summary": _build_executive_summary(
            baseline_duration, p50, p90, on_time_probability,
            activity_risk, material_risk, cascade_risk, state
        ),
    }

    return result


def _build_executive_summary(
    baseline, p50, p90, on_time_pct,
    activity_risk, material_risk, cascade_risk, state
) -> dict:
    """Build a plain-English executive summary a site manager can understand."""

    delay_days = p50 - baseline
    risk_level = (
        "Critical" if on_time_pct < 30 else
        "High" if on_time_pct < 50 else
        "Medium" if on_time_pct < 70 else
        "Low"
    )

    # Top risk material
    top_mat = material_risk[0]["material"] if material_risk else "Unknown"
    top_mat_freq = material_risk[0]["delay_frequency"] if material_risk else 0

    # Top cascade risk
    top_cascade = cascade_risk[0] if cascade_risk else None

    headline = (
        f"Your project in {state} has a {on_time_pct:.0f}% chance of finishing on time. "
        f"Most likely, it will take {p50:.0f} days "
        f"({'+' if delay_days > 0 else ''}{delay_days:.0f} days vs the {baseline:.0f}-day plan). "
        f"In the worst 10% of scenarios, it could take {p90:.0f} days."
    )

    actions = []
    if top_mat_freq > 30:
        actions.append(
            f"Pre-order {top_mat} — {top_mat_freq:.0f}% of its deliveries land late "
            f"under these conditions"
        )
    if top_cascade:
        actions.append(
            f"Secure backup supplier for {top_cascade['material']} — delays cascade to "
            f"{len(top_cascade['activities_affected'])} downstream activities"
        )
    if on_time_pct < 50:
        actions.append(
            "Consider splitting critical material orders across 2+ suppliers to reduce single-point-of-failure risk"
        )
    if activity_risk and activity_risk[0]["pct_simulations_delayed"] > 40:
        actions.append(
            f"Add float to '{activity_risk[0]['activity']}' — late material pushes it in "
            f"{activity_risk[0]['pct_simulations_delayed']:.0f}% of runs, by "
            f"{activity_risk[0]['avg_delay_when_delayed']:.0f} days on average"
        )
    if not actions:
        actions.append("Supply chain risk is well-managed. Monitor monsoon forecasts for changes.")

    return {
        "risk_level": risk_level,
        "headline": headline,
        "recommended_actions": actions[:5],
        "key_metric": f"{on_time_pct:.0f}% on-time probability",
    }


# ── Prescriptive Optimization ────────────────────────────────

def optimize_procurement(
    materials: list,
    state: str,
    month: int,
    budget_inr: float = 5000000,
    models: dict = None,
) -> dict:
    """
    Prescriptive optimisation: choose the supplier allocation that buys the most
    stock-out risk reduction per rupee of premium over single sourcing.

    Enumerates real allocations across the supplier database, scores each on
    (cost, shortfall probability) under a correlated-failure model, and returns
    both the chosen strategy and the full non-dominated frontier.

    `models` is accepted for interface symmetry with run_simulation; the
    allocation maths runs off the supplier records and does not need them.
    """
    try:
        from suppliers_db import find_suppliers
    except ImportError:
        return _fallback_optimization(materials, state, month, budget_inr)

    results = []
    total_risk_reduction = 0

    for mat_info in materials:
        mat = mat_info["material_type"]
        qty = mat_info["quantity"]

        suppliers = find_suppliers(mat, state)
        if not suppliers:
            suppliers = find_suppliers(mat)

        if not suppliers:
            results.append({
                "material": mat,
                "strategy": "No suppliers found",
                "primary_supplier": "N/A",
                "backup_supplier": "N/A",
                "split_ratio": "100/0",
                "estimated_delay_reduction_pct": 0,
                "cost_inr": 0,
            })
            continue

        # Sort by reliability
        suppliers_sorted = sorted(suppliers, key=lambda s: -s["reliability_score"])
        primary = suppliers_sorted[0]
        backup = suppliers_sorted[1] if len(suppliers_sorted) > 1 else None

        # ── Choose the split by optimisation, not by rule of thumb ──
        # Evaluate every candidate allocation on (cost, shortfall probability)
        # and take the knee of the frontier: the most risk bought per rupee of
        # premium over single-sourcing.
        #
        # The risk metric is the probability of a COMPLETE stock-out (too little
        # usable material on site to work), not "delay risk" in general —
        # splitting an order does not make a single supplier faster.
        #
        # Two suppliers are NOT independent: the same monsoon, the same festival
        # shutdown and the same fuel shock hit both. `rho` is the share of risk
        # that is common-mode and cannot be diversified away. Assuming
        # independence is what produced the old, indefensible "97% reduction".
        primary_rel = primary["reliability_score"]
        single_stockout = 1 - primary_rel
        single_cost = qty * primary["price_index"] * 1000
        rho = COMMON_MODE_CORRELATION_BASE + monsoon_intensity(month) * 0.30

        split, strategy, delay_reduction = 1.0, "Single source (no viable backup)", 0.0
        order_cost = single_cost

        if backup:
            best_value = 0.0
            for candidate in (0.85, 0.70, 0.60, 0.50):
                cand_risk = _shortfall_probability(
                    single_stockout, 1 - backup["reliability_score"], candidate, rho
                )
                cand_cost = qty * 1000 * (
                    candidate * primary["price_index"]
                    + (1 - candidate) * backup["price_index"]
                ) * 1.02  # split handling premium
                reduction = max(
                    0.0, (single_stockout - cand_risk) / max(single_stockout, 1e-9) * 100
                )
                premium_pct = max(0.01, (cand_cost / single_cost - 1) * 100)
                value = reduction / premium_pct   # risk removed per 1% extra spend
                if value > best_value:
                    best_value, split, delay_reduction = value, candidate, reduction
                    order_cost = cand_cost

            if delay_reduction <= 0:
                split, strategy, order_cost = 1.0, "Single source (splitting adds cost, not safety)", single_cost
            elif split >= 0.80:
                strategy = "Primary-heavy split (reliable primary)"
            elif split >= 0.60:
                strategy = "Balanced split"
            else:
                strategy = "Even split (maximum hedge)"

        total_risk_reduction += delay_reduction

        results.append({
            "material": mat,
            "quantity": qty,
            "strategy": strategy,
            "primary_supplier": primary["name"],
            "primary_allocation_pct": round(split * 100),
            "backup_supplier": backup["name"] if backup and split < 1.0 else "N/A",
            "backup_allocation_pct": round((1 - split) * 100),
            "estimated_delay_reduction_pct": round(delay_reduction, 1),
            "estimated_cost_inr": round(order_cost, 0),
            "cost_premium_pct": round((order_cost / single_cost - 1) * 100, 2),
            "primary_lead_days": primary["avg_lead_days"],
        })

    pareto_points = _build_pareto_frontier(materials, state, month)
    avg_reduction = total_risk_reduction / max(len(materials), 1)

    return {
        "procurement_strategy": results,
        "total_delay_risk_reduction_pct": round(avg_reduction, 1),
        "pareto_frontier": pareto_points,
        "common_mode_correlation": round(
            COMMON_MODE_CORRELATION_BASE + monsoon_intensity(month) * 0.30, 2
        ),
        "recommendation": (
            f"Splitting these orders across a primary and a backup supplier cuts the "
            f"probability of a complete on-site stock-out by {avg_reduction:.0f}%. "
            f"That is the diversifiable portion only — "
            f"{(COMMON_MODE_CORRELATION_BASE + monsoon_intensity(month) * 0.30):.0%} of the "
            f"risk this month is common-mode (monsoon and corridor-wide disruption) and "
            f"cannot be hedged by supplier choice; that portion needs schedule buffer instead."
        ),
    }


def _shortfall_probability(p1: float, p2: float, split: float, rho: float,
                           usable_threshold: float = 0.5) -> float:
    """
    Probability that less than `usable_threshold` of the order lands on time,
    given allocation `split` to the primary supplier.

    Uses a common-shock decomposition: with probability rho_c BOTH suppliers are
    hit by the same corridor-wide event; otherwise each fails independently.
    This is what makes an 85/15 split barely safer than single sourcing (losing
    the primary still starves the site) while a 50/50 split genuinely hedges.
    """
    rho_c = rho * min(p1, p2)
    denom = max(1e-9, 1 - rho_c)
    q1 = min(1.0, max(0.0, (p1 - rho_c) / denom))
    q2 = min(1.0, max(0.0, (p2 - rho_c) / denom))

    p_both = rho_c + (1 - rho_c) * q1 * q2
    p_only_primary_late = (1 - rho_c) * q1 * (1 - q2)
    p_only_backup_late = (1 - rho_c) * q2 * (1 - q1)

    risk = p_both
    if (1 - split) < usable_threshold:      # losing the primary starves the site
        risk += p_only_primary_late
    if split < usable_threshold:            # losing the backup starves the site
        risk += p_only_backup_late
    return float(min(1.0, risk))


def _build_pareto_frontier(materials: list, state: str, month: int) -> list:
    """
    Enumerate real supplier allocations and return the non-dominated
    (cost, stock-out risk) set.

    This is an actual Pareto frontier computed over the supplier database, not a
    drawn curve: every point corresponds to a concrete, nameable allocation the
    buyer can execute.
    """
    from suppliers_db import find_suppliers

    rho = COMMON_MODE_CORRELATION_BASE + monsoon_intensity(month) * 0.30
    splits = [1.0, 0.85, 0.70, 0.60, 0.50]

    candidates = []
    for split in splits:
        total_cost = 0.0
        total_risk = 0.0
        counted = 0
        for mat_info in materials:
            suppliers = find_suppliers(mat_info["material_type"], state)[:5]
            if not suppliers:
                continue
            qty = mat_info["quantity"]
            primary = suppliers[0]
            backup = suppliers[1] if len(suppliers) > 1 else None

            if split >= 1.0 or backup is None:
                cost = qty * primary["price_index"] * 1000
                risk = 1 - primary["reliability_score"]
            else:
                cost = qty * 1000 * (
                    split * primary["price_index"] + (1 - split) * backup["price_index"]
                )
                # Splitting also costs a small handling/mobilisation premium
                cost *= 1.02
                risk = _shortfall_probability(
                    1 - primary["reliability_score"],
                    1 - backup["reliability_score"],
                    split, rho,
                )
            total_cost += cost
            total_risk += risk
            counted += 1

        if counted:
            candidates.append({
                "split": split,
                "cost": total_cost,
                "risk": total_risk / counted,
            })

    if not candidates:
        return []

    # Keep only non-dominated points: an option stays on the frontier unless
    # some other option is at least as cheap AND at least as safe, and strictly
    # better on one of the two.
    def dominated(c):
        return any(
            o is not c
            and o["cost"] <= c["cost"] and o["risk"] <= c["risk"]
            and (o["cost"] < c["cost"] or o["risk"] < c["risk"])
            for o in candidates
        )

    frontier = {id(c) for c in candidates if not dominated(c)}
    candidates.sort(key=lambda c: c["cost"])

    base_cost = min(c["cost"] for c in candidates)
    base_risk = max(c["risk"] for c in candidates)

    points = []
    for c in candidates:
        points.append({
            "on_frontier": id(c) in frontier,
            "extra_spend_pct": round((c["cost"] / base_cost - 1) * 100, 2),
            "risk_reduction_pct": round(
                (base_risk - c["risk"]) / max(base_risk, 1e-9) * 100, 1
            ),
            "stockout_probability_pct": round(c["risk"] * 100, 1),
            "estimated_cost_inr": round(c["cost"], 0),
            "label": (
                "Single source (cheapest)" if c["split"] >= 1.0
                else f"{round(c['split'] * 100)}/{round((1 - c['split']) * 100)} split"
            ),
        })
    return points


def _fallback_optimization(materials, state, month, budget):
    """Simple optimization when scipy/suppliers_db not available."""
    results = []
    for mat_info in materials:
        mat = mat_info["material_type"]
        qty = mat_info["quantity"]
        results.append({
            "material": mat,
            "quantity": qty,
            "strategy": "Order early and split across 2 suppliers",
            "primary_supplier": f"Best available {mat} supplier",
            "primary_allocation_pct": 70,
            "backup_supplier": "Second-best supplier",
            "backup_allocation_pct": 30,
            "estimated_delay_reduction_pct": 25.0,
            "estimated_cost_inr": qty * 1200,
            "primary_lead_days": 14,
        })
    return {
        "procurement_strategy": results,
        "total_delay_risk_reduction_pct": 25.0,
        "pareto_frontier": [],
        "recommendation": "Split critical material orders across 2+ suppliers to reduce delay risk by ~25%.",
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NirmanAI — Monte Carlo Supply Chain Simulation")
    print("=" * 60)

    print("\nRunning 5,000 simulations for a Residential Apartment in Bihar (July)...")
    result = run_simulation(
        project_type="Residential Apartment",
        state="Bihar",
        start_month=7,
        n_simulations=5000,
    )

    tl = result["project_timeline"]
    summary = result["executive_summary"]

    print(f"\n--- Project Timeline ---")
    print(f"  Baseline (no delays):  {tl['baseline_duration_days']} days")
    print(f"  Best case (P10):       {tl['best_case_days']} days")
    print(f"  Most likely (P50):     {tl['most_likely_days']} days")
    print(f"  Worst case (P90):      {tl['worst_case_days']} days")
    print(f"  On-time probability:   {tl['on_time_probability_pct']}%")
    print(f"  Risk Level:            {summary['risk_level']}")

    print(f"\n--- Executive Summary ---")
    print(f"  {summary['headline']}")

    print(f"\n--- Top Risk Activities ---")
    for act in result["activity_risk_ranking"][:5]:
        print(f"  {act['activity']}: delayed in {act['pct_simulations_delayed']}% of sims "
              f"(avg {act['avg_delay_when_delayed']} days)")

    print(f"\n--- Material Risk ---")
    for mat in result["material_risk_ranking"][:5]:
        crit = " ** CRITICAL **" if mat["is_critical"] else ""
        print(f"  {mat['material']}: causes delays {mat['delay_frequency']}% of the time{crit}")

    if result["single_points_of_failure"]:
        print(f"\n--- Single Points of Failure ---")
        for spof in result["single_points_of_failure"]:
            print(f"  {spof['activity']} depends solely on {spof['single_material']} "
                  f"(delayed {spof['delay_frequency']}%)")

    print(f"\n--- Recommended Actions ---")
    for i, action in enumerate(summary["recommended_actions"], 1):
        print(f"  {i}. {action}")

    print(f"\n--- Optimization ---")
    opt = optimize_procurement(
        materials=[
            {"material_type": "TMT Steel", "quantity": 80},
            {"material_type": "OPC Cement", "quantity": 500},
            {"material_type": "River Sand", "quantity": 300},
        ],
        state="Bihar",
        month=7,
    )
    print(f"  Overall delay risk reduction: {opt['total_delay_risk_reduction_pct']}%")
    print(f"  Recommendation: {opt['recommendation']}")

    print(f"\n  Simulation completed in {result['simulation_config']['elapsed_seconds']:.1f}s")
    print("[OK] Simulation engine ready.")
