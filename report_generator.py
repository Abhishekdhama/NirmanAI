"""
NirmanAI — Site Supply Risk Brief generator
===========================================
Renders the one-page report a site engineer carries into the morning meeting.

Data comes from the same engines as the dashboard: the Monte Carlo simulator,
the wastage model and the supplier database. When those are unavailable the
report still renders, but it says so on the page rather than silently printing
placeholder numbers.
"""

import datetime
import os

from jinja2 import Environment, FileSystemLoader

try:
    from weasyprint import HTML
    WEASYPRINT_AVAILABLE = True
except Exception:
    # WeasyPrint needs native pango/cairo libraries. Missing them must degrade
    # to HTML, never crash the export.
    WEASYPRINT_AVAILABLE = False

MONTHS = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
          "July": 7, "August": 8, "September": 9, "October": 10,
          "November": 11, "December": 12}

DEFAULT_BOQ = {"OPC Cement": 800, "River Sand": 450, "TMT Steel": 60,
               "Fly Ash Bricks": 28000, "Vitrified Tiles": 1400, "Plywood": 700}


def _fmt_inr(value: float) -> str:
    """
    Full rupees with Indian digit grouping. One unit per column — a table that
    mixes "Rs 76,529" and "Rs 2.50 L" cannot be scanned.
    """
    value = float(value)
    sign = "-" if value < 0 else ""
    digits = f"{abs(value):.0f}"
    if len(digits) <= 3:
        return f"{sign}Rs {digits}"
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return f"{sign}Rs {','.join(groups)},{tail}"


def _month_int(current_month) -> int:
    if isinstance(current_month, int):
        return current_month
    return MONTHS.get(str(current_month), 7)


def generate_report_data(project_name, project_type, state, current_month,
                         models=None, boq=None, site_conditions=None):
    """
    Build the report context from the live engines.

    Returns a dict ready for the Jinja template. Falls back to a clearly
    labelled unavailable state rather than to invented numbers.
    """
    month_int = _month_int(current_month)
    boq = boq or DEFAULT_BOQ
    site = {
        "project_type": project_type, "state": state, "project_size": 12000,
        "month": month_int, "contractor_exp": 8,
        "workforce_skill": "Semi-skilled", "supervision": "Average",
    }
    site.update(site_conditions or {})

    context = {
        "executive_summary": "",
        "timeline": None,
        "n_simulations": 0,
        "weather": {"condition": "Not available", "impact": "Not available"},
        "risk_deliveries": [],
        "wastage_forecasts": [],
        "recommended_actions": [],
        "total_overrun": None,
        "on_time_class": "",
        "provenance": "",
    }

    # ── 1. Project simulation ────────────────────────────────
    sim = None
    try:
        from simulation_engine import run_simulation
        sim = run_simulation(project_type, state, month_int,
                             n_simulations=3000, models=models)
    except Exception as exc:
        print(f"[report] Simulation unavailable: {type(exc).__name__}: {exc}")

    if sim:
        tl = sim["project_timeline"]
        summary = sim["executive_summary"]
        on_time = tl["on_time_probability_pct"]
        context["executive_summary"] = summary["headline"]
        context["timeline"] = tl
        context["n_simulations"] = f"{sim['simulation_config']['n_simulations']:,}"
        context["recommended_actions"] = summary.get("recommended_actions", [])
        context["on_time_class"] = (
            "risk-green" if on_time >= 70 else
            "risk-yellow" if on_time >= 50 else
            "risk-orange" if on_time >= 30 else "risk-red")
        context["risk_deliveries"] = _risk_deliveries(sim, state)
    else:
        context["executive_summary"] = (
            "Simulation engine unavailable for this run — the timeline section of this "
            "report could not be produced. Re-run `python setup.py` and regenerate.")
        context["recommended_actions"] = [
            "Re-run the setup pipeline so the models and simulator are available.",
        ]

    # ── 2. Weather ───────────────────────────────────────────
    context["weather"] = _weather(state, month_int)

    # ── 3. Wastage ───────────────────────────────────────────
    try:
        from demo_data import build_wastage_forecast
        wf = build_wastage_forecast(models, boq, site)
        rows, total = [], 0.0
        for _, r in wf.iterrows():
            total += float(r["cost_overrun"])
            rows.append({
                "material": r["material"],
                "blueprint": f"{r['blueprint_qty']:,.0f}",
                "order_qty": f"{r['order_qty']:,.0f}",
                "forecast": f"{r['wastage_pct']:.1f}",
                "status": r["category"],
                "status_color": {"High": "red", "Medium": "orange",
                                 "Low": "green"}.get(r["category"], "orange"),
                "cost": _fmt_inr(r["cost_overrun"]),
            })
        context["wastage_forecasts"] = rows
        context["total_overrun"] = _fmt_inr(total) if rows else None
    except Exception as exc:
        print(f"[report] Wastage forecast unavailable: {type(exc).__name__}: {exc}")

    # ── 4. Provenance line ───────────────────────────────────
    source = "trained XGBoost/LightGBM models" if models else \
             "transparent rule-based fallback (models not trained)"
    context["provenance"] = (
        f"Delay and wastage figures produced by the {source}. "
        "Purchase orders and training data are synthetic; supplier records are curated.")

    return context


def _risk_deliveries(sim, state):
    """Materials ranked by risk, each paired with a real recommended supplier."""
    try:
        from suppliers_db import find_suppliers
    except Exception:
        find_suppliers = None

    rows = []
    for m in sim.get("material_risk_ranking", [])[:7]:
        late = m["delay_frequency"]
        level = "High" if late > 50 else "Medium" if late > 25 else "Low"
        color = {"High": "red", "Medium": "orange", "Low": "green"}[level]

        supplier_name = "No supplier on file"
        if find_suppliers:
            matches = find_suppliers(m["material"], state)
            if matches:
                best = matches[0]
                supplier_name = (f"{best['name']} "
                                 f"({best['reliability_score']:.0%} reliable, "
                                 f"{best['avg_lead_days']}d lead)")

        rows.append({
            "material": m["material"],
            "supplier": supplier_name,
            "risk_level": level,
            "risk_level_color": color,
            "late_pct": f"{late:.0f}",
            "reason": (f"Slips in {m['project_impact_pct']:.0f}% of projects across its "
                       f"{m['deliveries_per_project']:.0f} scheduled deliveries"),
        })
    return rows


def _weather(state, month_int):
    """Live weather when an API key is configured, seasonal profile otherwise."""
    try:
        from weather import get_weather_risk_summary
        risk = get_weather_risk_summary(state)
        if risk and risk.get("source") == "live":
            return {"condition": risk.get("description", "Unknown"),
                    "impact": f"Estimated delivery delay impact: "
                              f"+{risk.get('impact_on_delivery', 0)}%"}
    except Exception:
        pass

    from demo_data import monsoon_intensity
    m_int = monsoon_intensity(month_int)
    if m_int > 0.6:
        condition = "Peak monsoon — heavy, sustained rainfall expected"
        impact = ("Aggregate and sand haulage disrupted; cement curing delayed; "
                  "state highways prone to washouts and diversions.")
    elif m_int > 0.2:
        condition = "Shoulder monsoon — intermittent heavy rain"
        impact = "Intermittent transit delays; moisture-sensitive materials at risk."
    elif month_int in (10, 11):
        condition = "Post-monsoon, festival shutdown window"
        impact = ("Diwali and Chhath close plants and depots for up to a week; "
                  "trucking capacity tightens sharply.")
    else:
        condition = "Dry season — favourable transport conditions"
        impact = "Minimal weather-driven delay risk; the best window to place bulk orders."

    return {"condition": f"{condition} ({state}, seasonal profile)", "impact": impact}


def generate_pdf_report(report_id, project_name, project_type, state, current_month,
                        output_dir="reports/generated", models=None,
                        boq=None, site_conditions=None):
    """
    Render the brief to PDF, falling back to HTML when WeasyPrint's native
    dependencies are missing. Returns the path actually written, so the caller
    can label the download correctly.
    """
    os.makedirs(output_dir, exist_ok=True)

    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template("report.html")

    context = generate_report_data(project_name, project_type, state, current_month,
                                   models, boq=boq, site_conditions=site_conditions)
    context.update({
        "report_id": report_id,
        "project_name": project_name,
        "project_type": project_type,
        "state": state,
        "current_month": current_month,
        "generation_date": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
    })

    rendered = template.render(context)

    if WEASYPRINT_AVAILABLE:
        path = os.path.join(output_dir, f"{report_id}.pdf")
        try:
            HTML(string=rendered, base_url=__file__).write_pdf(path)
            print(f"[report] PDF written -> {path}")
            return path
        except Exception as exc:
            print(f"[report] WeasyPrint failed ({exc}); falling back to HTML.")
    else:
        print("[report] WeasyPrint unavailable; exporting HTML.")

    path = os.path.join(output_dir, f"{report_id}.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    print(f"[report] HTML written -> {path}")
    return path


if __name__ == "__main__":
    from model_store import load_models

    out = generate_pdf_report(
        report_id="RPT-DEMO-001",
        project_name="Ganga Riverside — Tower B",
        project_type="Residential Apartment",
        state="Bihar",
        current_month="July",
        output_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reports", "generated"),
        models=load_models(),
    )
    print(f"Report saved at: {out}")
