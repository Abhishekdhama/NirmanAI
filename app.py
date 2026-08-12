"""
NirmanAI — Site Supply Risk Console
===================================
Streamlit dashboard for construction supply-chain risk.

Run: streamlit run app.py

Every number on screen is produced by the trained models or the simulation
engine at request time. Where a panel cannot be model-backed (the KAYA
integration concept), it is labelled as such on the panel itself.
"""

import traceback
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import ui
from demo_data import (
    MATERIAL_PRICES,
    STATE_LOGISTICS,
    build_alerts,
    build_order_book,
    build_wastage_forecast,
    monsoon_intensity,
    order_book_kpis,
)

st.set_page_config(
    page_title="NirmanAI — Site Supply Risk Console",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.CSS, unsafe_allow_html=True)

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

STATES = ["Maharashtra", "Tamil Nadu", "Karnataka", "Gujarat", "Rajasthan",
          "Uttar Pradesh", "Bihar", "West Bengal", "Madhya Pradesh",
          "Telangana", "Kerala", "Punjab", "Odisha", "Jharkhand", "Haryana"]

PROJECT_TYPES = ["Residential Apartment", "Commercial Complex", "Industrial Warehouse",
                 "Road & Highway", "Bridge Construction", "Metro Rail",
                 "Data Center", "Hospital", "School/College"]

MATERIALS = list(MATERIAL_PRICES.keys())

# Two pre-wired scenarios. The contrast between them is the demo: identical
# product, opposite risk picture, driven entirely by the model.
SCENARIOS = {
    "Monsoon crunch — Patna, Bihar (July)": dict(
        project_name="Ganga Riverside — Tower B", project_type="Residential Apartment",
        state="Bihar", project_size=12000, month=7,
        workforce_skill="Semi-skilled", supervision="Poor", contractor_exp=4,
    ),
    "Dry season — Ahmedabad, Gujarat (February)": dict(
        project_name="Sabarmati Tech Park — Phase 1", project_type="Commercial Complex",
        state="Gujarat", project_size=25000, month=2,
        workforce_skill="Skilled", supervision="Good", contractor_exp=15,
    ),
    "Festival window — Lucknow, UP (October)": dict(
        project_name="Gomti Nagar Metro — Depot", project_type="Metro Rail",
        state="Uttar Pradesh", project_size=40000, month=10,
        workforce_skill="Skilled", supervision="Average", contractor_exp=9,
    ),
}


# ══════════════════════════════════════════════════════════════
# Infrastructure: model loading, caching, error containment
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading NirmanAI models…")
def get_models():
    from model_store import load_models
    return load_models()


@contextmanager
def safe_panel(what: str):
    """
    Contain a failure to one panel instead of blanking the whole page.

    A hackathon demo that shows a red traceback is over. This keeps the rest of
    the dashboard usable and tells the user exactly what to do.
    """
    try:
        yield
    except Exception as exc:
        st.error(f"**{what} is unavailable.** {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc(), language="text")
        st.caption("Everything else on this page still works. "
                   "Re-run `python setup.py` if this persists.")


@st.cache_data(show_spinner=False, ttl=600)
def cached_order_book(state: str, month: int, project_name: str, models_ready: bool):
    return build_order_book(get_models() if models_ready else None,
                            state, month, project_name)


@st.cache_data(show_spinner=False, ttl=600)
def cached_wastage(materials_qty: dict, ctx: dict, models_ready: bool):
    return build_wastage_forecast(get_models() if models_ready else None,
                                  materials_qty, ctx)


@st.cache_data(show_spinner=False, ttl=600)
def cached_simulation(project_type: str, state: str, month: int, n: int, models_ready: bool):
    from simulation_engine import run_simulation
    return run_simulation(project_type, state, month, n,
                          models=get_models() if models_ready else None)


@st.cache_data(show_spinner=False, ttl=600)
def cached_monthly_curve(state: str, models_ready: bool):
    """
    Average delay probability by month for this state, scored by the model.

    Replaces what used to be a hardcoded list of twelve numbers that never
    changed when you changed the state.
    """
    from demo_data import build_order_book as _book
    models = get_models() if models_ready else None
    return [
        float(_book(models, state, m, "seasonality-probe", n_orders=12)
              ["delay_probability"].mean())
        for m in range(1, 13)
    ]


def fmt_inr(value: float) -> str:
    """
    Headline format: crore / lakh, the way an Indian finance team says a number
    out loud. Use for single KPI tiles only — see fmt_inr_full for tables.
    """
    value = float(value)
    if abs(value) >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    if abs(value) >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    return f"₹{value:,.0f}"


def fmt_inr_full(value: float) -> str:
    """
    Table format: full rupees with Indian digit grouping (₹2,50,431).

    Columns must use ONE unit. Mixing "₹76,529" and "₹2.50 L" in the same column
    means the reader has to convert in their head to tell which row is bigger,
    which defeats the point of a table.
    """
    value = float(value)
    sign = "-" if value < 0 else ""
    digits = f"{abs(value):.0f}"
    if len(digits) <= 3:
        return f"{sign}₹{digits}"
    # Last three digits, then groups of two (Indian lakh/crore convention)
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while len(head) > 2:
        groups.insert(0, head[-2:])
        head = head[:-2]
    if head:
        groups.insert(0, head)
    return f"{sign}₹{','.join(groups)},{tail}"


MODELS = get_models()
MODELS_READY = MODELS is not None
METRICS = (MODELS or {}).get("metrics", {}) or {}


# ══════════════════════════════════════════════════════════════
# Sidebar — the single input surface
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-heading">Start here</div>', unsafe_allow_html=True)
    scenario = st.selectbox(
        "Load a project", list(SCENARIOS.keys()),
        help="Pre-configured sites. Everything below stays editable — "
             "change the state or month and every prediction on the page re-runs.",
    )
    preset = SCENARIOS[scenario]

    st.markdown('<div class="sidebar-heading">Project</div>', unsafe_allow_html=True)
    project_name = st.text_input("Project name", preset["project_name"])
    project_type = st.selectbox("Project type", PROJECT_TYPES,
                                index=PROJECT_TYPES.index(preset["project_type"]))
    state = st.selectbox("Site location (state)", STATES,
                         index=STATES.index(preset["state"]),
                         help="Drives logistics quality, monsoon severity and "
                              "haulage distance from every supplier.")
    project_size = st.slider("Built-up area (sq ft)", 1000, 60000,
                             preset["project_size"], 1000)
    month_name = st.selectbox("Procurement month", MONTHS, index=preset["month"] - 1,
                              help="The single biggest driver of delivery risk in India.")
    current_month = MONTHS.index(month_name) + 1

    st.markdown('<div class="sidebar-heading">Site conditions</div>', unsafe_allow_html=True)
    workforce_skill = st.selectbox("Workforce skill", ["Unskilled", "Semi-skilled", "Skilled", "Expert"],
                                   index=["Unskilled", "Semi-skilled", "Skilled", "Expert"]
                                   .index(preset["workforce_skill"]))
    supervision = st.selectbox("Supervision quality", ["Poor", "Average", "Good", "Excellent"],
                               index=["Poor", "Average", "Good", "Excellent"]
                               .index(preset["supervision"]))
    contractor_exp = st.slider("Contractor experience (years)", 1, 30,
                               preset["contractor_exp"])

    m_int = monsoon_intensity(current_month)
    logistics = STATE_LOGISTICS.get(state, 0.6)
    st.markdown("---")
    weather_icon = "🌧️" if m_int > 0.6 else "🌦️" if m_int > 0.2 else "☀️"
    festival = ("<br>Festival shutdown window"
                if current_month in (10, 11) else "")
    # Built as one line per element: Streamlit runs this through a markdown
    # parser first, and indented newlines inside a <p> make it close the tag
    # early, leaving a literal "</p>" visible in the sidebar.
    st.markdown(
        f'<div class="panel" style="margin:0;">'
        f'<p class="panel-title">{weather_icon} {month_name} conditions</p>'
        f'<p class="panel-note">'
        f'Monsoon intensity <b style="color:var(--amber)">{m_int:.0%}</b><br>'
        f'{state} logistics index <b style="color:var(--amber)">{logistics:.2f}</b>'
        f'{festival}</p></div>',
        unsafe_allow_html=True,
    )

PROJECT_CTX = dict(
    project_type=project_type, state=state, project_size=project_size,
    month=current_month, contractor_exp=contractor_exp,
    workforce_skill=workforce_skill, supervision=supervision,
)


# ══════════════════════════════════════════════════════════════
# Masthead
# ══════════════════════════════════════════════════════════════

if MODELS_READY:
    auc = METRICS.get("auc")
    chips = (
        f'<span class="chip chip-live">● Models live</span>'
        f'<span class="chip">XGBoost + LightGBM</span>'
        + (f'<span class="chip">AUC {auc:.3f}</span>' if auc else "")
        + '<span class="chip">Conformal intervals</span>'
        '<span class="chip">SHAP per order</span>'
    )
else:
    chips = ('<span class="chip chip-warn">⚠ Heuristic mode</span>'
             '<span class="chip">Run <code>python setup.py</code> to train</span>')

st.markdown(ui.masthead(chips), unsafe_allow_html=True)

if not MODELS_READY:
    st.warning(
        "**Trained models not found — the dashboard is running on a transparent "
        "rule-based fallback.** Every figure below is still computed live, but it is "
        "not a model output. Run `python setup.py` (about two minutes) to train them.",
        icon="⚠️",
    )

# Scored order book drives the masthead proof line and the whole first tab.
order_book = pd.DataFrame()
kpis = order_book_kpis(order_book)
with safe_panel("The live order book"):
    order_book = cached_order_book(state, current_month, project_name, MODELS_READY)
    kpis = order_book_kpis(order_book)

if kpis["total"]:
    risk_line = (
        f"<strong>{kpis['at_risk']} of {kpis['total']}</strong> open orders are more likely "
        f"than not to arrive late, putting <strong>{fmt_inr(kpis['value_at_risk'])}</strong> "
        f"and <strong>{kpis['exposed_days']:.0f} crew-days</strong> at risk."
    )
else:
    risk_line = "Select a project in the sidebar to score its open orders."

if MODELS_READY and METRICS.get("bayes_ceiling_auc"):
    proof_line = (
        f"AUC <strong>{METRICS['auc']:.3f}</strong> against a theoretical ceiling of "
        f"{METRICS['bayes_ceiling_auc']:.3f} — <strong>"
        f"{METRICS['signal_captured_pct']:.0f}%</strong> of the learnable signal. "
        f"Intervals are conformal, with {METRICS.get('conformal_empirical_coverage_pct', 90):.0f}% "
        "measured coverage."
    )
else:
    proof_line = ("Every prediction ships with a calibrated interval and the SHAP factors "
                  "the model actually used — no unexplained scores.")

st.markdown(ui.premise_band(risk_line, proof_line), unsafe_allow_html=True)

with st.expander("📐 How this works, and what is real — the honest version"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**The models**

- **Delay classifier** — XGBoost, {METRICS.get('n_training_rows', 5000):,} orders,
  38 features. Test AUC **{METRICS.get('auc', 0):.3f}**.
- **Delay magnitude** — LightGBM quantile regressor, MAE
  **{METRICS.get('regressor_mae_days', 0):.1f} days** on late orders.
- **Uncertainty** — conformal prediction, target 90% coverage,
  **{METRICS.get('conformal_empirical_coverage_pct', 0):.1f}%** measured on held-out data.
- **Wastage** — LightGBM regressor plus 10th/90th percentile quantile models.
- **Explanations** — SHAP TreeExplainer, computed per order, not a rule list.

Delivery outcomes are partly random, so no classifier can reach AUC 1.0. The
**Bayes-optimal ceiling here is {METRICS.get('bayes_ceiling_auc', 0):.3f}** — we
report it next to our score rather than quoting an impressive-sounding number
without context.
        """)
    with c2:
        st.markdown("""
**What is synthetic, and what is not**

- **Purchase orders are synthetic.** We have no ERP to read from. They are
  generated deterministically from the project you pick, then scored by the
  real models — change the state or month and every row re-scores.
- **Suppliers are a curated database** of 71 real Indian material suppliers
  with hand-assigned reliability and lead times. Distances are computed from
  state-capital coordinates, not invented per call.
- **Training data is synthetic**, generated from published benchmarks (CIDC
  wastage ranges, IMD monsoon profiles, state logistics indices, the festival
  calendar). The relationships are domain knowledge; the rows are not real
  procurement records. We do not claim otherwise.
- **The KAYA Jarvis panel is an integration concept**, clearly labelled, and
  is the only non-live element in this dashboard.
        """)

st.markdown("")


# ══════════════════════════════════════════════════════════════
# Tabs — numbered to make the intended path obvious
# ══════════════════════════════════════════════════════════════

tab_radar, tab_order, tab_waste, tab_sim, tab_plan, tab_agent = st.tabs([
    "1 · Risk Radar",
    "2 · Check an Order",
    "3 · Wastage & Cost",
    "4 · Project Simulator",
    "5 · Plan & Report",
    "🤖 Ask Jarvis",
])


# ══════════════════════════════════════════════════════════════
# 1 · RISK RADAR
# ══════════════════════════════════════════════════════════════

with tab_radar:
    st.markdown(ui.section(
        "", "Risk Radar",
        f"Every open purchase order for <b>{project_name}</b>, scored by the delay model "
        f"for {month_name} conditions in {state}."), unsafe_allow_html=True)

    if order_book.empty:
        st.markdown(ui.empty_state(
            "📭", "No orders to score",
            "Pick a project in the sidebar to generate and score its order book."),
            unsafe_allow_html=True)
    else:
      with safe_panel("The Risk Radar"):
        avg_p = kpis["avg_delay_prob"]
        tiles = [
            ("Open orders", f"{kpis['total']}", f"{project_type} · {state}", "#eef2f9"),
            ("Likely to slip", f"{kpis['at_risk']}",
             f"{kpis['at_risk'] / kpis['total']:.0%} of the book", ui.RISK_COLORS["High"]),
            ("Critical", f"{kpis['critical']}",
             "≥75% delay probability", ui.RISK_COLORS["Critical"]),
            ("Value at risk", fmt_inr(kpis["value_at_risk"]),
             "Order value on at-risk POs", ui.RISK_COLORS["Medium"]),
            ("Schedule exposure", f"{kpis['exposed_days']:.0f} d",
             "Probability-weighted crew-days", ui.RISK_COLORS["High"]
             if kpis["exposed_days"] > 40 else ui.RISK_COLORS["Low"]),
        ]
        for col, (label, value, sub, color) in zip(st.columns(5), tiles):
            col.markdown(ui.kpi(label, value, sub, color), unsafe_allow_html=True)

        st.markdown(ui.why(
            "<b>Schedule exposure</b> is the sum of (delay probability × expected days late) "
            "across every order — the crew-days you should expect to lose this month if you "
            "change nothing. It is the number that converts risk into money."),
            unsafe_allow_html=True)

        alerts = build_alerts(order_book, top_n=3)
        if alerts:
            st.markdown(f"#### ⚠️ Act on these {len(alerts)} first")
            for a in alerts:
                st.markdown(ui.alert_card(a), unsafe_allow_html=True)
        else:
            st.success(
                f"**No high-risk orders.** In {month_name}, {state}'s logistics conditions "
                f"put every open order below the 55% delay threshold. Average risk across "
                f"the book is {avg_p:.0%}.", icon="✅")

        st.markdown("#### Full order book")
        display = order_book.copy()
        display["Risk"] = display["risk_label"]
        # ProgressColumn formats the raw value, so a 0-1 probability with a
        # "%.0f%%" format renders 0.87 as "1%". Store it as 0-100.
        display["Delay risk"] = display["delay_probability"] * 100
        display["If late"] = display["conditional_delay_days"].map(lambda d: f"{d:.0f} d")
        display["90% interval"] = display.apply(
            lambda r: ui.format_interval(r["ci_lower"], r["ci_upper"]), axis=1)
        display["Value"] = display["order_value_inr"].map(fmt_inr_full)
        display["Top driver"] = display["top_risk_factors"].map(
            lambda f: f[0] if f else "—")
        table = display[["po_id", "material", "supplier", "route", "distance_km",
                         "Risk", "Delay risk", "If late", "90% interval", "Value",
                         "Top driver"]].rename(columns={
                             "po_id": "PO", "material": "Material", "supplier": "Supplier",
                             "route": "Route", "distance_km": "km"})

        st.dataframe(
            table,
            use_container_width=True, hide_index=True, height=420,
            column_config={
                "Delay risk": st.column_config.ProgressColumn(
                    "Delay risk", format="%.0f%%", min_value=0, max_value=100),
                "km": st.column_config.NumberColumn("km", format="%d"),
            },
        )
        st.caption("Sorted by risk. Every row is a live model call — the driver column is "
                   "the top positive SHAP contributor for that specific order.")

        col_a, col_b = st.columns(2)
        with col_a:
            with safe_panel("The risk mix chart"):
                counts = order_book["risk_label"].value_counts()
                ordered = [r for r in ["Critical", "High", "Medium", "Low"] if r in counts]
                fig = go.Figure(go.Pie(
                    labels=ordered, values=[counts[r] for r in ordered], hole=0.58,
                    marker=dict(colors=[ui.RISK_COLORS[r] for r in ordered],
                                line=dict(color="#131823", width=2)),
                    textinfo="label+value", textfont=dict(size=12),
                ))
                fig.add_annotation(text=f"<b>{kpis['total']}</b><br>orders",
                                   showarrow=False, font=dict(size=15, color="#eef2f9"))
                ui.apply_plot_theme(fig, height=330, title="Risk mix across the order book")
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            with safe_panel("The seasonality chart"):
                curve = cached_monthly_curve(state, MODELS_READY)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=MONTH_ABBR, y=[p * 100 for p in curve],
                    mode="lines+markers", fill="tozeroy",
                    line=dict(color=ui.RISK_COLORS["High"], width=2.5, shape="spline"),
                    fillcolor="rgba(255,140,26,0.13)",
                    marker=dict(size=7, color=[
                        ui.RISK_COLORS["Critical"] if p > 0.6 else
                        ui.RISK_COLORS["High"] if p > 0.45 else
                        ui.RISK_COLORS["Medium"] if p > 0.3 else ui.RISK_COLORS["Low"]
                        for p in curve]),
                    hovertemplate="%{x}: %{y:.0f}% avg delay risk<extra></extra>",
                ))
                fig.add_vline(x=current_month - 1, line_dash="dot",
                              line_color="rgba(255,255,255,0.45)",
                              annotation_text="You are here",
                              annotation_font_size=11)
                ui.apply_plot_theme(fig, height=330,
                                    title=f"When to buy in {state} — model-scored by month")
                fig.update_layout(yaxis=dict(title="Avg delay probability (%)",
                                             range=[0, 100],
                                             gridcolor="rgba(255,255,255,0.06)"),
                                  showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Each point re-scores a 12-order basket for {state} in that "
                           "month. The shape is the model's, not a drawn curve.")

    with st.expander("🔗 KAYA Jarvis integration — concept, not live"):
        st.info("This panel is a mock-up of how NirmanAI risk scores would drive KAYA's "
                "outbound agent. It is the only illustrative element in this dashboard.",
                icon="ℹ️")
        if not order_book.empty:
            top = order_book.iloc[0]
            st.markdown(f"""
> **Jarvis (proposed automation)**
> "NirmanAI scored **{top['po_id']} — {top['material']}** from {top['supplier']}
> at **{top['delay_probability']:.0%}** delay risk
> ({top['top_risk_factors'][0].lower() if len(top['top_risk_factors']) else 'multiple factors'}).
> I would contact two alternate suppliers on this corridor and hold a
> {max(3, int(top['conditional_delay_days']))}-day buffer."

The risk score, supplier and reasoning above are real model output for the
currently selected project. Only the outbound action is hypothetical.
            """)


# ══════════════════════════════════════════════════════════════
# 2 · CHECK AN ORDER
# ══════════════════════════════════════════════════════════════

with tab_order:
    st.markdown(ui.section(
        "Step 2", "Check a specific order",
        "", "Check a specific order",
        "Score any order before you place it. You get a probability, a calibrated "
        "day range, the drivers behind it, and the wastage buffer to add."),
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        material = st.selectbox("Material", MATERIALS, index=MATERIALS.index("River Sand"))
        sup_tier = st.selectbox("Supplier tier", [
            "Tier 1 (Large Manufacturer)",
            "Tier 2 (Regional Distributor)",
            "Tier 3 (Local Supplier)"], index=1)
    with c2:
        origin = st.selectbox("Ships from", STATES, index=STATES.index("Rajasthan"))
        quantity = st.number_input("Quantity", 10, 50000, 300, step=10,
                                   help=f"Priced at ₹{MATERIAL_PRICES.get(material, 500):,}/unit")
    with c3:
        past_del = st.slider("Supplier's past delay rate", 0.02, 0.70, 0.35, 0.01,
                             help="From your own GRN history, or the supplier database.")
        order_month = MONTHS.index(
            st.selectbox("Delivery month", MONTHS, index=current_month - 1,
                         key="order_month")) + 1

    st.markdown("")
    go_predict = st.button("Score this order", type="primary", use_container_width=True)

    if not go_predict:
        st.markdown(ui.empty_state(
            "🎯", "Ready when you are",
            "Set the order above and hit <b>Score this order</b>. Try River Sand from "
            "Rajasthan to Bihar in July, then switch the month to February — the same "
            "order goes from critical to routine."), unsafe_allow_html=True)
    else:
        with safe_panel("Order scoring"):
            from suppliers_db import state_distance_km

            distance = state_distance_km(origin, state)
            reliability = 1 - past_del
            fake_supplier = {"state": origin, "tier": sup_tier,
                             "reliability_score": reliability,
                             "avg_lead_days": 14, "name": "Selected supplier"}

            from demo_data import _heuristic_delay, build_delay_input
            payload = build_delay_input(material, fake_supplier, state, order_month,
                                        quantity, distance_km=distance,
                                        rng=np.random.default_rng(7))

            with st.spinner("Scoring against 38 features…"):
                if MODELS_READY:
                    from train_delay_model import predict_delay
                    result = predict_delay(
                        MODELS["clf_delay"], MODELS["reg_delay"], MODELS["conformal"],
                        MODELS["enc_delay"], MODELS["feat_delay"], payload,
                        explainer=MODELS.get("explainer"))
                else:
                    result = _heuristic_delay(payload)

                wastage = cached_wastage({material: quantity},
                                         dict(PROJECT_CTX, month=order_month),
                                         MODELS_READY)

            label = result["risk_label"]
            color = ui.RISK_COLORS[label]

            left, right = st.columns([3, 2])
            with left:
                m1, m2, m3 = st.columns(3)
                m1.markdown(ui.kpi("Delay probability",
                                   f"{result['delay_probability']:.0%}",
                                   "chance it misses the promised date", color),
                            unsafe_allow_html=True)
                m2.markdown(ui.kpi("If it is late",
                                   f"{result['conditional_delay_days']:.0f} d",
                                   "90% of similar orders: " + ui.format_interval(
                                       result["conditional_ci_lower"],
                                       result["conditional_ci_upper"])),
                            unsafe_allow_html=True)
                m3.markdown(ui.kpi("Risk band", label,
                                   f"score {result['risk_score']}/100", color),
                            unsafe_allow_html=True)
                st.markdown("")

                factors = "".join(f"<li style='margin-bottom:4px;'>{f}</li>"
                                  for f in result["top_risk_factors"])
                source = ("SHAP — the model's own top contributors for this order"
                          if result.get("explanation_source") == "shap"
                          else "rule-based fallback")
                st.markdown(
                    f'<div class="panel" style="border-left:3px solid {color};">'
                    f'<p class="panel-title" style="color:{color};">'
                    f'{ui.RISK_ICON[label]} {label} risk — '
                    f'{result["delay_probability"]:.0%} chance this order slips</p>'
                    f'<p class="panel-note" style="margin:9px 0 5px;">'
                    f'Why the model says so:</p>'
                    f'<ul style="margin:0;font-size:12.5px;color:var(--text);'
                    f'padding-left:18px;">{factors}</ul>'
                    f'<p class="panel-note" style="margin-top:10px;font-size:11px;'
                    f'color:var(--dim);">Source: {source} · route {distance:,} km · '
                    f'{MONTHS[order_month - 1]}</p></div>',
                    unsafe_allow_html=True)

            with right:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result["risk_score"],
                    # Inset the domain so the 0 and 100 axis labels are not
                    # clipped by the plot edge.
                    domain={"x": [0.08, 0.92], "y": [0, 0.92]},
                    number={"suffix": "<span style='font-size:15px;'>/100</span>",
                            "font": {"color": color, "size": 38}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#64738c",
                                 "tickfont": {"size": 10}},
                        "bar": {"color": color, "thickness": 0.72},
                        "bgcolor": "rgba(255,255,255,0.03)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 30], "color": "rgba(61,220,132,0.14)"},
                            {"range": [30, 55], "color": "rgba(242,194,0,0.14)"},
                            {"range": [55, 75], "color": "rgba(255,140,26,0.14)"},
                            {"range": [75, 100], "color": "rgba(255,77,79,0.16)"},
                        ],
                    }))
                ui.apply_plot_theme(fig, height=260, title="Delay risk score")
                fig.update_layout(margin=dict(l=24, r=24, t=48, b=8))
                st.plotly_chart(fig, use_container_width=True)

            if not wastage.empty:
                w = wastage.iloc[0]
                st.markdown("#### How much should you actually order?")
                st.markdown(ui.why(
                    "The blueprint quantity is never the order quantity. The wastage model "
                    "sizes the buffer from <b>this site's</b> workforce skill, supervision "
                    "quality and monsoon exposure — the three levers a site manager "
                    "actually controls."), unsafe_allow_html=True)
                waste_color = {"High": ui.RISK_COLORS["Critical"],
                               "Medium": ui.RISK_COLORS["Medium"],
                               "Low": ui.RISK_COLORS["Low"]}[w["category"]]
                w1, w2, w3 = st.columns(3)
                w1.markdown(ui.kpi("Predicted wastage", f"{w['wastage_pct']:.1f}%",
                                   f"10th–90th percentile: "
                                   f"{w['wastage_low']:.1f}–{w['wastage_high']:.1f}%",
                                   waste_color), unsafe_allow_html=True)
                w2.markdown(ui.kpi("Order this much", f"{w['order_qty']:,.0f}",
                                   f"blueprint says {w['blueprint_qty']:,}"),
                            unsafe_allow_html=True)
                w3.markdown(ui.kpi("Cost of the waste", fmt_inr(w["cost_overrun"]),
                                   f"{w['category']} wastage band", waste_color),
                            unsafe_allow_html=True)
                st.markdown("")
                st.caption("Drivers: " + " · ".join(w["drivers"]))


# ══════════════════════════════════════════════════════════════
# 3 · WASTAGE & COST
# ══════════════════════════════════════════════════════════════

with tab_waste:
    st.markdown(ui.section(
        "Step 3", "Wastage &amp; cost overrun",
        "", "Wastage &amp; cost overrun",
        f"Per-material wastage for <b>{project_name}</b>, scored from this site's actual "
        f"conditions: {workforce_skill.lower()} workforce, {supervision.lower()} "
        f"supervision, {contractor_exp} years of contractor experience."),
        unsafe_allow_html=True)

    default_boq = {"OPC Cement": 800, "River Sand": 450, "TMT Steel": 60,
                   "Fly Ash Bricks": 28000, "Vitrified Tiles": 1400, "Plywood": 700}
    chosen = st.multiselect("Materials in the Bill of Quantities", MATERIALS,
                            default=list(default_boq.keys()))

    if not chosen:
        st.markdown(ui.empty_state(
            "📦", "No materials selected",
            "Pick at least one material above to forecast its wastage and cost overrun."),
            unsafe_allow_html=True)
    else:
        boq = {}
        qty_cols = st.columns(min(len(chosen), 6))
        for i, mat in enumerate(chosen):
            with qty_cols[i % len(qty_cols)]:
                boq[mat] = st.number_input(mat, 1, 200000,
                                           default_boq.get(mat, 500),
                                           key=f"boq_{mat}",
                                           help=f"₹{MATERIAL_PRICES.get(mat, 500):,}/unit")
        # Remembered so the exported report covers the BoQ the user set up here.
        st.session_state["last_boq"] = dict(boq)

        with safe_panel("The wastage forecast"):
            with st.spinner("Scoring each material…"):
                wf = cached_wastage(boq, PROJECT_CTX, MODELS_READY)

            total_overrun = float(wf["cost_overrun"].sum())
            weighted = float(
                (wf["wastage_pct"] * wf["blueprint_qty"]).sum() / wf["blueprint_qty"].sum())

            k1, k2, k3 = st.columns(3)
            k1.markdown(ui.kpi("Projected overrun", fmt_inr(total_overrun),
                               "Value of material you will throw away",
                               ui.RISK_COLORS["Critical"]), unsafe_allow_html=True)
            # Colour against the industry benchmark, not an arbitrary cutoff —
            # showing 18.7% in warning amber next to "benchmark: 20-30%" tells
            # the reader two contradictory things at once.
            if weighted >= 20:
                weighted_color, weighted_note = (
                    ui.RISK_COLORS["Critical"], "inside the 20–30% industry norm — bad news")
            elif weighted >= 12:
                weighted_color, weighted_note = (
                    ui.RISK_COLORS["Medium"], "below the 20–30% industry norm")
            else:
                weighted_color, weighted_note = (
                    ui.RISK_COLORS["Low"], "well below the 20–30% industry norm")
            k2.markdown(ui.kpi("Weighted wastage", f"{weighted:.1f}%",
                               weighted_note, weighted_color), unsafe_allow_html=True)
            k3.markdown(ui.kpi("Materials in High band", f"{(wf['category'] == 'High').sum()}",
                               f"of {len(wf)} in the BoQ", ui.RISK_COLORS["Medium"]),
                        unsafe_allow_html=True)

            st.markdown("")
            left, right = st.columns([5, 4])
            with left:
                table = wf.copy()
                table["Wastage"] = table["wastage_pct"].map(lambda v: f"{v:.1f}%")
                table["10–90% range"] = table.apply(
                    lambda r: f"{r['wastage_low']:.0f}–{r['wastage_high']:.0f}%", axis=1)
                table["Order qty"] = table["order_qty"].map(lambda v: f"{v:,.0f}")
                table["Blueprint"] = table["blueprint_qty"].map(lambda v: f"{v:,.0f}")
                table["Cost of waste"] = table["cost_overrun"].map(fmt_inr_full)
                st.dataframe(
                    table[["material", "Blueprint", "Order qty", "Wastage",
                           "10–90% range", "category", "Cost of waste"]].rename(columns={
                               "material": "Material", "category": "Band"}),
                    use_container_width=True, hide_index=True, height=250)
                st.caption("Every wastage figure carries its 10th–90th percentile range from "
                           "the quantile models — a point estimate alone would be misleading.")

                # Percentages and rupees rank materials differently, and rupees
                # are what the site actually loses. Surfacing the gap turns a
                # table into an insight: the worst % is rarely the worst cost.
                by_cost = wf.sort_values("cost_overrun", ascending=False).iloc[0]
                by_pct = wf.sort_values("wastage_pct", ascending=False).iloc[0]
                if by_cost["material"] != by_pct["material"]:
                    st.info(
                        f"**Chase {by_cost['material']}, not {by_pct['material']}.** "
                        f"{by_pct['material']} has the worst wastage rate "
                        f"({by_pct['wastage_pct']:.1f}%), but {by_cost['material']} loses more "
                        f"money — {fmt_inr_full(by_cost['cost_overrun'])} at only "
                        f"{by_cost['wastage_pct']:.1f}% wastage, because it is the expensive "
                        f"one. Percentages rank materials differently than rupees do.",
                        icon="💡")

            with right:
                fig = go.Figure(go.Bar(
                    x=wf["material"], y=wf["wastage_pct"],
                    marker=dict(color=[ui.RISK_COLORS["Critical"] if c == "High"
                                       else ui.RISK_COLORS["Medium"] if c == "Medium"
                                       else ui.RISK_COLORS["Low"] for c in wf["category"]],
                                line=dict(width=0)),
                    error_y=dict(type="data", symmetric=False,
                                 array=(wf["wastage_high"] - wf["wastage_pct"]).clip(lower=0),
                                 arrayminus=(wf["wastage_pct"] - wf["wastage_low"]).clip(lower=0),
                                 color="rgba(255,255,255,0.4)", thickness=1.2, width=4),
                    text=[f"{v:.1f}%" for v in wf["wastage_pct"]],
                    textposition="outside", textfont=dict(size=11),
                    hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
                ))
                # Anchor the threshold labels outside the plotting area — the
                # default right-hand placement lands them on top of the last bar.
                # Short labels only — anything longer gets clipped at the axis.
                # The colours carry the low/high meaning; the caption spells it out.
                fig.add_hline(y=5, line_dash="dot", line_color=ui.RISK_COLORS["Low"],
                              annotation_text="5%", annotation_position="top left",
                              annotation_font_size=10,
                              annotation_font_color=ui.RISK_COLORS["Low"])
                fig.add_hline(y=15, line_dash="dot", line_color=ui.RISK_COLORS["Critical"],
                              annotation_text="15%", annotation_position="top left",
                              annotation_font_size=10,
                              annotation_font_color=ui.RISK_COLORS["Critical"])
                ui.apply_plot_theme(
                    fig, height=360,
                    title="Wastage by material, with uncertainty<br>"
                          "<sub style='font-size:10px;color:#8d9bb5;'>"
                          "dotted lines: 5% low band, 15% high band</sub>")
                headroom = float((wf["wastage_high"].max() if not wf.empty else 20) * 1.18)
                fig.update_layout(xaxis_tickangle=-25,
                                  # Left margin has to clear the threshold labels,
                                  # which anchor outside the plotting area.
                                  margin=dict(l=64, r=10, t=48, b=70),
                                  yaxis=dict(title="Wastage (%)", range=[0, headroom],
                                             gridcolor="rgba(255,255,255,0.06)"))
                st.plotly_chart(fig, use_container_width=True)

            # ── Scenario ladder: re-run the model, do not hardcode ──
            st.markdown("#### What would better supervision be worth?")
            st.markdown(ui.why(
                "Each row below is a <b>fresh model run</b> on the same Bill of Quantities "
                "with only the supervision level changed. This is the business case for "
                "hiring one more site engineer, priced by the model."), unsafe_allow_html=True)

            with st.spinner("Re-scoring under each supervision level…"):
                ladder = []
                for level in ["Poor", "Average", "Good", "Excellent"]:
                    scen = cached_wastage(boq, dict(PROJECT_CTX, supervision=level),
                                          MODELS_READY)
                    if scen.empty:
                        continue
                    ladder.append({
                        "Supervision": level + (" (current)" if level == supervision else ""),
                        "_level": level,
                        "Weighted wastage %": round(float(
                            (scen["wastage_pct"] * scen["blueprint_qty"]).sum()
                            / scen["blueprint_qty"].sum()), 1),
                        "Cost of waste": float(scen["cost_overrun"].sum()),
                    })

            if ladder:
                ld = pd.DataFrame(ladder)
                current_cost = float(
                    ld.loc[ld["_level"] == supervision, "Cost of waste"].iloc[0])
                best_cost = float(ld["Cost of waste"].min())
                saving = current_cost - best_cost

                lc1, lc2 = st.columns([3, 4])
                with lc1:
                    show = ld[["Supervision", "Weighted wastage %", "Cost of waste"]].copy()
                    show["Cost of waste"] = show["Cost of waste"].map(fmt_inr_full)
                    st.dataframe(show, use_container_width=True, hide_index=True)
                    if saving > 0:
                        st.success(
                            f"Moving from **{supervision}** to **Excellent** supervision "
                            f"saves **{fmt_inr(saving)}** of wasted material on this BoQ "
                            f"alone.", icon="💡")
                    else:
                        st.info("This site is already at the best modelled supervision "
                                "level for wastage.", icon="✅")
                with lc2:
                    fig = go.Figure(go.Bar(
                        x=ld["_level"], y=ld["Cost of waste"],
                        marker=dict(color=[ui.RISK_COLORS["Critical"] if lv == supervision
                                           else "#2f3d55" for lv in ld["_level"]],
                                    line=dict(width=0)),
                        text=[fmt_inr(v) for v in ld["Cost of waste"]],
                        textposition="outside", textfont=dict(size=11),
                        hovertemplate="%{x}: %{text}<extra></extra>"))
                    ui.apply_plot_theme(fig, height=300,
                                        title="Cost of wasted material by supervision level")
                    fig.update_layout(yaxis=dict(title="₹",
                                                 gridcolor="rgba(255,255,255,0.06)"))
                    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# 4 · PROJECT SIMULATOR
# ══════════════════════════════════════════════════════════════

with tab_sim:
    st.markdown(ui.section(
        "", "Project simulator — the whole schedule, 10,000 times",
        "Single-order risk is not the real question. The real question is what one late "
        "delivery does to <b>everything downstream</b>. This runs the full activity "
        "network as a Monte Carlo."), unsafe_allow_html=True)

    st.markdown(ui.why(
        "<b>Why this is not just 10,000 random numbers:</b> we score the delay model once "
        "per material per calendar month (about 130 calls) to build a calibrated risk "
        "profile, then draw 10,000 scenarios from those profiles through the real activity "
        "dependency graph. Scoring inside the loop would be ~300,000 model calls and take "
        "minutes; this way the ML sets every probability and magnitude, and the whole thing "
        "finishes in under a second."), unsafe_allow_html=True)

    n_sims = st.select_slider("Scenarios to simulate", [1000, 5000, 10000, 20000],
                              value=10000)
    run_sim = st.button("Run simulation", type="primary", use_container_width=True,
                        key="run_sim")

    if not run_sim:
        st.markdown(ui.empty_state(
            "🎲", f"Simulate {project_type} in {state}, starting {month_name}",
            "You will get an on-time probability, the full duration distribution, which "
            "activities are the real bottlenecks, and which material delays cascade "
            "furthest downstream."), unsafe_allow_html=True)
    else:
        with safe_panel("The project simulator"):
            with st.spinner(f"Running {n_sims:,} project scenarios…"):
                sim = cached_simulation(project_type, state, current_month,
                                        n_sims, MODELS_READY)

            tl = sim["project_timeline"]
            summary = sim["executive_summary"]
            cfg = sim["simulation_config"]
            on_time = tl["on_time_probability_pct"]
            risk_color = (ui.RISK_COLORS["Low"] if on_time >= 70 else
                          ui.RISK_COLORS["Medium"] if on_time >= 50 else
                          ui.RISK_COLORS["High"] if on_time >= 30 else
                          ui.RISK_COLORS["Critical"])

            tiles = [
                ("On-time probability", f"{on_time:.0f}%",
                 f"within 10% of the {tl['baseline_duration_days']}-day plan", risk_color),
                ("Most likely duration", f"{tl['most_likely_days']:.0f} d",
                 f"{tl['delay_over_baseline_days']:+.0f} d vs plan", "#eef2f9"),
                ("Worst case (P90)", f"{tl['worst_case_days']:.0f} d",
                 "1 run in 10 is at least this bad", ui.RISK_COLORS["High"]),
                ("Overall risk", summary["risk_level"],
                 f"{cfg['n_simulations']:,} scenarios in {cfg['elapsed_seconds']}s",
                 risk_color),
            ]
            for col, (label, value, sub, color) in zip(st.columns(4), tiles):
                col.markdown(ui.kpi(label, value, sub, color), unsafe_allow_html=True)

            st.markdown(
                f'<div class="panel" style="border-left:3px solid {risk_color};'
                f'margin-top:12px;"><p class="panel-title">Executive summary</p>'
                f'<p class="panel-note" style="font-size:13.5px;color:var(--text);'
                f'margin-top:6px;">{summary["headline"]}</p></div>',
                unsafe_allow_html=True)

            source_note = ("delay probabilities from the trained XGBoost model"
                           if cfg.get("risk_source") == "ml"
                           else "a transparent physics-based fallback (models not trained)")
            st.caption(
                f"{cfg['n_simulations']:,} scenarios · {cfg['n_model_calls']} model calls · "
                f"{source_note} · {cfg['schedule_absorption']:.0%} of each material delay "
                f"assumed absorbed by schedule float.")

            # Duration distribution
            dist = sim["duration_distribution"]
            centers = [(dist["bin_edges"][i] + dist["bin_edges"][i + 1]) / 2
                       for i in range(len(dist["counts"]))]
            fig = go.Figure(go.Bar(
                x=centers, y=dist["counts"],
                marker=dict(color="rgba(76,194,255,0.55)", line=dict(width=0)),
                hovertemplate="%{x:.0f} days: %{y} runs<extra></extra>"))
            fig.add_vline(x=tl["baseline_duration_days"], line_dash="dash",
                          line_color=ui.RISK_COLORS["Low"],
                          annotation_text="Contract plan", annotation_font_size=11)
            fig.add_vline(x=tl["most_likely_days"], line_dash="dash",
                          line_color=ui.RISK_COLORS["High"],
                          annotation_text="Most likely", annotation_font_size=11)
            fig.add_vline(x=tl["worst_case_days"], line_dash="dot",
                          line_color=ui.RISK_COLORS["Critical"],
                          annotation_text="P90", annotation_font_size=11)
            ui.apply_plot_theme(fig, height=340,
                                title="Where this project actually finishes")
            fig.update_layout(
                xaxis=dict(title="Total project duration (days)",
                           gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Scenarios", gridcolor="rgba(255,255,255,0.06)"),
                bargap=0.04)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### Which activities actually slip")
                acts = pd.DataFrame(sim["activity_risk_ranking"][:8])
                if not acts.empty:
                    st.dataframe(
                        acts[["activity", "pct_simulations_delayed",
                              "avg_delay_when_delayed", "max_delay_observed"]].rename(
                            columns={"activity": "Activity",
                                     "pct_simulations_delayed": "Slips in %",
                                     "avg_delay_when_delayed": "Avg slip (d)",
                                     "max_delay_observed": "Worst (d)"}),
                        use_container_width=True, hide_index=True, height=310,
                        column_config={"Slips in %": st.column_config.ProgressColumn(
                            "Slips in %", format="%.0f%%", min_value=0, max_value=100)})

                if sim.get("single_points_of_failure"):
                    st.markdown("##### ⚠️ Single points of failure")
                    for s in sim["single_points_of_failure"][:4]:
                        st.markdown(
                            f"- **{s['activity']}** depends on *{s['single_material']}* "
                            f"alone — late in {s['delay_frequency']:.0f}% of runs. "
                            f"{s['recommendation']}.")

            with c2:
                st.markdown("##### Which materials cause it")
                mats = pd.DataFrame(sim["material_risk_ranking"])
                if not mats.empty:
                    fig = go.Figure(go.Bar(
                        x=mats["delay_frequency"], y=mats["material"], orientation="h",
                        marker=dict(color=[ui.RISK_COLORS["Critical"] if c
                                           else ui.RISK_COLORS["Medium"]
                                           for c in mats["is_critical"]],
                                    line=dict(width=0)),
                        text=[f"{v:.0f}%" for v in mats["delay_frequency"]],
                        textposition="outside", textfont=dict(size=11),
                        customdata=mats[["deliveries_per_project", "project_impact_pct"]],
                        hovertemplate="%{y}<br>%{x:.0f}% of its deliveries land late"
                                      "<br>%{customdata[0]:.0f} deliveries per project"
                                      "<br>slips at least once in %{customdata[1]:.0f}% "
                                      "of projects<extra></extra>"))
                    ui.apply_plot_theme(fig, height=330,
                                        title="Share of each material's deliveries that land late")
                    fig.update_layout(
                        xaxis=dict(title="% of deliveries late", range=[0, 108],
                                   gridcolor="rgba(255,255,255,0.06)"),
                        yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Per delivery, not per project — a material used by four "
                               "activities would otherwise saturate at 100%.")

                if sim.get("cascade_risk"):
                    st.markdown("##### Cascade effects")
                    for cr in sim["cascade_risk"][:3]:
                        st.markdown(
                            f"- **{cr['material']}** — {cr['cascade_events']:,} cascade "
                            f"events, averaging {cr['avg_cascade_delay']:.0f} days, "
                            f"hitting {len(cr['activities_affected'])} downstream activities.")

            st.markdown("##### What to do about it")
            for i, action in enumerate(summary["recommended_actions"], 1):
                st.markdown(f"**{i}.** {action}")


# ══════════════════════════════════════════════════════════════
# 5 · PLAN & REPORT
# ══════════════════════════════════════════════════════════════

with tab_plan:
    st.markdown(ui.section(
        "", "Procurement plan &amp; site report",
        "Turn everything above into an order schedule your buyer can act on today, "
        "then export it."), unsafe_allow_html=True)

    plan_materials = st.multiselect(
        "Materials to plan", MATERIALS,
        default=["TMT Steel", "OPC Cement", "River Sand", "Fly Ash Bricks"],
        key="plan_mats")

    if not plan_materials:
        st.markdown(ui.empty_state(
            "📋", "Nothing to plan yet",
            "Select the materials in your Bill of Quantities to generate an order "
            "schedule with wastage buffers and supplier allocations."),
            unsafe_allow_html=True)
    else:
        plan_boq = {}
        cols = st.columns(min(len(plan_materials), 5))
        for i, mat in enumerate(plan_materials):
            with cols[i % len(cols)]:
                plan_boq[mat] = st.number_input(mat, 1, 200000, 400, key=f"plan_{mat}")

        if st.button("Generate procurement plan", type="primary",
                     use_container_width=True, key="gen_plan"):
            with safe_panel("The procurement planner"):
                with st.spinner("Scoring delay risk, sizing buffers and optimising "
                                "supplier allocation…"):
                    from simulation_engine import optimize_procurement
                    from suppliers_db import find_suppliers, state_distance_km

                    wf = cached_wastage(plan_boq, PROJECT_CTX, MODELS_READY)
                    wmap = wf.set_index("material").to_dict("index") if not wf.empty else {}

                    rows = []
                    for mat, qty in plan_boq.items():
                        sups = find_suppliers(mat, state)
                        supplier = sups[0] if sups else {
                            "name": "No supplier on file", "state": state,
                            "tier": "Tier 2 (Regional Distributor)",
                            "reliability_score": 0.7, "avg_lead_days": 14}
                        distance = state_distance_km(supplier["state"], state)

                        from demo_data import _heuristic_delay, build_delay_input
                        payload = build_delay_input(
                            mat, supplier, state, current_month, qty,
                            distance_km=distance, rng=np.random.default_rng(11))
                        if MODELS_READY:
                            from train_delay_model import predict_delay
                            res = predict_delay(
                                MODELS["clf_delay"], MODELS["reg_delay"],
                                MODELS["conformal"], MODELS["enc_delay"],
                                MODELS["feat_delay"], payload,
                                explainer=MODELS.get("explainer"))
                        else:
                            res = _heuristic_delay(payload)

                        # Lead-time buffer sized from the model, not a fixed rule.
                        buffer_days = res["delay_probability"] * res["conditional_delay_days"]
                        order_by = supplier["avg_lead_days"] + buffer_days
                        w = wmap.get(mat, {})

                        rows.append({
                            "Material": mat,
                            "Order this many days out": round(order_by),
                            "Risk": res["risk_label"],
                            "Delay risk": res["delay_probability"] * 100,
                            "Blueprint": qty,
                            "Order qty (incl. wastage)": round(w.get("order_qty", qty)),
                            "Wastage": f"{w.get('wastage_pct', 0):.1f}%",
                            "Supplier": supplier["name"],
                            "Lead": f"{supplier['avg_lead_days']} d",
                            "Buffer": f"+{buffer_days:.0f} d",
                            "Why": res["top_risk_factors"][0]
                            if res["top_risk_factors"] else "—",
                            "_cost": w.get("cost_overrun", 0),
                        })

                plan = pd.DataFrame(rows).sort_values(
                    "Order this many days out", ascending=False)

                st.markdown("#### Order schedule")
                st.markdown(ui.why(
                    "The lead-time buffer is <b>risk-weighted</b>: probability of delay × "
                    "expected days late. A 90%-risk cement order gets a real buffer; a "
                    "5%-risk tile order gets almost none. A flat '2 weeks early for "
                    "everything' rule ties up working capital for no reason."),
                    unsafe_allow_html=True)

                st.dataframe(
                    plan.drop(columns=["_cost"]),
                    use_container_width=True, hide_index=True,
                    column_config={
                        "Delay risk": st.column_config.ProgressColumn(
                            "Delay risk", format="%.0f%%", min_value=0, max_value=100),
                        "Order this many days out": st.column_config.NumberColumn(
                            "Order this many days out", format="%d d"),
                    })

                total_waste = float(plan["_cost"].sum())
                if total_waste:
                    st.warning(
                        f"Wastage buffer across this BoQ costs **{fmt_inr(total_waste)}** — "
                        "already built into the order quantities above, so you buy it once "
                        "instead of discovering it mid-pour.", icon="💸")

                # ── Supplier allocation & Pareto ──
                st.markdown("#### Supplier allocation — cost versus stock-out risk")
                opt = optimize_procurement(
                    [{"material_type": m, "quantity": q} for m, q in plan_boq.items()],
                    state, current_month)

                st.markdown(ui.why(
                    "Two suppliers on the same corridor are <b>not independent</b> — the "
                    f"same monsoon hits both. We model that common-mode correlation at "
                    f"<b>{opt['common_mode_correlation']:.0%}</b> for {month_name}, which is "
                    "why splitting an order removes some risk but never all of it. A model "
                    "that assumed independence would claim a ~97% reduction here, and be "
                    "wrong."), unsafe_allow_html=True)

                alloc = pd.DataFrame(opt["procurement_strategy"])
                if not alloc.empty:
                    ac1, ac2 = st.columns([5, 4])
                    with ac1:
                        show = alloc[["material", "strategy", "primary_supplier",
                                      "primary_allocation_pct", "backup_supplier",
                                      "backup_allocation_pct",
                                      "estimated_delay_reduction_pct",
                                      "cost_premium_pct"]].rename(columns={
                                          "material": "Material", "strategy": "Strategy",
                                          "primary_supplier": "Primary",
                                          "primary_allocation_pct": "Primary %",
                                          "backup_supplier": "Backup",
                                          "backup_allocation_pct": "Backup %",
                                          "estimated_delay_reduction_pct": "Stock-out risk ↓",
                                          "cost_premium_pct": "Cost premium"})
                        st.dataframe(show, use_container_width=True, hide_index=True,
                                     column_config={
                                         "Stock-out risk ↓": st.column_config.NumberColumn(
                                             "Stock-out risk ↓", format="%.0f%%"),
                                         "Cost premium": st.column_config.NumberColumn(
                                             "Cost premium", format="%.1f%%")})
                        st.info(opt["recommendation"], icon="🎯")

                    with ac2:
                        pf = pd.DataFrame(opt.get("pareto_frontier", []))
                        if not pf.empty:
                            fig = go.Figure()
                            off = pf[~pf["on_frontier"]]
                            on = pf[pf["on_frontier"]]
                            if not off.empty:
                                fig.add_trace(go.Scatter(
                                    x=off["extra_spend_pct"], y=off["risk_reduction_pct"],
                                    mode="markers", name="Dominated",
                                    marker=dict(size=9, color="#3a475f",
                                                line=dict(width=0)),
                                    hovertemplate="%{text}<extra></extra>",
                                    text=off["label"]))
                            fig.add_trace(go.Scatter(
                                x=on["extra_spend_pct"], y=on["risk_reduction_pct"],
                                mode="lines+markers+text", name="Pareto frontier",
                                line=dict(color=ui.RISK_COLORS["Medium"], width=2),
                                marker=dict(size=12, color=ui.RISK_COLORS["Medium"],
                                            line=dict(width=0)),
                                text=on["label"], textposition="top center",
                                textfont=dict(size=10),
                                hovertemplate="%{text}<br>+%{x:.1f}% spend → "
                                              "%{y:.0f}% less risk<extra></extra>"))
                            ui.apply_plot_theme(fig, height=330,
                                                title="Every allocation we evaluated")
                            fig.update_layout(
                                xaxis=dict(title="Extra spend (%)",
                                           gridcolor="rgba(255,255,255,0.06)"),
                                yaxis=dict(title="Stock-out risk removed (%)",
                                           gridcolor="rgba(255,255,255,0.06)"),
                                showlegend=True,
                                legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Grey points are dominated — something else is both "
                                       "cheaper and safer. Only the amber line is worth "
                                       "considering.")

    st.markdown("---")
    st.markdown("#### 📄 Export the site report")
    st.caption("A single-page brief the site engineer can carry into the morning meeting: "
               "risk summary, material forecast and the action list.")

    if st.button("Generate report", use_container_width=True, key="gen_report"):
        with safe_panel("The report generator"):
            with st.spinner("Building the report…"):
                import uuid
                from report_generator import generate_pdf_report

                rid = f"RPT-{uuid.uuid4().hex[:8].upper()}"
                path = generate_pdf_report(
                    report_id=rid, project_name=project_name, project_type=project_type,
                    state=state, current_month=month_name,
                    output_dir="reports/generated",
                    models=MODELS if MODELS_READY else None,
                    # Report the project the user actually configured, not defaults.
                    boq=st.session_state.get("last_boq"),
                    site_conditions=PROJECT_CTX)

            is_pdf = str(path).lower().endswith(".pdf")
            st.success(f"Report **{rid}** ready.", icon="✅")
            with open(path, "rb") as fh:
                st.download_button(
                    label=f"⬇️  Download {'PDF' if is_pdf else 'HTML'} report",
                    data=fh.read(),
                    file_name=f"NirmanAI_{rid}.{'pdf' if is_pdf else 'html'}",
                    mime="application/pdf" if is_pdf else "text/html",
                    use_container_width=True)
            if not is_pdf:
                st.caption("WeasyPrint is not available in this environment, so the report "
                           "was exported as HTML. It prints to PDF from any browser.")


# ══════════════════════════════════════════════════════════════
# ASK JARVIS
# ══════════════════════════════════════════════════════════════

with tab_agent:
    st.markdown(ui.section(
        "", "Ask Jarvis",
        "A procurement assistant wired to the same engines as the rest of this "
        "dashboard: the simulator, the optimiser, the supplier database and the "
        "report generator."), unsafe_allow_html=True)

    QUICK = {
        "🎲 Simulate my project": "Simulate my project timeline risk",
        "🏭 Find alternate suppliers": f"Find alternate OPC Cement suppliers for {state}",
        "🎯 Optimise my procurement": "Generate an optimised procurement strategy for TMT Steel",
        "📄 Generate a report": "Generate a risk report for this project",
    }

    st.caption("Try one of these, or type your own question below.")
    pending = None
    for col, (label, prompt_text) in zip(st.columns(4), QUICK.items()):
        if col.button(label, use_container_width=True, key=f"quick_{label}"):
            pending = prompt_text

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool"):
                st.caption(f"🔧 Called: `{msg['tool']}`")

    typed = st.chat_input("Ask about delays, suppliers, simulations or reports…")
    question = typed or pending

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with safe_panel("Jarvis"):
                with st.spinner("Working…"):
                    from agent import process_agent_message
                    res = process_agent_message(
                        question, "nirmanai-session",
                        context={"project_type": project_type, "state": state,
                                 "month": current_month, "project_name": project_name})
                st.markdown(res["message"])
                if res["tool_used"]:
                    st.caption(f"🔧 Called: `{res['tool_used']}`")
                st.session_state.messages.append({
                    "role": "assistant", "content": res["message"],
                    "tool": res["tool_used"]})

    if not st.session_state.messages:
        st.markdown(ui.empty_state(
            "🤖", "Jarvis is connected to the live engines",
            "It does not just talk — it runs the Monte Carlo simulator, queries the "
            "71-supplier database and calls the procurement optimiser, then reports what "
            "they returned. Without a Gemini key it still works, using direct tool "
            "routing."), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown(
    f"""<div class="footer-note">
      <b style="color:#8d9bb5;">NirmanAI</b> · Team Aim-Nexus, IIT Madras ·
      KAYA × IIT India Hackathon 2026<br>
      Every prediction carries a calibrated interval and the factors behind it.
    </div>""",
    unsafe_allow_html=True,
)
