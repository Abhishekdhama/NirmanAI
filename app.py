"""
NirmanAI — Streamlit Dashboard
==================================
Full-featured real-time dashboard for construction supply chain intelligence.
Runs locally: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import sys
from datetime import datetime, timedelta
import random

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NirmanAI | AI Supply Chain Intelligence",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background: #0f1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16213e 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    
    .risk-critical { color: #ff4444; font-weight: 700; }
    .risk-high     { color: #ff8800; font-weight: 700; }
    .risk-medium   { color: #ffcc00; font-weight: 600; }
    .risk-low      { color: #44ff88; font-weight: 600; }
    
    .alert-box {
        background: rgba(255,68,68,0.1);
        border-left: 4px solid #ff4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .warning-box {
        background: rgba(255,136,0,0.1);
        border-left: 4px solid #ff8800;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .ok-box {
        background: rgba(68,255,136,0.08);
        border-left: 4px solid #44ff88;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    
    h1, h2, h3 { color: #e8eaf6; }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 15px; font-weight: 500; padding: 8px 20px;
    }
    
    div[data-testid="metric-container"] {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all trained models once per process. Returns a dict, or None if unavailable."""
    try:
        return {
            "clf_delay":   joblib.load("models/delay_classifier.pkl"),
            "reg_delay":   joblib.load("models/delay_regressor.pkl"),
            "q_hat":       joblib.load("models/delay_q_hat.pkl"),
            "enc_delay":   joblib.load("models/delay_encoders.pkl"),
            "feat_delay":  joblib.load("models/delay_features.pkl"),
            "reg_wast":    joblib.load("models/wastage_regressor.pkl"),
            "reg_wast_lo": joblib.load("models/wastage_regressor_lo.pkl"),
            "reg_wast_hi": joblib.load("models/wastage_regressor_hi.pkl"),
            "enc_wast":    joblib.load("models/wastage_encoders.pkl"),
            "feat_wast":   joblib.load("models/wastage_features.pkl"),
        }
    except Exception:
        return None

def monsoon_intensity(month):
    profile = {1:0.0,2:0.0,3:0.0,4:0.05,5:0.15,
               6:0.7,7:0.9,8:0.85,9:0.6,10:0.2,11:0.05,12:0.0}
    return profile.get(month, 0.0)

# Mock data for when models aren't loaded yet
def get_mock_deliveries():
    materials = ["TMT Steel","OPC Cement","River Sand","Structural Steel",
                 "Fly Ash Bricks","HDPE Pipes","Electrical Cable","Vitrified Tiles"]
    suppliers = ["JSW Steel - Bellary","ACC Cement - Rajasthan",
                 "Ambuja Cement - Gujarat","Tata Steel - Jharkhand",
                 "Local Supplier - Patna","Regional Dist. - Lucknow"]
    states = ["Maharashtra","Tamil Nadu","Bihar","Uttar Pradesh",
              "Rajasthan","Karnataka","West Bengal","Kerala"]
    random.seed(42)
    rows = []
    for i in range(18):
        mat  = random.choice(materials)
        orig = random.choice(states)
        dest = random.choice(states)
        risk = random.choices(["Critical","High","Medium","Low"],
                              weights=[0.15,0.25,0.35,0.25])[0]
        prob = {"Critical":0.88,"High":0.67,"Medium":0.42,"Low":0.15}[risk]
        days = {"Critical":random.randint(8,18),"High":random.randint(4,9),
                "Medium":random.randint(1,5),"Low":0}[risk]
        color= {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}[risk]
        rows.append({
            "Status": color,
            "Material":   mat,
            "Supplier":   random.choice(suppliers),
            "Route":      f"{orig} → {dest}",
            "Risk":       risk,
            "Delay Prob": f"{prob:.0%}",
            "Est. Delay": f"{days} days" if days else "On time",
            "Order Value":f"₹{random.randint(2,80)*10000:,}",
        })
    return pd.DataFrame(rows)

def get_mock_wastage():
    data = [
        {"Material":"OPC Cement","Blueprint":500,"Estimated":573,"Wastage %":14.6,"Category":"Medium","Overrun":"₹27,740"},
        {"Material":"River Sand","Blueprint":300,"Estimated":378,"Wastage %":26.0,"Category":"High","Overrun":"₹1,40,400"},
        {"Material":"TMT Steel", "Blueprint":80, "Estimated":84, "Wastage %":5.0, "Category":"Low","Overrun":"₹2,48,000"},
        {"Material":"Fly Ash Bricks","Blueprint":25000,"Estimated":29250,"Wastage %":17.0,"Category":"High","Overrun":"₹34,000"},
        {"Material":"Vitrified Tiles","Blueprint":1200,"Estimated":1308,"Wastage %":9.0,"Category":"Medium","Overrun":"₹70,200"},
        {"Material":"Plywood","Blueprint":800,"Estimated":992,"Wastage %":24.0,"Category":"High","Overrun":"₹18,240"},
    ]
    return pd.DataFrame(data)

# ── HEADER ────────────────────────────────────────────────────
col_logo, col_title, col_status = st.columns([1, 5, 2])
with col_logo:
    st.markdown("## 🏗️")
with col_title:
    st.markdown("# NirmanAI")
    st.markdown("<p style='color:#8892b0;margin-top:-12px;'>AI Supply Chain Intelligence for Indian Construction</p>",
                unsafe_allow_html=True)
with col_status:
    models = load_models()
    models_ok = models is not None
    if models_ok:
        st.success("✅ Models Loaded")
    else:
        st.warning("⚠️ Demo Mode (Run setup.py first)")

st.markdown("---")

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Project Context")
    project_name = st.text_input("Project Name", "Prestige Heights — Phase 2")
    project_type = st.selectbox("Project Type", [
        "Residential Apartment","Commercial Complex","Industrial Warehouse",
        "Road & Highway","Bridge Construction","Metro Rail","Data Center","Hospital"
    ])
    state = st.selectbox("Project State", [
        "Maharashtra","Tamil Nadu","Karnataka","Gujarat","Rajasthan",
        "Uttar Pradesh","Bihar","West Bengal","Madhya Pradesh","Telangana",
        "Kerala","Punjab","Odisha","Jharkhand","Haryana"
    ])
    project_size = st.slider("Project Size (sq ft)", 1000, 50000, 8000, 500)
    current_month = st.slider("Current Month", 1, 12, datetime.now().month)

    st.markdown("---")
    st.markdown("### 🌧️ Site Conditions")
    workforce_skill = st.selectbox("Workforce Skill", ["Unskilled","Semi-skilled","Skilled","Expert"])
    supervision     = st.selectbox("Supervision Quality", ["Poor","Average","Good","Excellent"])
    contractor_exp  = st.slider("Contractor Experience (yrs)", 1, 30, 8)

    st.markdown("---")
    m_int = monsoon_intensity(current_month)
    if m_int > 0.6:
        st.markdown(f"🌧️ **Monsoon Alert**: Intensity {m_int:.0%}")
    else:
        st.markdown(f"☀️ Weather: Normal (Monsoon {m_int:.0%})")

# ── KPI ROW ───────────────────────────────────────────────────
def kpi_card(value, label, sub, value_color="#4fc3f7", sub_color="#8892b0"):
    return f"""
    <div class="metric-card">
        <h3 style="color:{value_color};margin:0;font-size:28px;">{value}</h3>
        <p style="color:#e8eaf6;margin:6px 0 0;font-size:13px;font-weight:600;">{label}</p>
        <p style="color:{sub_color};margin:2px 0 0;font-size:11px;">{sub}</p>
    </div>"""

kpis = [
    ("23",     "📦 Active Orders",      "+3 this week",        "#4fc3f7", "#44ff88"),
    ("7",      "⚠️ At-Risk Deliveries", "↑ 2 from last week",  "#ff8800", "#ff8800"),
    ("3",      "🔴 Critical Alerts",    "Action needed",       "#ff4444", "#ff4444"),
    ("₹18.4L", "💰 Projected Overrun",  "vs ₹12L baseline",    "#ffcc00", "#8892b0"),
    ("13.2%",  "📉 Avg Wastage Est.",   "Industry: 20-30%",    "#44ff88", "#8892b0"),
]
for col, (val, label, sub, vc, sc) in zip(st.columns(5), kpis):
    with col:
        st.markdown(kpi_card(val, label, sub, vc, sc), unsafe_allow_html=True)

st.markdown("---")

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📡 Live Delivery Monitor",
    "📦 Wastage Intelligence",
    "🔮 Predict New Order",
    "📊 Analytics & Insights",
    "📋 Smart Procurement Plan"
])

# ═══════════════════════════════════════════
# TAB 1: LIVE DELIVERY MONITOR
# ═══════════════════════════════════════════
with tab1:
    st.markdown("### Active Delivery Orders — Risk Dashboard")
    st.caption(f"Project: **{project_name}** | Last updated: {datetime.now().strftime('%d %b %Y, %H:%M')}")

    # Critical alerts banner
    st.markdown("""
    <div class="alert-box">
        🔴 <strong>CRITICAL:</strong> River Sand from Rajasthan is at 91% delay risk — 
        heavy monsoon on NH-48 + supplier has 52% historical delay rate. 
        <strong>Action: Contact alternate supplier immediately.</strong>
    </div>
    <div class="alert-box">
        🔴 <strong>CRITICAL:</strong> TMT Steel (Jharkhand → Bihar) — 
        84% delay risk. Truck strike reported on NH-30. Est. delay: 11 days.
    </div>
    <div class="warning-box">
        🟠 <strong>HIGH RISK:</strong> Fly Ash Bricks order overlaps with Diwali shutdown window. 
        Recommend reordering now or scheduling post-festival.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Delivery table
    df_del = get_mock_deliveries()

    def color_risk(val):
        colors = {"Critical":"background-color:#ff444430",
                  "High":"background-color:#ff880025",
                  "Medium":"background-color:#ffcc0020",
                  "Low":"background-color:#44ff8815"}
        return colors.get(val, "")

    st.dataframe(
        df_del.style.map(color_risk, subset=["Risk"]),
        use_container_width=True, height=420
    )

    # Risk distribution pie
    col_pie, col_timeline = st.columns(2)
    with col_pie:
        risk_counts = df_del["Risk"].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Delivery Risk Distribution",
            color=risk_counts.index,
            color_discrete_map={"Critical":"#ff4444","High":"#ff8800",
                                "Medium":"#ffcc00","Low":"#44ff88"},
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#e8eaf6", title_font_size=15
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_timeline:
        # Monsoon intensity timeline
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        intensities = [monsoon_intensity(m) for m in range(1,13)]
        fig_mon = go.Figure()
        fig_mon.add_trace(go.Scatter(
            x=months, y=[i*100 for i in intensities],
            fill="tozeroy", name="Monsoon Intensity",
            line=dict(color="#4fc3f7", width=2),
            fillcolor="rgba(79,195,247,0.2)"
        ))
        fig_mon.add_vline(x=current_month-1, line_dash="dash",
                          line_color="#ff8800", annotation_text="Now")
        fig_mon.update_layout(
            title="Monsoon Disruption Risk — Annual Pattern",
            xaxis_title="Month", yaxis_title="Intensity (%)",
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#e8eaf6", title_font_size=15,
            yaxis=dict(range=[0,100]), showlegend=False
        )
        st.plotly_chart(fig_mon, use_container_width=True)

    # KAYA integration preview (Pitch Deck Slide 9)
    st.markdown("---")
    st.markdown("### 🔗 KAYA AI Integration Preview")
    st.markdown("""
    <div style="background:#1a1d2e;border:1px solid #2a2d3e;border-radius:12px;padding:20px;">
        <p style="color:#4fc3f7;font-weight:600;margin:0;">KAYA Jarvis — Proactive Supplier Outreach</p>
        <p style="color:#8892b0;font-size:13px;margin:8px 0 16px;">
            Powered by NirmanAI Risk Intelligence
        </p>
        <div style="background:#0f1117;border-radius:8px;padding:14px;margin-bottom:10px;">
            🤖 <strong style="color:#e8eaf6;">Jarvis (automated)</strong>
            <span style="color:#8892b0;font-size:12px;"> · just now</span><br>
            <span style="color:#cdd6e4;font-size:13px;">
            "NirmanAI flagged your TMT Steel order (Jharkhand → Bihar) at 84% delay risk.
            I've already contacted 2 alternate suppliers and scheduled a callback for 3pm today.
            Estimated delay prevented: 11 days."
            </span>
        </div>
        <div style="background:#0f1117;border-radius:8px;padding:14px;">
            🤖 <strong style="color:#e8eaf6;">Jarvis (automated)</strong>
            <span style="color:#8892b0;font-size:12px;"> · 2 hours ago</span><br>
            <span style="color:#cdd6e4;font-size:13px;">
            "River Sand delivery from Rajasthan — NirmanAI predicts 91% chance of
            12-day delay due to monsoon on NH-48. Recommend: split order across
            Madhya Pradesh supplier. Saving ₹1.2L in idle labor cost."
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 2: WASTAGE INTELLIGENCE
# ═══════════════════════════════════════════
with tab2:
    st.markdown(f"### Material Wastage Forecast — {project_name}")
    st.caption(f"Workforce: **{workforce_skill}** | Supervision: **{supervision}** | Month: **{current_month}**")

    df_wast = get_mock_wastage()

    col_wt, col_wc = st.columns(2)

    with col_wt:
        st.markdown("#### Per-Material Wastage Estimates")

        def color_cat(val):
            return {"High":"background-color:#ff444330",
                    "Medium":"background-color:#ffcc0020",
                    "Low":"background-color:#44ff8815"}.get(val,"")

        st.dataframe(
            df_wast.style.map(color_cat, subset=["Category"]),
            use_container_width=True, height=300
        )

        total_overrun = 1_40_400 + 34_000 + 2_48_000 + 70_200 + 18_240 + 27_740
        st.error(f"💸 **Total Projected Cost Overrun: ₹{total_overrun:,.0f}**")
        st.info("💡 Improving supervision from **Poor → Good** would reduce overrun by ~35%")

    with col_wc:
        # Wastage bar chart
        fig_wast = go.Figure()
        colors = ["#ff4444" if c=="High" else "#ffcc00" if c=="Medium" else "#44ff88"
                  for c in df_wast["Category"]]
        fig_wast.add_trace(go.Bar(
            x=df_wast["Material"],
            y=df_wast["Wastage %"],
            marker_color=colors,
            text=[f"{w}%" for w in df_wast["Wastage %"]],
            textposition="outside"
        ))
        fig_wast.add_hline(y=5,  line_dash="dot", line_color="#44ff88",
                           annotation_text="Low threshold (5%)")
        fig_wast.add_hline(y=15, line_dash="dot", line_color="#ff4444",
                           annotation_text="High threshold (15%)")
        fig_wast.update_layout(
            title="Estimated Wastage % by Material",
            xaxis_title="", yaxis_title="Wastage (%)",
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#e8eaf6", title_font_size=15,
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig_wast, use_container_width=True)

    # Scenario analysis
    st.markdown("#### 🔄 Scenario Analysis — Impact of Better Supervision")
    scenario_data = {
        "Scenario": ["Current (Poor supervision)","Good supervision","Excellent supervision"],
        "Sand Wastage %": [26.0, 18.0, 12.0],
        "Cement Wastage %": [14.6, 10.0, 7.0],
        "Total Overrun ₹": [538_580, 320_000, 185_000],
    }
    st.dataframe(pd.DataFrame(scenario_data), use_container_width=True)

# ═══════════════════════════════════════════
# TAB 3: PREDICT NEW ORDER
# ═══════════════════════════════════════════
with tab3:
    st.markdown("### 🔮 Predict Delivery Risk for a New Order")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        material  = st.selectbox("Material Type", [
            "TMT Steel","OPC Cement","River Sand","Coarse Aggregate",
            "Fly Ash Bricks","AAC Blocks","Structural Steel",
            "Electrical Cable","HDPE Pipes","Vitrified Tiles","Plywood","Paint"
        ])
        sup_tier  = st.selectbox("Supplier Tier", [
            "Tier 1 (Large Manufacturer)",
            "Tier 2 (Regional Distributor)",
            "Tier 3 (Local Supplier)"
        ])
        origin    = st.selectbox("Origin State", [
            "Maharashtra","Gujarat","Rajasthan","Karnataka",
            "Tamil Nadu","Jharkhand","Odisha","Punjab"
        ])
        dest_s    = st.selectbox("Destination State", [
            "Bihar","Uttar Pradesh","West Bengal","Maharashtra",
            "Tamil Nadu","Kerala","Madhya Pradesh","Haryana"
        ])

    with col_f2:
        distance  = st.slider("Distance (km)", 50, 2000, 800)
        quantity  = st.number_input("Order Quantity", 10, 10000, 150)
        past_del  = st.slider("Supplier's Past Delay Rate", 0.05, 0.70, 0.35)
        order_month = st.slider("Order Month", 1, 12, current_month)

    if st.button("🔮 Predict Delay Risk", type="primary", use_container_width=True):
        m_int_ord = monsoon_intensity(order_month)

        wastage_result = None
        if models_ok:
            from train_delay_model import predict_delay
            from train_wastage_model import predict_wastage
            inp = {
                "month": order_month, "day_of_week": 0, "quarter": (order_month-1)//3+1,
                "is_festival_period": 0, "material_type": material,
                "supplier_tier": sup_tier, "origin_state": origin,
                "destination_state": dest_s, "distance_km": distance,
                "order_quantity": quantity,
                "promised_lead_days": 14,
                "monsoon_intensity": m_int_ord,
                "monsoon_sensitivity": 0.5,
                "dest_logistics_score": 0.6,
                "orig_logistics_score": 0.7,
                "dest_monsoon_severity": 0.65,
                "supplier_reliability": 1 - past_del * 0.8,
                "past_delay_rate": past_del,
                # New features from Kanchan's notebook:
                "vehicle_type": "Truck - Heavy",
                "temperature": 28.0,
                "humidity": 65.0 if m_int_ord < 0.3 else 85.0,
                "traffic_status": "Moderate",
                "waiting_time": 15,
                "inventory_level": 500,
                "asset_utilization": 80.0,
                "demand_forecast": 400,
                "order_value_inr": quantity * 1500,
                "road_quality": 0.65,
                "supplier_capacity": 80,
                "fuel_price_index": 105.0,
                "driver_experience": 10,
            }
            wast_inp = {
                "project_type": project_type, "state": state,
                "project_size_sqft": project_size,
                "project_duration_months": 12,
                "month_of_construction": order_month,
                "contractor_experience_yrs": contractor_exp,
                "num_workers": max(20, project_size // 200),
                "workforce_skill_level": workforce_skill,
                "supervision_quality": supervision,
                "material_type": material,
                "blueprint_quantity": quantity,
                "logistics_score": 0.6,
                "monsoon_intensity": m_int_ord,
                "monsoon_sensitivity": 0.5,
            }
            try:
                with st.spinner("NirmanAI is calculating risk..."):
                    result = predict_delay(
                        models["clf_delay"], models["reg_delay"], models["q_hat"],
                        models["enc_delay"], models["feat_delay"], inp
                    )
                    wastage_result = predict_wastage(
                        models["reg_wast"], models["reg_wast_lo"], models["reg_wast_hi"],
                        models["enc_wast"], models["feat_wast"], wast_inp
                    )
            except Exception:
                st.error("⚠️ Model prediction failed — showing demo estimate instead. "
                         "Re-run `python setup.py` to retrain the models.")
                models_ok = False

        if not models_ok:
            # Demo mode mock
            delay_prob = min(0.95, 0.15 + m_int_ord*0.4 + past_del*0.5 + (distance/5000)*0.2)
            pred_days  = int(delay_prob * 14) if delay_prob > 0.5 else 0
            risk_map   = {True: ("High" if delay_prob<0.75 else "Critical"), False: "Low"}
            result = {
                "delay_probability": round(delay_prob,3),
                "is_delayed": delay_prob>=0.5,
                "predicted_delay_days": pred_days,
                "ci_lower": max(0, pred_days-3),
                "ci_upper": pred_days+4,
                "risk_score": int(delay_prob*100),
                "risk_label": risk_map[delay_prob>=0.5],
                "top_risk_factors": [
                    f"Monsoon intensity: {m_int_ord:.0%}" if m_int_ord>0.4 else "Normal weather conditions",
                    f"Supplier delay rate: {past_del:.0%}",
                    f"Route distance: {distance} km"
                ]
            }

        # Display result
        color_map = {"Low":"green","Medium":"orange","High":"red","Critical":"red"}
        risk_label = result["risk_label"]

        r1, r2, r3 = st.columns(3)
        r1.metric("Delay Probability", f"{result['delay_probability']:.1%}")
        r2.metric("Predicted Delay",   f"{result['predicted_delay_days']:.0f} days",
                  f"Range: {result['ci_lower']:.0f}–{result['ci_upper']:.0f} days")
        r3.metric("Risk Score",        f"{result['risk_score']}/100")

        box_class = "alert-box" if risk_label in ["Critical","High"] else \
                    "warning-box" if risk_label=="Medium" else "ok-box"
        icon = {"Critical":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}[risk_label]

        factors_html = "".join([f"<li>{rf}</li>" for rf in result["top_risk_factors"]])
        st.markdown(f"""
        <div class="{box_class}">
            {icon} <strong>Risk Level: {risk_label}</strong>
            <p style="margin:8px 0 4px;font-size:13px;color:#aab4c8;">Key risk factors:</p>
            <ul style="margin:0;font-size:13px;color:#cdd6e4;">{factors_html}</ul>
        </div>
        """, unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=result["risk_score"],
            domain={"x":[0,1],"y":[0,1]},
            title={"text":"Risk Score", "font":{"color":"#e8eaf6","size":16}},
            number={"suffix":"/100","font":{"color":"#e8eaf6"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#8892b0"},
                "bar":{"color":{"Low":"#44ff88","Medium":"#ffcc00",
                                "High":"#ff8800","Critical":"#ff4444"}[risk_label]},
                "steps":[
                    {"range":[0,30],"color":"rgba(68,255,136,0.15)"},
                    {"range":[30,55],"color":"rgba(255,204,0,0.15)"},
                    {"range":[55,75],"color":"rgba(255,136,0,0.15)"},
                    {"range":[75,100],"color":"rgba(255,68,68,0.15)"},
                ],
                "threshold":{"line":{"color":"white","width":3},"value":result["risk_score"]}
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#1a1d2e", font_color="#e8eaf6", height=280
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        if wastage_result:
            st.markdown("#### 📦 Wastage Estimate for This Order")
            w1, w2, w3 = st.columns(3)
            w1.metric("Predicted Wastage",
                      f"{wastage_result['predicted_wastage_pct']:.1f}%",
                      f"Range: {wastage_result['wastage_range_low']:.1f}–{wastage_result['wastage_range_high']:.1f}%")
            w2.metric("Order With Buffer",
                      f"{wastage_result['actual_qty_estimate']:,.0f} units",
                      f"Blueprint: {wastage_result['blueprint_quantity']:,} units")
            w3.metric("Est. Cost Overrun",
                      f"₹{wastage_result['estimated_cost_overrun_inr']:,.0f}",
                      f"Category: {wastage_result['wastage_category']}")

            wast_box = {"High":"alert-box","Medium":"warning-box","Low":"ok-box"}[wastage_result["wastage_category"]]
            wast_factors = "".join(f"<li>{f}</li>" for f in wastage_result["risk_factors"])
            st.markdown(f"""
            <div class="{wast_box}">
                <strong>Wastage drivers:</strong>
                <ul style="margin:4px 0 0;font-size:13px;color:#cdd6e4;">{wast_factors}</ul>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TAB 4: ANALYTICS
# ═══════════════════════════════════════════
with tab4:
    st.markdown("### 📊 India Construction Supply Chain — Analytics")

    col_a1, col_a2 = st.columns(2)

    with col_a1:
        # State logistics heatmap
        states_data = {
            "State":["Gujarat","Maharashtra","Haryana","Tamil Nadu","Karnataka",
                     "Telangana","Punjab","West Bengal","Rajasthan","Madhya Pradesh",
                     "Uttar Pradesh","Kerala","Odisha","Bihar","Jharkhand"],
            "Logistics Score":[0.82,0.78,0.74,0.75,0.72,0.70,0.74,0.62,
                               0.65,0.55,0.58,0.68,0.50,0.45,0.42]
        }
        df_states = pd.DataFrame(states_data).sort_values("Logistics Score")
        fig_states = px.bar(df_states, x="Logistics Score", y="State",
                            orientation="h",
                            color="Logistics Score",
                            color_continuous_scale="RdYlGn",
                            title="State Logistics Quality Index")
        fig_states.update_layout(
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#e8eaf6", title_font_size=14,
            coloraxis_showscale=False, height=420
        )
        st.plotly_chart(fig_states, use_container_width=True)

    with col_a2:
        # Monthly delay probability trend
        months_label = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]
        delay_probs = [0.18,0.15,0.17,0.22,0.28,0.52,
                       0.68,0.63,0.48,0.35,0.20,0.16]
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=months_label, y=[p*100 for p in delay_probs],
            mode="lines+markers",
            line=dict(color="#4fc3f7", width=3),
            marker=dict(size=8,
                        color=["#ff4444" if p>0.5 else "#ffcc00" if p>0.3
                               else "#44ff88" for p in delay_probs]),
            fill="tozeroy", fillcolor="rgba(79,195,247,0.1)",
            name="Avg Delay Probability"
        ))
        fig_trend.add_hline(y=50, line_dash="dash", line_color="#ff4444",
                            annotation_text="High risk threshold")
        fig_trend.add_vline(x=current_month-1, line_dash="dot",
                            line_color="rgba(255,255,255,0.55)", annotation_text="Now")
        fig_trend.update_layout(
            title="Average Delay Probability by Month",
            yaxis_title="Delay Probability (%)",
            paper_bgcolor="#1a1d2e", plot_bgcolor="#1a1d2e",
            font_color="#e8eaf6", title_font_size=14, height=420,
            showlegend=False
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # India market stats
    st.markdown("---")
    st.markdown("#### 🇮🇳 India Construction Market Intelligence")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Market Size (2025)", "$640B", "2nd largest globally")
    m2.metric("Projected (2030)",   "$1.4T", "+119% growth")
    m3.metric("Annual Wastage Loss","₹1.5L Cr", "20-30% of materials")
    m4.metric("Projects Delayed",   "77%",  "Avg 20-month overrun")

    st.info("""
    **NirmanAI ROI Estimate**: On a typical ₹50 crore residential project, 
    NirmanAI reduces wastage by 8-12% and prevents 2-3 critical delivery delays.
    **Estimated savings: ₹18–28 lakhs per project.**  
    At ₹50K/month SaaS pricing, ROI is achieved within the first month.
    """)

# ═══════════════════════════════════════════
# TAB 5: SMART PROCUREMENT PLANNER
# ═══════════════════════════════════════════
with tab5:
    st.markdown("### 📋 Smart Procurement Planner")
    st.caption("AI-generated week-by-week order schedule to minimize delays and wastage")

    st.markdown("#### Enter Bill of Quantities")
    materials_selected = st.multiselect(
        "Select materials for this project",
        ["TMT Steel","OPC Cement","River Sand","Coarse Aggregate",
         "Fly Ash Bricks","Structural Steel","Electrical Cable","HDPE Pipes"],
        default=["TMT Steel","OPC Cement","River Sand"]
    )

    if materials_selected:
        bom_data = []
        qty_cols = st.columns(min(len(materials_selected), 4))
        for i, mat in enumerate(materials_selected):
            with qty_cols[i % len(qty_cols)]:
                qty = st.number_input(f"{mat}", 10, 10000, 100, key=f"bom_{mat}",
                                      help="Blueprint quantity")
            bom_data.append({"material_type": mat, "blueprint_quantity": qty})

        if st.button("📋 Generate Procurement Plan", type="primary", use_container_width=True):
            schedule_data = []
            total_buffer_cost = 0

            with st.spinner("NirmanAI is building your procurement schedule..."):
                if models_ok:
                    from train_delay_model import predict_delay
                    from train_wastage_model import predict_wastage
                    m_int_plan = monsoon_intensity(current_month)

                    for item in bom_data:
                        mat, qty = item["material_type"], item["blueprint_quantity"]
                        delay_res = predict_delay(
                            models["clf_delay"], models["reg_delay"], models["q_hat"],
                            models["enc_delay"], models["feat_delay"],
                            {
                                "month": current_month, "day_of_week": 0,
                                "quarter": (current_month-1)//3+1, "is_festival_period": 0,
                                "material_type": mat,
                                "supplier_tier": "Tier 2 (Regional Distributor)",
                                "origin_state": "Maharashtra", "destination_state": state,
                                "distance_km": 800, "order_quantity": qty,
                                "promised_lead_days": 14,
                                "monsoon_intensity": m_int_plan, "monsoon_sensitivity": 0.5,
                                "dest_logistics_score": 0.6, "orig_logistics_score": 0.7,
                                "dest_monsoon_severity": 0.6,
                                "supplier_reliability": 0.7, "past_delay_rate": 0.35,
                                "vehicle_type": "Truck - Medium",
                                "temperature": 32.0,
                                "humidity": 70.0,
                                "traffic_status": "Clear",
                                "waiting_time": 10,
                                "inventory_level": 600,
                                "asset_utilization": 85.0,
                                "demand_forecast": 500,
                                "order_value_inr": qty * 1500,
                                "road_quality": 0.7,
                                "supplier_capacity": 90,
                                "fuel_price_index": 100.0,
                                "driver_experience": 12,
                            }
                        )
                        wast_res = predict_wastage(
                            models["reg_wast"], models["reg_wast_lo"], models["reg_wast_hi"],
                            models["enc_wast"], models["feat_wast"],
                            {
                                "project_type": project_type, "state": state,
                                "project_size_sqft": project_size,
                                "project_duration_months": 12,
                                "month_of_construction": current_month,
                                "contractor_experience_yrs": contractor_exp,
                                "num_workers": max(20, project_size // 200),
                                "workforce_skill_level": workforce_skill,
                                "supervision_quality": supervision,
                                "material_type": mat, "blueprint_quantity": qty,
                                "logistics_score": 0.6,
                                "monsoon_intensity": m_int_plan, "monsoon_sensitivity": 0.5,
                            }
                        )
                        risk_level = delay_res["risk_label"]
                        order_week = {"Critical":"Week 1","High":"Week 1",
                                      "Medium":"Week 2","Low":"Week 3"}[risk_level]
                        reason = delay_res["top_risk_factors"][0]
                        buffer_qty = wast_res["actual_qty_estimate"]
                        total_buffer_cost += wast_res["estimated_cost_overrun_inr"]
                        schedule_data.append({
                            "Material": mat,
                            "Order By": order_week,
                            "Risk Level": risk_level,
                            "Delay Prob": f"{delay_res['delay_probability']:.0%}",
                            "Blueprint Qty": qty,
                            "Order Qty (with wastage buffer)": f"{buffer_qty:,.0f}",
                            "Wastage Est.": f"{wast_res['predicted_wastage_pct']:.1f}%",
                            "Reason": reason,
                        })
                else:
                    for i, item in enumerate(bom_data):
                        risk_level = ["Low","Medium","High","Critical"][i % 4]
                        order_week = ["Week 3","Week 2","Week 1","Week 1"][i % 4]
                        schedule_data.append({
                            "Material": item["material_type"],
                            "Order By": order_week,
                            "Risk Level": risk_level,
                            "Reason": ("Monsoon risk — order early"
                                       if risk_level in ["High","Critical"]
                                       else "Standard lead time"),
                        })

            st.markdown("#### Recommended Procurement Schedule")
            df_plan = pd.DataFrame(schedule_data).sort_values("Order By")

            def color_plan_risk(val):
                return {"Critical":"background-color:#ff444430",
                        "High":"background-color:#ff880025",
                        "Medium":"background-color:#ffcc0020",
                        "Low":"background-color:#44ff8815"}.get(val, "")

            st.dataframe(
                df_plan.style.map(color_plan_risk, subset=["Risk Level"]),
                use_container_width=True, hide_index=True
            )

            if models_ok and total_buffer_cost:
                st.warning(f"💸 Total projected wastage overrun across BoQ: **₹{total_buffer_cost:,.0f}** — "
                           "already included in the recommended order quantities above.")
            st.success("💡 Ordering high-risk materials 2 weeks early prevents ₹8-15 lakh in idle labor costs")
    else:
        st.info("Select at least one material above to generate a procurement plan.")

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#4a5568;font-size:12px;'>"
    "NirmanAI | Built by Team Aim-Nexus, IIT Madras | "
    "KAYA x IIT India Hackathon 2026 | "
    "Every prediction includes confidence intervals — because responsible AI never overpromises."
    "</p>",
    unsafe_allow_html=True
)
