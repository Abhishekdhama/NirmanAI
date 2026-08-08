"""
NirmanAI — Design System
========================
One place for the visual language, so every panel in app.py looks like it
belongs to the same product.

The look is deliberately NOT a generic dark "AI dashboard". It borrows from
site signage and drawing sets: a graphite/blueprint base, safety-amber as the
action colour, and risk colours that match what a site manager already reads on
a hoarding board (red / amber / green).
"""

RISK_COLORS = {
    "Critical": "#ff4d4f",
    "High": "#ff8c1a",
    "Medium": "#f2c200",
    "Low": "#3ddc84",
}

RISK_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

# Plotly layout shared by every chart so they read as one family.
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c7d0e0", size=12),
    title_font=dict(size=14, color="#eef2f9"),
    margin=dict(l=10, r=10, t=48, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
  --bg:        #0b0e14;
  --surface:   #131823;
  --surface-2: #1a2130;
  --line:      #253044;
  --line-soft: rgba(255,255,255,0.07);
  --text:      #eef2f9;
  --muted:     #8d9bb5;
  --dim:       #64738c;
  --amber:     #ffb020;
  --amber-dim: rgba(255,176,32,0.12);
  --blue:      #4cc2ff;
  --red:       #ff4d4f;
  --orange:    #ff8c1a;
  --yellow:    #f2c200;
  --green:     #3ddc84;
}

/* Streamlit sets its own font-family on the markdown containers, so applying
   Inter only to html/body loses on specificity. Name the component classes
   explicitly. The stack degrades to the platform UI font if Google Fonts is
   unreachable — a demo laptop is not always online. */
:root { --font: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }

html, body, [class*="css"], .stApp,
.masthead, .masthead-title, .masthead-sub,
.premise-card, .premise-label, .premise-body,
.kpi, .kpi-label, .kpi-value, .kpi-sub,
.alert, .alert-title, .alert-meta, .alert-because, .alert-action,
.panel, .panel-title, .panel-note,
.section-title, .section-sub, .why, .chip,
.empty, .empty-title, .empty-body, .footer-note {
  font-family: var(--font);
}
.alert-po, .section-step { font-family: 'JetBrains Mono', ui-monospace, monospace; }

.stApp { background: var(--bg); }
.block-container { padding-top: 1.4rem; max-width: 1500px; }

h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.02em; }
p, li, span, label { color: var(--text); }

/* ── Masthead ─────────────────────────────────────────── */
.masthead {
  display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
  padding: 20px 24px; border-radius: 16px; margin-bottom: 14px;
  background:
    linear-gradient(135deg, rgba(255,176,32,0.10) 0%, rgba(76,194,255,0.06) 55%, transparent 100%),
    var(--surface);
  border: 1px solid var(--line);
}
.masthead-mark {
  width: 46px; height: 46px; border-radius: 12px; flex: 0 0 auto;
  display: grid; place-items: center; font-size: 24px;
  background: var(--amber-dim); border: 1px solid rgba(255,176,32,0.35);
}
.masthead-text { flex: 1 1 320px; min-width: 260px; }
.masthead-title {
  font-size: 27px; font-weight: 800; color: var(--text); line-height: 1.1;
  margin: 0; letter-spacing: -0.03em;
}
.masthead-title em { font-style: normal; color: var(--amber); }
.masthead-sub { font-size: 13.5px; color: var(--muted); margin: 5px 0 0; line-height: 1.5; }

/* ── Chips & badges ───────────────────────────────────── */
.chip {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 999px; font-size: 11.5px; font-weight: 600;
  border: 1px solid var(--line); background: var(--surface-2); color: var(--muted);
  white-space: nowrap;
}
.chip-live  { border-color: rgba(61,220,132,0.4); background: rgba(61,220,132,0.10); color: var(--green); }
.chip-warn  { border-color: rgba(255,176,32,0.4); background: var(--amber-dim); color: var(--amber); }
.chip-row   { display: flex; gap: 8px; flex-wrap: wrap; }

/* ── The problem → solution band ──────────────────────── */
.premise {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 12px; margin-bottom: 16px;
}
.premise-card {
  padding: 15px 17px; border-radius: 13px;
  background: var(--surface); border: 1px solid var(--line);
  border-top: 2px solid var(--line);
}
.premise-card.is-problem  { border-top-color: var(--red); }
.premise-card.is-solution { border-top-color: var(--amber); }
.premise-card.is-proof    { border-top-color: var(--green); }
.premise-label {
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--dim); margin-bottom: 7px;
}
.premise-body { font-size: 13px; color: var(--text); line-height: 1.55; }
.premise-body strong { color: var(--amber); font-weight: 700; }

/* ── KPI tiles ────────────────────────────────────────── */
/* Fixed label and value bands keep the numbers on one baseline across a row,
   even when one card's label wraps to two lines. */
.kpi {
  padding: 16px 18px; border-radius: 13px; height: 100%;
  display: flex; flex-direction: column;
  background: var(--surface); border: 1px solid var(--line);
  transition: border-color .15s ease, transform .15s ease;
}
.kpi:hover { border-color: rgba(255,176,32,0.35); transform: translateY(-1px); }
.kpi-label {
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; line-height: 1.35;
  text-transform: uppercase; color: var(--dim); margin: 0 0 8px;
  min-height: 28px;
}
.kpi-value {
  font-size: 28px; font-weight: 800; line-height: 1.05; margin: 0;
  letter-spacing: -0.03em; font-variant-numeric: tabular-nums;
  min-height: 30px;
}
.kpi-sub {
  font-size: 11.5px; color: var(--muted); margin: 8px 0 0; line-height: 1.45;
  flex: 1 1 auto;
}

/* ── Alert cards ──────────────────────────────────────── */
.alert {
  border-radius: 12px; padding: 15px 17px; margin-bottom: 10px;
  background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--muted);
}
.alert.sev-Critical { border-left-color: var(--red);    background: linear-gradient(90deg, rgba(255,77,79,0.07), var(--surface) 55%); }
.alert.sev-High     { border-left-color: var(--orange); background: linear-gradient(90deg, rgba(255,140,26,0.07), var(--surface) 55%); }
.alert-head { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; margin-bottom: 8px; }
.alert-title { font-size: 14.5px; font-weight: 700; color: var(--text); }
.alert-po {
  font-family: 'JetBrains Mono', monospace; font-size: 11px;
  color: var(--dim); background: var(--surface-2);
  padding: 2px 7px; border-radius: 5px; border: 1px solid var(--line);
}
.alert-prob { margin-left: auto; font-size: 19px; font-weight: 800; font-variant-numeric: tabular-nums; }
.alert-meta { font-size: 12px; color: var(--muted); line-height: 1.6; margin-bottom: 9px; }
.alert-because {
  font-size: 12px; color: var(--muted); padding: 8px 11px; border-radius: 8px;
  background: rgba(255,255,255,0.03); border: 1px solid var(--line-soft); margin-bottom: 9px;
}
.alert-because b { color: var(--text); font-weight: 600; }
.alert-action {
  font-size: 12.5px; color: var(--text); font-weight: 600;
  padding: 9px 12px; border-radius: 8px;
  background: var(--amber-dim); border: 1px solid rgba(255,176,32,0.28);
}
.alert-action:before { content: "→ "; color: var(--amber); font-weight: 800; }

/* ── Generic panel ────────────────────────────────────── */
.panel {
  border-radius: 13px; padding: 18px 20px;
  background: var(--surface); border: 1px solid var(--line); margin-bottom: 12px;
}
.panel-title { font-size: 14px; font-weight: 700; color: var(--text); margin: 0 0 4px; }
.panel-note  { font-size: 12px; color: var(--muted); line-height: 1.55; margin: 0; }

/* ── Section headers ──────────────────────────────────── */
.section {
  display: flex; align-items: baseline; gap: 12px;
  margin: 6px 0 4px; flex-wrap: wrap;
}
.section-step {
  font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
  color: var(--amber); background: var(--amber-dim);
  border: 1px solid rgba(255,176,32,0.3);
  padding: 3px 9px; border-radius: 6px; white-space: nowrap;
}
.section-title { font-size: 19px; font-weight: 700; color: var(--text); letter-spacing: -0.02em; }
.section-sub { font-size: 12.5px; color: var(--muted); margin: 2px 0 14px; line-height: 1.55; }

/* ── Explainer callout ────────────────────────────────── */
.why {
  font-size: 12px; color: var(--muted); line-height: 1.6;
  padding: 11px 14px; border-radius: 9px; margin: 4px 0 14px;
  background: rgba(76,194,255,0.05); border: 1px solid rgba(76,194,255,0.18);
}
.why b { color: var(--blue); font-weight: 600; }

/* ── Empty state ──────────────────────────────────────── */
.empty {
  text-align: center; padding: 42px 24px; border-radius: 13px;
  background: var(--surface); border: 1px dashed var(--line);
}
.empty-icon { font-size: 30px; margin-bottom: 10px; opacity: 0.75; }
.empty-title { font-size: 15px; font-weight: 700; color: var(--text); margin-bottom: 5px; }
.empty-body { font-size: 12.5px; color: var(--muted); line-height: 1.6; }

/* ── Tabs ─────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  gap: 3px; border-bottom: 1px solid var(--line); padding-bottom: 0;
}
.stTabs [data-baseweb="tab"] {
  font-size: 13.5px; font-weight: 600; padding: 10px 16px;
  color: var(--muted); border-radius: 9px 9px 0 0;
}
.stTabs [aria-selected="true"] { color: var(--amber) !important; background: var(--amber-dim); }

/* ── Streamlit widget polish ──────────────────────────── */
[data-testid="stMetric"] {
  background: var(--surface); border: 1px solid var(--line);
  border-radius: 11px; padding: 13px 15px;
}
[data-testid="stMetricLabel"] p { font-size: 11.5px !important; color: var(--dim) !important; font-weight: 600; }
[data-testid="stMetricValue"] { font-size: 24px !important; font-weight: 800 !important; letter-spacing: -0.02em; }

section[data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }

.stButton > button {
  border-radius: 9px; font-weight: 650; font-size: 13.5px;
  border: 1px solid var(--line); transition: all .15s ease;
}
.stButton > button[kind="primary"] {
  background: var(--amber); color: #14181f; border-color: var(--amber);
}
.stButton > button[kind="primary"]:hover { background: #ffc247; border-color: #ffc247; color: #14181f; }

hr { border-color: var(--line); margin: 18px 0; }
[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 11px; }

/* Sidebar label sizing */
section[data-testid="stSidebar"] label p { font-size: 12.5px !important; }

.sidebar-heading {
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--dim); margin: 16px 0 6px;
}
.footer-note {
  text-align: center; color: var(--dim); font-size: 11.5px;
  line-height: 1.7; padding: 8px 0 20px;
}
</style>
"""


def masthead(status_chips: str) -> str:
    return f"""
    <div class="masthead">
      <div class="masthead-mark">🏗️</div>
      <div class="masthead-text">
        <p class="masthead-title">Nirman<em>AI</em></p>
        <p class="masthead-sub">
          Delivery-risk and material-wastage forecasting for Indian construction sites —
          so a site manager knows which orders will slip <em style="color:var(--amber);font-style:normal;">before</em> they slip.
        </p>
      </div>
      <div class="chip-row">{status_chips}</div>
    </div>"""


def premise_band(risk_line: str, proof_line: str) -> str:
    return f"""
    <div class="premise">
      <div class="premise-card is-problem">
        <div class="premise-label">The problem</div>
        <div class="premise-body">
          <strong>77%</strong> of Indian construction projects run late, and
          <strong>20–30%</strong> of materials are wasted on site. Both are found out
          after the money is spent.
        </div>
      </div>
      <div class="premise-card is-solution">
        <div class="premise-label">What NirmanAI does</div>
        <div class="premise-body">
          Scores every open purchase order for delay risk, sizes the wastage buffer per
          material, and simulates what a slip does to the <strong>whole schedule</strong>.
        </div>
      </div>
      <div class="premise-card is-proof">
        <div class="premise-label">On this project, right now</div>
        <div class="premise-body">{risk_line}</div>
      </div>
      <div class="premise-card">
        <div class="premise-label">Why trust it</div>
        <div class="premise-body">{proof_line}</div>
      </div>
    </div>"""


def section(step: str, title: str, subtitle: str = "") -> str:
    step_html = f'<span class="section-step">{step}</span>' if step else ""
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    return f"""
    <div class="section">{step_html}<span class="section-title">{title}</span></div>
    {sub}"""


def why(text: str) -> str:
    return f'<div class="why">{text}</div>'


def empty_state(icon: str, title: str, body: str) -> str:
    return f"""
    <div class="empty">
      <div class="empty-icon">{icon}</div>
      <div class="empty-title">{title}</div>
      <div class="empty-body">{body}</div>
    </div>"""


def kpi(label: str, value: str, sub: str, color: str = "#eef2f9") -> str:
    return f"""
    <div class="kpi">
      <p class="kpi-label">{label}</p>
      <p class="kpi-value" style="color:{color};">{value}</p>
      <p class="kpi-sub">{sub}</p>
    </div>"""


def format_interval(lower: float, upper: float) -> str:
    """
    Render a conformal interval readably.

    Conformal lower bounds are often exactly 0 — honest, but "0–21 days" next to
    an expectation of 9 days reads like a bug. Phrase it as a ceiling instead.
    """
    if upper <= 0:
        return "under a day"
    if lower <= 0.5:
        return f"up to {upper:.0f} days"
    return f"{lower:.0f}–{upper:.0f} days"


def alert_card(a: dict) -> str:
    color = RISK_COLORS.get(a["severity"], "#8d9bb5")
    factors = " · ".join(a["all_factors"][:3])
    window = format_interval(a["ci_lower"], a["ci_upper"])
    return f"""
    <div class="alert sev-{a['severity']}">
      <div class="alert-head">
        <span class="alert-po">{a['po_id']}</span>
        <span class="alert-title">{a['title']}</span>
        <span class="alert-prob" style="color:{color};">{a['probability']:.0%}</span>
      </div>
      <div class="alert-meta">
        {a['supplier']} &nbsp;·&nbsp; ₹{a['value']:,.0f} at stake &nbsp;·&nbsp;
        if late, expect <b style="color:{color};">{a['expected_days']:.0f} days</b>
        (90% of similar orders: {window})
      </div>
      <div class="alert-because"><b>Model's reasoning:</b> {factors}</div>
      <div class="alert-action">{a['action']}</div>
    </div>"""


def apply_plot_theme(fig, height=None, title=None):
    """Apply the shared chart styling to a Plotly figure."""
    layout = dict(PLOT_LAYOUT)
    if height:
        layout["height"] = height
    if title:
        layout["title"] = title
    fig.update_layout(**layout)
    return fig
