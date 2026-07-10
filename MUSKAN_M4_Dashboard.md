# 🖥️ Muskan's Work — Dashboard & Integration
# Team Aim-Nexus | KAYA x IIT Hackathon 2026
# Your Role: Member 4 — Full-Stack (Dashboard + Integration)

---

## What You Own
The **Streamlit Dashboard** (`app.py`) — the live, interactive demo that judges
will actually SEE. You own the look, feel, and experience of NirmanAI.
This is what makes or breaks the "wow factor."
You also own the **KAYA Integration mock** — the slide/screen that could win us
the internship offer.

---

## What's Already Built For You

The file `app.py` is a complete, working Streamlit dashboard.
It has 4 tabs, dark theme, Plotly charts, and connects to all ML models.

What's already there:
- ✅ Dark theme with custom CSS (Inter font, consistent colors)
- ✅ Sidebar: project context inputs (state, workforce, month, etc.)
- ✅ KPI row: 5 metric cards at the top
- ✅ Tab 1 — Live Delivery Monitor (alert banners + delivery table + charts)
- ✅ Tab 2 — Wastage Intelligence (table + bar chart + scenario analysis)
- ✅ Tab 3 — Predict New Order (form inputs + gauge chart + risk result)
- ✅ Tab 4 — Analytics (state logistics heatmap + monthly trend chart)
- ✅ Footer with team info

**Your job: run it, understand it, polish it, and add 2 things (below).**

---

## Step 0 — Setup (Do This First)

```bash
# Step 1: Install and train
python setup.py

# Step 2: Launch dashboard
streamlit run app.py
```

Opens at: **http://localhost:8501**

Walk through all 4 tabs. Click everything. See what works and what needs polish.

---

## Your Tasks — Phase 1 (Now → July 10, Pitch Deck)

### Day 1 (July 7) — Run, explore, note issues
- [ ] Run `python setup.py` successfully
- [ ] Run `streamlit run app.py`
- [ ] Go through all 4 tabs — note anything that looks off visually
- [ ] Take 3 screenshots of the best-looking parts:
  - Tab 1: the alert banners + risk table
  - Tab 3: a completed prediction with the gauge chart
  - Tab 4: the state logistics heatmap
- [ ] Send these screenshots to team lead — they go in **Pitch Deck Slide 6**

### Day 2 (July 8) — Polish what's there
- [ ] Make the KPI cards look sharper. Currently they use `st.metric()`.
  Try wrapping them in custom HTML for better styling:
  ```python
  st.markdown("""
  <div class="metric-card">
      <h3 style="color:#4fc3f7;margin:0;">23</h3>
      <p style="color:#8892b0;margin:4px 0 0;">Active Orders</p>
  </div>
  """, unsafe_allow_html=True)
  ```
- [ ] Make the delivery risk table more visual:
  Add emoji status column: 🔴 Critical, 🟠 High, 🟡 Medium, 🟢 Low
  (Already in the code — verify it displays correctly)
- [ ] Check that Tab 3 "Predict New Order" button works end-to-end
- [ ] Take final polished screenshots for pitch deck

### Day 3 (July 9) — KAYA integration mock screen
- [ ] **This is the most important task for the pitch deck.**
  Create a screenshot/mockup of what NirmanAI would look like
  INSIDE KAYA's platform. Create a new section in `app.py`:

  ```python
  # Add this at bottom of Tab 1 or as a 5th tab
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
  ```

- [ ] Take a screenshot of this section — it's **Pitch Deck Slide 9**

---

## Your Tasks — Phase 2 (July 11 → July 31, Prototype)

### Week 1 (July 11-17) — Connect real models to Tab 3

Currently Tab 3 runs in "demo mode" (mock predictions) when models aren't loaded.
Make it use the REAL models properly.

The models load at the top of `app.py`:
```python
@st.cache_resource
def load_models():
    # Already implemented — check if it returns True
```

In the prediction button section, connect:
```python
from train_delay_model import predict_delay
from train_wastage_model import predict_wastage

# These functions already exist — just make sure they're called correctly
result = predict_delay(clf_delay, reg_delay, q_hat, enc_delay, feat_delay, inp)
```

Verify with Kanchan and Ishank that their function signatures match.

### Week 2 (July 18-24) — Add Smart Procurement Planner (Tab 5)

This is the "third module" — the one that ties everything together.
Add a new tab: **"📋 Smart Procurement Plan"**

```python
with tab5:  # Add tab5 to the tabs line
    st.markdown("### 📋 Smart Procurement Planner")
    st.caption("AI-generated week-by-week order schedule to minimize delays and wastage")

    # Bill of Quantities input
    st.markdown("#### Enter Bill of Quantities")
    materials_selected = st.multiselect(
        "Select materials for this project",
        ["TMT Steel","OPC Cement","River Sand","Coarse Aggregate",
         "Fly Ash Bricks","Structural Steel","Electrical Cable","HDPE Pipes"]
    )

    if materials_selected:
        bom_data = []
        for mat in materials_selected:
            qty = st.number_input(f"{mat} — Blueprint Quantity", 10, 10000, 100, key=mat)
            bom_data.append({"material_type": mat, "blueprint_quantity": qty})

        if st.button("📋 Generate Procurement Plan", type="primary"):
            # Generate week-by-week schedule
            st.markdown("#### Recommended Procurement Schedule")
            
            schedule_data = []
            for i, item in enumerate(bom_data):
                risk_level = ["Low","Medium","High","Critical"][i % 4]
                order_week = ["Week 1","Week 1","Week 2","Week 3"][i % 4]
                schedule_data.append({
                    "Material": item["material_type"],
                    "Order By": order_week,
                    "Risk Level": risk_level,
                    "Reason": f"{'Monsoon risk — order early' if risk_level in ['High','Critical'] else 'Standard lead time'}"
                })
            
            st.dataframe(pd.DataFrame(schedule_data), use_container_width=True)
            st.success("💡 Ordering high-risk materials 2 weeks early prevents ₹8-15 lakh in idle labor costs")
```

Improve this logic in Week 2 using Kanchan's and Ishank's real model outputs.

### Week 3 (July 25-31) — Final polish checklist
- [ ] All 5 tabs working end-to-end with real models
- [ ] Mobile-friendly layout (test in browser with narrower window)
- [ ] No hardcoded mock values where real model predictions should be
- [ ] Error handling — if a model fails to load, show a clear message
- [ ] Loading spinners on predictions:
  ```python
  with st.spinner("NirmanAI is calculating risk..."):
      result = predict_delay(...)
  ```
- [ ] Clean up commented-out code
- [ ] Test on a fresh machine (different team member's laptop) — does it run?

---

## Design Guidelines (Keep Consistent)

### Color Palette (DO NOT change these)
```
Background:     #0f1117
Card background: #1a1d2e
Border:          #2a2d3e
Primary blue:    #4fc3f7
Warning orange:  #ff8800
Critical red:    #ff4444
Safe green:      #44ff88
Text:            #e8eaf6
Subtext:         #8892b0
```

### What Makes the Dashboard Look Good
1. **Dark background throughout** — never use white/light sections
2. **Color-coded risk** — always use red/orange/yellow/green consistently
3. **Big numbers for KPIs** — judges need to read from across a table
4. **Plotly charts** — already used. Keep using Plotly. Don't mix with Matplotlib in UI.
5. **Alert banners** — the HTML `div` boxes with colored left borders are the best visual
6. **Consistent emoji use** — one emoji per section header, not everywhere

### Things That Will Look Bad (Avoid)
- ❌ White backgrounds or light mode
- ❌ Too much text on one page
- ❌ Raw error messages visible to user
- ❌ Unstyled `st.dataframe()` without custom colors
- ❌ Missing loading states (spinning forever without feedback)

---

## How Your Module Connects to Others

```
Kanchan's predict_delay()    ──→  app.py Tab 3 (Predict New Order)
Ishank's predict_wastage()   ──→  app.py Tab 2 + Tab 3
Ishank's calculate_savings() ──→  app.py Tab 2 (Scenario Analysis)

app.py Tab 1 screenshots     ──→  Pitch Deck Slide 6
app.py KAYA mock section     ──→  Pitch Deck Slide 9
```

You're the last step before the judge sees the product.
Make it look like a real product, not a student project.

---

## How to Take Good Demo Screenshots for the Pitch

1. **Set browser zoom to 90%** — shows more content, cleaner
2. **Use Chrome** — renders CSS better than Edge/Firefox
3. **Dark background should fill the entire screenshot** — no white browser chrome visible
4. **Crop tightly** around the interesting part
5. **Take in 1920×1080 resolution** if possible
6. **Test with real inputs** before screenshotting — the risk gauge at "High" looks better than "Low"

Try these inputs for the best-looking Tab 3 prediction:
- Material: River Sand
- Supplier: Tier 3 (Local Supplier)
- Origin: Rajasthan
- Destination: Bihar
- Distance: 1400 km
- Past delay rate: 55%
- Month: August

This should give you a "Critical" risk prediction — red gauge, dramatic looking. Screenshot that.

---

## Deliverables Summary

| By When | What |
|---|---|
| July 9 | 3 dashboard screenshots for pitch deck + KAYA integration mock section |
| July 17 | Tab 3 connected to real models (not demo mode) |
| July 24 | Tab 5 (Smart Procurement Planner) working |
| July 31 | All tabs polished + loading states + error handling + tested on fresh machine |

---

**Remember:** You're the one judges will be watching when the team lead runs the demo.
The dashboard needs to load fast, look sharp, and not crash.
Test it 10 times before the finale. Good luck Muskan! 🚀
