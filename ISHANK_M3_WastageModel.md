# 📦 Ishank's Work — Wastage Estimation Module
# Team Aim-Nexus | KAYA x IIT Hackathon 2026
# Your Role: Member 3 — ML Engineer (Wastage Estimation)

---

## What You Own
The **Wastage Estimator** — the model that predicts how much material a
construction project will ACTUALLY use vs. what the blueprint says.
This is the "money slide" of our pitch. When we say "₹18-28 lakhs saved
per project," that comes from your model's numbers.

---

## What's Already Built For You

The file `train_wastage_model.py` is already written and working.
It contains:

- ✅ Data loading + feature engineering
- ✅ LightGBM regressor (predicts wastage %)
- ✅ Quantile regression (10th/90th percentile — gives uncertainty range)
- ✅ LightGBM classifier (Low / Medium / High wastage category)
- ✅ SHAP explainability (why is wastage high?)
- ✅ A `predict_wastage()` function the dashboard calls
- ✅ Cost overrun estimate in ₹ (using material price lookup)
- ✅ Models saved to `models/` folder automatically

**You do NOT rewrite this from scratch. You run it, understand it, improve it.**

---

## Step 0 — Setup (Do This First)

```bash
# Open D:\kaya_IIT_hackathon in VS Code terminal
python setup.py
```

This installs all dependencies, generates data, and trains all models.
Takes ~5-10 minutes. Run it once.

---

## Your Tasks — Phase 1 (Now → July 10, Pitch Deck)

### Day 1 (July 7) — Understand the code
- [ ] Read `train_wastage_model.py` top to bottom
- [ ] Read the wastage section in `generate_data.py` (the `generate_wastage_dataset()` function)
- [ ] Run the training: `python train_wastage_model.py`
- [ ] Look at the output:
  - What's the MAE (Mean Absolute Error) in wastage %?
  - What does the SHAP plot (`reports/shap_wastage.png`) look like?
  - Which features matter most?
- [ ] Open `data/material_wastage.csv` in Excel/VS Code — browse the data

### Day 2 (July 8) — Find insights + improve
- [ ] Load the dataset and do basic analysis:
  ```python
  import pandas as pd
  df = pd.read_csv("data/material_wastage.csv")

  # Which material has the highest average wastage?
  print(df.groupby("material_type")["wastage_pct"].mean().sort_values(ascending=False))

  # Which state has the worst wastage?
  print(df.groupby("state")["wastage_pct"].mean().sort_values(ascending=False))

  # Unskilled vs Skilled workforce difference?
  print(df.groupby("workforce_skill_level")["wastage_pct"].mean())
  ```
- [ ] Write down 3 "insight bullets" from this analysis — real numbers, e.g.:
  - "River Sand has avg 26% wastage vs 5% for TMT Steel"
  - "Unskilled workforce wastes 40% more than Skilled workforce"
  - "July-August (monsoon) sees 35% higher cement wastage than January"
- [ ] These insights go DIRECTLY into the pitch deck (Slide 8 and the demo script)

### Day 3 (July 9) — Pitch deck + demo check
- [ ] Take a screenshot of SHAP plot (`reports/shap_wastage.png`)
- [ ] Run `streamlit run app.py` → go to "Wastage Intelligence" tab
- [ ] Verify the wastage table looks correct
- [ ] Send your 3 insight bullets to team lead for the pitch deck

---

## Your Tasks — Phase 2 (July 11 → July 31, Prototype)

### Week 1 (July 11-17) — Strengthen the model

**Task 1: Add a cost savings calculator**

Write a function that answers: "If we improve supervision from Poor → Good,
how much money do we save on this project?"

```python
def calculate_savings(current_config: dict, improved_config: dict,
                      reg, reg_lo, reg_hi, encoders, feature_names) -> dict:
    """
    Compares two scenarios and returns:
    - current_wastage_pct
    - improved_wastage_pct
    - wastage_reduction_pct
    - cost_saving_inr
    """
    current  = predict_wastage(reg, reg_lo, reg_hi, encoders, feature_names, current_config)
    improved = predict_wastage(reg, reg_lo, reg_hi, encoders, feature_names, improved_config)
    
    saving = current["estimated_cost_overrun_inr"] - improved["estimated_cost_overrun_inr"]
    return {
        "current_wastage":   current["predicted_wastage_pct"],
        "improved_wastage":  improved["predicted_wastage_pct"],
        "cost_saving_inr":   saving,
        "reduction_pct":     current["predicted_wastage_pct"] - improved["predicted_wastage_pct"]
    }
```

This function is what powers the "Scenario Analysis" section in the dashboard.

**Task 2: Add material-specific benchmarks**

Add a dictionary with industry benchmark wastage rates (cite CIDC report):
```python
INDUSTRY_BENCHMARKS = {
    "TMT Steel":        {"acceptable": 5.0,  "source": "CIDC 2023"},
    "OPC Cement":       {"acceptable": 8.0,  "source": "CIDC 2023"},
    "River Sand":       {"acceptable": 12.0, "source": "CIDC 2023"},
    "Coarse Aggregate": {"acceptable": 6.0,  "source": "CIDC 2023"},
    "Fly Ash Bricks":   {"acceptable": 10.0, "source": "CIDC 2023"},
    # ... add all materials
}
```

When the model predicts wastage, compare to benchmark:
- "Your cement wastage is 14.6% vs industry benchmark of 8% — 82% above benchmark"

### Week 2 (July 18-24) — Build one new thing

Pick ONE:

**Option A (Recommended): Project-level total overrun estimator**
Given a full project Bill of Quantities (list of materials + quantities),
sum up total predicted cost overrun across all materials:

```python
def estimate_project_overrun(bom: list[dict], project_context: dict,
                              reg, reg_lo, reg_hi, encoders, feature_names) -> dict:
    """
    bom = [{"material_type": "OPC Cement", "blueprint_quantity": 500}, ...]
    Returns total overrun, per-material breakdown, top 3 risk materials
    """
    results = []
    for item in bom:
        inp = {**project_context, **item}
        pred = predict_wastage(reg, reg_lo, reg_hi, encoders, feature_names, inp)
        results.append({**item, **pred})
    
    total_overrun = sum(r["estimated_cost_overrun_inr"] for r in results)
    top_risks = sorted(results, key=lambda x: x["estimated_cost_overrun_inr"], reverse=True)[:3]
    return {"total_overrun_inr": total_overrun, "breakdown": results, "top_risks": top_risks}
```

**Option B: Wastage trend analysis**
Show how wastage changes month-by-month for a given material + location.
This creates a beautiful chart for the dashboard.

### Week 3 (July 25-31) — Clean up + submit
- [ ] Add docstrings to every function
- [ ] Test `predict_wastage()` with edge cases (unknown material, 0 workers)
- [ ] Create `models/wastage_model_card.md`:
  - Dataset: 3,000 synthetic records
  - Calibrated against: CIDC 2023 benchmarks
  - MAE: X.X% wastage
  - 80% prediction interval empirical coverage: ~80%
  - Key limitations: trained on synthetic data

---

## Key Concepts You Need to Understand

### Why Quantile Regression for Wastage?
For delays, we used conformal prediction.
For wastage, we use **quantile regression** — two separate models:
- One predicts the 10th percentile (optimistic scenario)
- One predicts the 90th percentile (pessimistic scenario)

Together they give: "Wastage will be between 8% and 22%, with 14% most likely."
This is how you show uncertainty without overcomplicating things.

### The Data Generation Logic (Important!)
Wastage in our data is driven by these multipliers (in `generate_data.py`):
- `skill_factor`: Unskilled = 1.4x more waste, Expert = 0.9x
- `supervision_factor`: Poor = 1.35x, Excellent = 0.85x
- `monsoon_factor`: Depends on material's monsoon sensitivity
- `experience_factor`: New contractors waste more

Your SHAP plot should show `workforce_skill_level` and `supervision_quality`
as the top features. If they're not — check the data.

### The ₹ Numbers Matter
When judges hear "our model predicts 14.6% cement wastage," they nod.
When they hear "that's ₹27,740 extra cost for THIS project's cement order alone,
and across all 6 materials the overrun is ₹5.38 lakhs on a small residential project,"
they lean forward. **Always translate wastage % into rupees.**

---

## How Your Module Connects to Others

```
Ishank's wastage_regressor.pkl    ──→  app.py (Tab 2 + Tab 3)
Ishank's wastage_regressor_lo.pkl ──→  app.py (uncertainty range shown in UI)
Ishank's wastage_regressor_hi.pkl ──→  app.py (uncertainty range shown in UI)
reports/shap_wastage.png          ──→  Pitch Deck Slide 5 (screenshot)
calculate_savings() function      ──→  app.py Tab 2: Scenario Analysis table
```

Muskan's dashboard calls your `predict_wastage()` and `calculate_savings()` functions.
Don't change the function signature without telling Muskan.

---

## If You Get Stuck

1. **Model giving negative wastage?** Add `np.clip(prediction, 0.5, 60.0)` — already done in code.
2. **Quantile model: lower > upper?** Add `max(wastage_pct, wastage_lo)` for lower bound.
3. **SHAP plot looks weird?** Make sure you're using `shap.TreeExplainer` not `shap.Explainer`.
4. **MAE seems too low (<3%)?** That's okay for synthetic data. Real-world would be 4-6%.

---

## Deliverables Summary

| By When | What |
|---|---|
| July 9 | Model running, 3 data insights ready, SHAP screenshot taken |
| July 17 | `calculate_savings()` function working + industry benchmarks added |
| July 24 | One new feature (Option A/B) complete and tested |
| July 31 | Clean code + `wastage_model_card.md` + edge cases handled |

---

**Key thing to remember:** Your module's output is what justifies the business case.
Every rupee in the "₹18-28 lakhs saved per project" claim comes from your model.
Make those numbers real and defensible. Good luck Ishank! 🚀
