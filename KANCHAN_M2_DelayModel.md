# 📊 Kanchan's Work — Delay Prediction Module
# Team Aim-Nexus | KAYA x IIT Hackathon 2026
# Your Role: Member 2 — ML Engineer (Delay Prediction)

---

## What You Own
The **Delay Predictor** — the model that predicts which material deliveries
will be late, by how many days, and WHY. This is the most critical module
because it's the one with the most "wow factor" in the demo.

---

## What's Already Built For You

The file `train_delay_model.py` is already written and working.
It contains:

- ✅ Data loading + feature engineering
- ✅ XGBoost classifier (will it be delayed? yes/no)
- ✅ LightGBM regressor (how many days late?)
- ✅ Conformal prediction (confidence intervals — e.g., "9 days ± 3")
- ✅ SHAP explainability (why is it at risk?)
- ✅ A `predict_delay()` function the dashboard calls
- ✅ Models saved to `models/` folder automatically

**You do NOT rewrite this from scratch. You run it, understand it, improve it.**

---

## Step 0 — Setup (Do This First)

```bash
# Open D:\kaya_IIT_hackathon in VS Code terminal
python setup.py
```

This installs all dependencies, generates data, and trains both models.
Takes ~5-10 minutes. Run it once.

---

## Your Tasks — Phase 1 (Now → July 10, Pitch Deck)

### Day 1 (July 7) — Understand the code
- [ ] Read `train_delay_model.py` top to bottom
- [ ] Read `generate_data.py` — understand what features are being generated
- [ ] Run the training: `python train_delay_model.py`
- [ ] Check the output — what's the AUC-ROC? What does the SHAP plot look like?
- [ ] Open `reports/shap_delay.png` — understand which features matter most

### Day 2 (July 8) — Improve & document
- [ ] Try tuning XGBoost: increase `n_estimators` to 500, try `max_depth=7`
- [ ] Run again and check if AUC improves
- [ ] Write down your model's final metrics in a comment at the top of the file:
  ```python
  # Final Metrics:
  # AUC-ROC: X.XX
  # Precision (Delayed class): X.XX
  # Recall (Delayed class): X.XX
  # Conformal interval coverage: ~80%
  ```
- [ ] Take a screenshot of the SHAP bar chart — this goes in the pitch deck slide 5

### Day 3 (July 9) — Pitch deck contribution
- [ ] Write 3 bullet points about your model for slide 5 (How It Works)
- [ ] Prepare one "interesting finding" from the data:
  e.g., "In August, river sand deliveries from Rajasthan have a 76% delay rate"
  This goes into the pitch as a real, credible insight
- [ ] Make sure `predict_delay()` works when called from `app.py`
- [ ] Test it: run `streamlit run app.py`, go to "Predict New Order" tab, check if prediction works

---

## Your Tasks — Phase 2 (July 11 → July 31, Prototype)

### Week 1 (July 11-17) — Make the model production-ready
- [ ] **Add more India-specific features** to the dataset generator. Ideas:
  - `road_quality_score` per state (NH vs SH vs district roads)
  - `supplier_size` (small/medium/large factory)
  - `order_value_inr` — high-value orders get priority
  - `season` categorical feature (pre-monsoon, monsoon, post-monsoon, winter)
- [ ] Retrain with new features, check if AUC improves
- [ ] Add a **calibration plot** — shows that when model says 80% confidence, it's actually right ~80% of the time. This is your AI safety contribution.
  ```python
  from sklearn.calibration import calibration_curve
  # Plot fraction_of_positives vs mean_predicted_probability
  ```

### Week 2 (July 18-24) — Add one new thing
Pick ONE of these to add (don't try all):

**Option A (Recommended): Multi-day delay breakdown**
Instead of just "9 days late", show:
- Day-by-day risk: "30% chance of >3 days, 15% chance of >7 days, 5% chance of >14 days"
- Use multiple quantile predictions

**Option B: Supplier performance tracker**
Given a supplier name/ID:
- Their historical delay rate
- Their performance trend over last 6 months
- Recommendation: "Avoid this supplier in monsoon season"

**Option C: Route risk map**
- Given origin → destination state pair
- Show a risk score for that route
- Integrate with a simple map visualization in the dashboard

### Week 3 (July 25-31) — Polish + submit
- [ ] Clean up your code — add docstrings to every function
- [ ] Make sure all models save properly to `models/`
- [ ] Write a `model_card.md` in the `models/` folder:
  - What data was used
  - Model architecture
  - Key metrics
  - Known limitations
- [ ] Test that `predict_delay()` handles edge cases (unknown state, missing value)

---

## Key Concepts You Need to Understand

### Why Conformal Prediction?
Most models give you a single number: "9 days late."
Conformal prediction gives you: "9 days late ± 3 days, with 80% confidence."
This is **calibrated uncertainty** — it's honest about what it doesn't know.
In construction, an overconfident wrong prediction costs crores.
This is what makes SupplyMind different from generic tools.

Read this if you want to go deeper:
- `from mapie.regression import MapieRegressor` — another way to do it
- Search: "conformal prediction tutorial Python" on Towards Data Science

### Why SHAP?
SHAP (SHapley Additive exPlanations) tells you WHY the model made a decision.
"This delivery is at risk because: monsoon intensity is 85% (most important),
supplier delay rate is 38% (second), route distance is 1200km (third)."
This is what judges call "explainable AI." It's a big deal.

### The Feature That Matters Most (Spoiler)
Based on the data generation logic, `past_delay_rate` and `monsoon_intensity`
will be the top SHAP features. If your SHAP plot doesn't show these at the top,
something went wrong — check the data.

---

## How Your Module Connects to Others

```
Kanchan's delay_classifier.pkl  ──→  app.py (Tab 1 + Tab 3)
Kanchan's delay_regressor.pkl   ──→  app.py (Tab 3: "Predict New Order")
Kanchan's delay_q_hat.pkl       ──→  app.py (confidence intervals shown in UI)
reports/shap_delay.png          ──→  Pitch Deck Slide 5 (screenshot)
```

Muskan's dashboard calls your `predict_delay()` function directly.
Make sure the function signature doesn't change.

---

## If You Get Stuck

1. **Model not training?** Run `python setup.py` first. Check all packages are installed.
2. **SHAP throwing errors?** Update: `pip install shap==0.45.1`
3. **AUC seems too high (>0.95)?** That's expected — synthetic data has clean patterns.
   Mention this honestly in Q&A: "On synthetic data, AUC is 0.92. Real-world data
   would be noisier, but we expect AUC > 0.80 based on similar supply chain literature."
4. **Confused about conformal prediction?** Ask the team lead (main coder built it).

---

## Deliverables Summary

| By When | What |
|---|---|
| July 9 | Model running, SHAP screenshot ready, 3 bullet points for pitch slide 5 |
| July 17 | Improved model with new features, calibration plot |
| July 24 | One new feature (Option A/B/C above) added and working |
| July 31 | Clean code + model_card.md + tested predict_delay() function |

---

**One last thing:** Attend the Track Awareness Session (if not already done).
Take notes on what KAYA engineers say about procurement pain points.
Those real pain points should show up in our demo conversations.

Good luck Kanchan! 🚀
