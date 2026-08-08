# NirmanAI — Deployment Guide

**For:** the team member doing the deploy.
**Assumes:** the code is already pushed to a public GitHub repo. You need no local setup — both services build from GitHub in the browser.

**Time:** ~15 minutes for the dashboard, ~10 more for the API.

---

## What you are deploying

Two separate services, because they are two different kinds of app:

| Service | What it is | Host | Why there |
|---|---|---|---|
| **Dashboard** | Streamlit app (`app.py`) — the demo | **Streamlit Community Cloud** | Free, purpose-built for Streamlit, one-click from GitHub |
| **REST API** | FastAPI app (`api.py`) — `/docs` | **Render** (free web service) | Streamlit Cloud runs *only* Streamlit apps; it cannot host FastAPI |

> **Important:** Streamlit Community Cloud cannot host the FastAPI service. It only accepts a Streamlit entrypoint. That is why the API goes to Render. If you would rather keep everything in one place, see [Alternative: both on Render](#alternative-both-services-on-render) at the bottom.

**The dashboard is the priority.** If you only have time for one, do that. The API is a differentiator ("KAYA Jarvis can call our endpoints"), not a demo dependency.

---

## Before you start — repo checklist

These are already in the repo. Confirm they survived the push, because each one is a silent failure if missing:

- [ ] `models/` contains **13** `.pkl` files (~13 MB) — **must be committed.** The app loads them; it does not train on deploy.
- [ ] `data/` contains **6** `.csv` files (~4 MB), including `delivery_delays.csv` and `material_wastage.csv` — committed for the same reason.
- [ ] `requirements.txt` — has pinned `scikit-learn==1.5.2`, `xgboost==3.4.0`, `lightgbm==4.7.0`, `mapie>=0.8,<1.0`. **Do not loosen these.** The `.pkl` files were serialised with these versions; a newer scikit-learn or MAPIE makes them fail to unpickle, and the app silently falls back to rule-based mode.
- [ ] `packages.txt` — Debian libraries for WeasyPrint (PDF export).
- [ ] `runtime.txt` — contains `python-3.12`.
- [ ] `.streamlit/config.toml` — theme and `toolbarMode = "minimal"`.

Verify nothing was gitignored by accident:

```bash
git ls-files models data | wc -l
```

Expect **19** files (13 models + 6 datasets). If you get 0, `models/` and `data/` did not get committed — fix that before deploying, or the hosted app will run in fallback mode and every prediction on screen will be a rule-of-thumb rather than a model output.

---

## Part 1 — Dashboard on Streamlit Community Cloud

### 1. Sign in

Go to **https://share.streamlit.io** and sign in with the GitHub account that owns (or can access) the repo. Authorise Streamlit to read your repositories.

### 2. Create the app

Click **Create app** → **Deploy a public app from GitHub**, then fill in:

| Field | Value |
|---|---|
| Repository | `<your-org>/<your-repo>` |
| Branch | `main` |
| Main file path | `app.py` |
| App URL | `nirmanai` (gives `nirmanai.streamlit.app` if free) |

Open **Advanced settings** and set **Python version** to **3.12**.

Leave **Secrets** empty — see [Secrets](#secrets-leave-these-empty) below.

Click **Deploy**.

### 3. Wait for the first build

First build takes **4–8 minutes** — it installs the Debian packages, then xgboost, lightgbm and shap. Watch the log panel on the right.

**It has worked when you see the dashboard load and the masthead shows a green `● Models live` chip plus `AUC 0.797`.**

### 4. Verify it properly

Do not just check that it loads. Check these five things:

1. Masthead chip says **`● Models live`** (green), not `⚠ Heuristic mode` (amber).
   *If amber:* the models did not load. See [Troubleshooting](#troubleshooting).
2. **Risk Radar** shows a populated order table with varied risk percentages, not all-identical values.
3. Sidebar → change **Site location** from Bihar to Gujarat and **Procurement month** to February. Every KPI must change (roughly 9-of-16 at risk → 0-of-16). This is the proof it is running live.
4. **4 · Project Simulator** → **Run simulation**. Should complete in about a second and report an on-time probability near 17% for Bihar/July.
5. **5 · Plan & Report** → **Generate report** → download. Should give a **PDF**. If it gives HTML, `packages.txt` did not apply — the app still works, but re-check that file.

### 5. Resource headroom

Measured peak memory is **264 MB** against Streamlit Cloud's **1 GB** limit, so there is comfortable headroom. You should not hit an OOM.

---

## Part 2 — REST API on Render

### 1. Sign in

Go to **https://render.com**, sign up with GitHub, authorise repo access.

### 2. Create the service

**New +** → **Web Service** → connect your repo. Settings:

| Field | Value |
|---|---|
| Name | `nirmanai-api` |
| Region | Singapore (closest to India) |
| Branch | `main` |
| Runtime | Python 3 |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn api:app --host 0.0.0.0 --port $PORT` |
| Instance type | Free |

Under **Environment**, add:

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.12.0` |

Click **Create Web Service**.

### 3. Verify

Once live, open `https://nirmanai-api.onrender.com/docs`. You should get the interactive Swagger UI.

Test the health endpoint — **`models_loaded` must be `true`**:

```bash
curl https://nirmanai-api.onrender.com/api/v1/health
```

Then a real prediction:

```bash
curl -X POST https://nirmanai-api.onrender.com/api/v1/predict/delay \
  -H 'Content-Type: application/json' \
  -d '{"material_type":"River Sand","supplier_tier":"Tier 3 (Local Supplier)","origin_state":"Rajasthan","destination_state":"Bihar","distance_km":1200,"order_quantity":300,"order_month":7,"past_delay_rate":0.5}'
```

Expect a delay probability around 0.78 with a `Critical` risk label and populated `top_risk_factors`.

### 4. Know the free-tier catch

Render's free tier **sleeps after 15 minutes of inactivity**, and the next request takes **~50 seconds** to wake it. That is fine for a link in a submission form. It is **not** fine for a live demo.

If you are going to show the API on stage, **open `/docs` two minutes before you present** so it is already warm.

---

## Secrets — leave these empty

Deploy with **no API keys**. Every integration degrades to a labelled fallback:

| Key | If absent |
|---|---|
| `GEMINI_API_KEY` | Jarvis routes directly to the simulator, optimiser and supplier database. Still fully functional. |
| `OPENWEATHERMAP_API_KEY` | Uses the IMD-calibrated seasonal profile, labelled as such on screen. |
| `NIRMANAI_API_KEY` | API accepts the default demo key. |

Two reasons not to add them: a public app with a Gemini key attached is an open bill, and the fallbacks are already honest and labelled — nothing looks broken without them.

---

## Demo-day checklist

**Run the demo from a laptop, not from the hosted URL.** Venue wifi is the most common way a good demo dies. The hosted link is for the submission form and for judges reviewing afterwards.

The night before:

- [ ] `git pull` on the demo laptop, then `python setup.py` — confirm it ends with `[OK] All models loaded`.
- [ ] `streamlit run app.py`, walk the full 5-step path once.
- [ ] Pre-run the simulator once so its result is cached.
- [ ] Generate one report so `reports/generated/` exists and is writable.
- [ ] Screenshot or PDF the key screens as an offline fallback.
- [ ] Confirm the hosted URL loads, so the submitted link is not dead.

Optional — hide raw tracebacks during the presentation. Add to `.streamlit/config.toml`:

```toml
[client]
showErrorDetails = false
```

Revert it afterwards so the team can still debug.

---

## Troubleshooting

**Masthead shows `⚠ Heuristic mode` instead of `● Models live`**
The `.pkl` files did not load. Either they were not committed (`git ls-files models | wc -l` should be 13), or a dependency version drifted. Check the build log for `[model_store] Required models unavailable`. Fix by confirming `requirements.txt` still pins scikit-learn, xgboost, lightgbm and `mapie<1.0`.

**Build fails on `weasyprint`**
`packages.txt` was not picked up. It must be at the repo root, not in a subfolder. The app runs without it — reports just export as HTML.

**Build times out or hits memory limits**
Should not happen at 264 MB peak. If it does, the usual cause is pip trying to build a package from source because no wheel matched — check that the Python version is set to **3.12**, not 3.13 or 3.9.

**Report downloads as `.html` instead of `.pdf`**
Expected when WeasyPrint's native libraries are missing. Confirm `packages.txt` is at the root. Not worth blocking the deploy over.

**App is slow on first load**
Cold start loads ~13 MB of models and builds the SHAP explainer: about 2–3 seconds, once per container. Subsequent interactions are cached.

**Streamlit Cloud app went to sleep**
Free apps sleep after about 7 days of no traffic and wake on the next visit. Open the URL once a day in the week before judging.

---

## Alternative: both services on Render

If you would rather not split across two providers, Render can host both. Add a second web service pointing at the same repo:

| Field | Value |
|---|---|
| Name | `nirmanai-dashboard` |
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true` |

**Trade-off:** you lose Streamlit Cloud's always-warm free tier and inherit Render's 15-minute sleep on the *dashboard* too — which is the one thing a judge is most likely to open unannounced. Prefer the split setup unless you have a reason not to.

---

## Summary

| | URL | Notes |
|---|---|---|
| Dashboard | `https://nirmanai.streamlit.app` | The demo. Free, no sleep during active judging. |
| REST API | `https://nirmanai-api.onrender.com/docs` | Warm it up ~2 min before showing it. |
| Repo | `https://github.com/<org>/<repo>` | Models and data committed; no build step needed. |

Put all three in the submission form.
