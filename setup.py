"""
NirmanAI — one-command setup

    python setup.py

Installs dependencies, generates the synthetic datasets, trains both model
families, and verifies that the artefacts actually load. Then:

    streamlit run app.py
"""

import subprocess
import sys

STEPS = 5


def run(cmd, description, step):
    print(f"\n[{step}/{STEPS}] {description}")
    print(f"      $ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n[FAILED] {description}")
        print("         Fix the error above and re-run `python setup.py`.")
        sys.exit(1)


print("\n" + "=" * 62)
print("  NirmanAI — Setup & Training Pipeline")
print("  Team Aim-Nexus | IIT Madras")
print("=" * 62)

if sys.version_info < (3, 9):
    print(f"\n[FAILED] Python {sys.version_info.major}.{sys.version_info.minor} "
          "is too old. NirmanAI needs Python 3.9 or newer.")
    sys.exit(1)

run(f'"{sys.executable}" -m pip install -q -r requirements.txt',
    "Installing dependencies", 1)
run(f'"{sys.executable}" generate_data.py',
    "Generating synthetic datasets", 2)
run(f'"{sys.executable}" train_delay_model.py',
    "Training the delay prediction models", 3)
run(f'"{sys.executable}" train_wastage_model.py',
    "Training the wastage estimation models", 4)

# Verifying the bundle here is the difference between finding a problem now and
# finding it on stage: a version mismatch can leave the artefacts on disk but
# unloadable, and the dashboard would silently drop to its fallback mode.
print(f"\n[{STEPS}/{STEPS}] Verifying the trained models load")
try:
    from model_store import load_models, conformal_label

    bundle = load_models()
    if bundle is None:
        raise RuntimeError("model_store could not load the required artefacts")

    metrics = bundle.get("metrics", {})
    print("      [OK] All models loaded")
    print(f"      Delay classifier AUC : {metrics.get('auc', 0):.3f} "
          f"(ceiling {metrics.get('bayes_ceiling_auc', 0):.3f})")
    print(f"      Interval method      : {conformal_label(bundle['conformal_type'])}")
    print(f"      SHAP explainer       : "
          f"{'ready' if bundle.get('explainer') is not None else 'unavailable'}")
except Exception as exc:
    print(f"      [FAILED] {type(exc).__name__}: {exc}")
    print("      The dashboard will still start, but in rule-based fallback mode.")
    sys.exit(1)

print("\n" + "=" * 62)
print("  Setup complete.")
print("=" * 62)
print("\n  Launch the dashboard:   streamlit run app.py")
print("  Launch the REST API:    uvicorn api:app --reload --port 8000")
print("\n  Dashboard opens at http://localhost:8501")
print("=" * 62 + "\n")
