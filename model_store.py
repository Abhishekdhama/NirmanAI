"""
NirmanAI — Model Store
======================
Single, robust entry point for loading the trained models.

Why this exists: the dashboard, the API and the simulation engine all need the
same bundle, and every one of them previously had its own slightly different
loader. Two of them silently fell back to fake "demo mode" whenever an optional
dependency (MAPIE) had a version mismatch — which made the whole product look
like a mockup even though the models were sitting right there on disk.

Loading rules:
  * The five core artefacts per model family are REQUIRED. Missing any of them
    means the models genuinely have not been trained yet.
  * The conformal interval estimator is OPTIONAL and degrades gracefully:
    adaptive MAPIE CQR -> split-conformal q_hat -> a conservative constant.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import joblib

# MAPIE logs "The predictions are ill-sorted." on every CQR call where the
# quantile heads cross. It is expected (MAPIE re-sorts them internally) and
# harmless, but it goes to the ROOT logger and floods the console during a live
# demo. Raising the level alone is not enough — anything that later calls
# logging.basicConfig() undoes it — so drop the record by content.
class _DropIllSorted(logging.Filter):
    def filter(self, record):
        return "ill-sorted" not in record.getMessage()


logging.getLogger().addFilter(_DropIllSorted())
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_DropIllSorted())
logging.getLogger("mapie").setLevel(logging.WARNING)

MODELS_DIR = "models"

REQUIRED = {
    "clf_delay": "delay_classifier.pkl",
    "reg_delay": "delay_regressor.pkl",
    "enc_delay": "delay_encoders.pkl",
    "feat_delay": "delay_features.pkl",
    "reg_wast": "wastage_regressor.pkl",
    "reg_wast_lo": "wastage_regressor_lo.pkl",
    "reg_wast_hi": "wastage_regressor_hi.pkl",
    "enc_wast": "wastage_encoders.pkl",
    "feat_wast": "wastage_features.pkl",
}

# Used when neither conformal artefact can be loaded. Deliberately wide: an
# over-wide interval is honest, a missing one is not.
FALLBACK_Q_HAT = 5.0


def _path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def load_models(models_dir: Optional[str] = None) -> Optional[dict]:
    """
    Load the full model bundle.

    Returns a dict on success, or None if the required artefacts are absent
    (i.e. `python setup.py` has not been run). Never raises.
    """
    global MODELS_DIR
    if models_dir:
        MODELS_DIR = models_dir

    bundle: dict = {}
    try:
        for key, filename in REQUIRED.items():
            bundle[key] = joblib.load(_path(filename))
    except Exception as exc:
        bundle_error = f"{type(exc).__name__}: {exc}"
        print(f"[model_store] Required models unavailable — {bundle_error}")
        return None

    conformal, conformal_type = _load_conformal()
    bundle["conformal"] = conformal
    bundle["conformal_type"] = conformal_type

    # Training metrics, surfaced in the UI so the accuracy numbers on screen are
    # the ones this build actually produced rather than copy typed into a slide.
    try:
        bundle["metrics"] = joblib.load(_path("delay_metrics.pkl"))
    except Exception:
        bundle["metrics"] = {}

    # Per-prediction SHAP explainer. Optional — predictions still work without
    # it, they just fall back to rule-based risk factors.
    try:
        from train_delay_model import build_delay_explainer
        bundle["explainer"] = build_delay_explainer(bundle["clf_delay"])
    except Exception:
        bundle["explainer"] = None

    return bundle


def _load_conformal():
    """
    Load the best conformal interval estimator that this environment can
    actually use. Every step is independently guarded because the MAPIE pickle
    only unpickles under mapie<1.0 — a newer install raises AttributeError, not
    FileNotFoundError, which is exactly what used to kill the whole bundle.
    """
    try:
        estimator = joblib.load(_path("delay_conformal.pkl"))
        if estimator is not None and hasattr(estimator, "predict"):
            return estimator, "mapie_cqr"
    except Exception:
        pass

    try:
        q_hat = joblib.load(_path("delay_q_hat.pkl"))
        return float(q_hat), "split_conformal"
    except Exception:
        pass

    print("[model_store] No conformal artefact found — using a conservative "
          f"+/-{FALLBACK_Q_HAT:.0f} day interval.")
    return FALLBACK_Q_HAT, "fallback_constant"


def conformal_label(conformal_type: str) -> str:
    """Human-readable description of the interval method, for the UI."""
    return {
        "mapie_cqr": "Adaptive conformalised quantile regression (90% coverage)",
        "split_conformal": "Split conformal prediction (90% coverage)",
        "fallback_constant": "Fixed conservative interval (uncalibrated)",
    }.get(conformal_type, "Conformal prediction")
