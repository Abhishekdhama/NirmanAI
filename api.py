"""
NirmanAI — FastAPI REST Backend
===================================
Production-grade API that KAYA Jarvis can call.
Endpoints for delay prediction, wastage estimation,
procurement optimization, weather intelligence, and report generation.

Run: uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import uuid
import time
from datetime import datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ── Pydantic Schemas ─────────────────────────────────────────

class DelayPredictionRequest(BaseModel):
    """Input for delay risk prediction."""
    material_type: str = Field(..., example="TMT Steel")
    supplier_tier: str = Field(..., example="Tier 2 (Regional Distributor)")
    origin_state: str = Field(..., example="Maharashtra")
    destination_state: str = Field(..., example="Bihar")
    distance_km: int = Field(..., ge=10, le=3000, example=800)
    order_quantity: float = Field(..., gt=0, example=150)
    order_month: int = Field(..., ge=1, le=12, example=7)
    promised_lead_days: int = Field(14, ge=1, le=90)
    past_delay_rate: float = Field(0.35, ge=0.0, le=1.0)
    vehicle_type: str = Field("Truck - Heavy", example="Truck - Heavy")
    temperature: Optional[float] = Field(None, description="Auto-filled from live weather if not provided")
    humidity: Optional[float] = Field(None)
    traffic_status: str = Field("Moderate", example="Moderate")
    waiting_time: int = Field(15, ge=0, le=120)
    inventory_level: int = Field(500, ge=0, le=5000)

class DelayPredictionResponse(BaseModel):
    """Output from delay risk prediction."""
    order_id: str
    delay_probability: float
    is_delayed: bool
    predicted_delay_days: float
    ci_lower: float
    ci_upper: float
    risk_score: int
    risk_label: str
    top_risk_factors: list[str]
    weather_source: str = "simulated"
    timestamp: str

class WastagePredictionRequest(BaseModel):
    """Input for wastage estimation."""
    project_type: str = Field(..., example="Residential Apartment")
    state: str = Field(..., example="Uttar Pradesh")
    project_size_sqft: int = Field(..., ge=100, le=500000, example=15000)
    project_duration_months: int = Field(12, ge=1, le=120)
    month_of_construction: int = Field(..., ge=1, le=12, example=7)
    contractor_experience_yrs: int = Field(8, ge=0, le=50)
    num_workers: int = Field(100, ge=1, le=5000)
    workforce_skill_level: str = Field(..., example="Semi-skilled")
    supervision_quality: str = Field(..., example="Average")
    material_type: str = Field(..., example="OPC Cement")
    blueprint_quantity: float = Field(..., gt=0, example=500)

class WastagePredictionResponse(BaseModel):
    """Output from wastage estimation."""
    predicted_wastage_pct: float
    wastage_range_low: float
    wastage_range_high: float
    actual_qty_estimate: float
    blueprint_quantity: float
    estimated_cost_overrun_inr: float
    wastage_category: str
    risk_factors: list[str]
    timestamp: str

class ProcurementPlanRequest(BaseModel):
    """Input for smart procurement plan."""
    project_type: str = Field(..., example="Residential Apartment")
    state: str = Field(..., example="Bihar")
    project_size_sqft: int = Field(15000, ge=100)
    current_month: int = Field(..., ge=1, le=12, example=7)
    workforce_skill_level: str = Field("Semi-skilled")
    supervision_quality: str = Field("Average")
    contractor_experience_yrs: int = Field(8, ge=0)
    materials: list[dict] = Field(
        ...,
        example=[
            {"material_type": "TMT Steel", "quantity": 80},
            {"material_type": "OPC Cement", "quantity": 500},
            {"material_type": "River Sand", "quantity": 300},
        ]
    )

class ProcurementPlanItem(BaseModel):
    material: str
    order_by: str
    risk_level: str
    delay_probability: str
    blueprint_qty: float
    order_qty_with_buffer: float
    wastage_estimate: str
    top_risk_factor: str

class ProcurementPlanResponse(BaseModel):
    schedule: list[ProcurementPlanItem]
    total_wastage_overrun_inr: float
    recommendation: str
    timestamp: str

class ReportRequest(BaseModel):
    """Input for PDF report generation."""
    project_name: str = Field("Prestige Heights - Phase 2")
    project_type: str = Field("Residential Apartment")
    state: str = Field("Bihar")
    current_month: int = Field(7, ge=1, le=12)

class ReportResponse(BaseModel):
    report_id: str
    status: str
    message: str

class ChatRequest(BaseModel):
    """Input for AI agent conversation."""
    message: str = Field(..., example="My cement supplier just said they'll be 10 days late. What should I do?")
    conversation_id: Optional[str] = Field(None)

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    uptime_seconds: float

# ── App Setup ────────────────────────────────────────────────

START_TIME = time.time()
MODELS = {}
REPORTS_DIR = "reports/generated"

def load_all_models():
    """Load all trained ML models."""
    global MODELS
    try:
        MODELS = {
            "clf_delay":   joblib.load("models/delay_classifier.pkl"),
            "reg_delay":   joblib.load("models/delay_regressor.pkl"),
            "enc_delay":   joblib.load("models/delay_encoders.pkl"),
            "feat_delay":  joblib.load("models/delay_features.pkl"),
            "reg_wast":    joblib.load("models/wastage_regressor.pkl"),
            "reg_wast_lo": joblib.load("models/wastage_regressor_lo.pkl"),
            "reg_wast_hi": joblib.load("models/wastage_regressor_hi.pkl"),
            "enc_wast":    joblib.load("models/wastage_encoders.pkl"),
            "feat_wast":   joblib.load("models/wastage_features.pkl"),
        }
        # Try loading MAPIE conformal model (new CQR), fall back to q_hat
        try:
            MODELS["conformal"] = joblib.load("models/delay_conformal.pkl")
            MODELS["conformal_type"] = "mapie_cqr"
        except FileNotFoundError:
            MODELS["conformal"] = joblib.load("models/delay_q_hat.pkl")
            MODELS["conformal_type"] = "fixed_q_hat"
        return True
    except Exception as e:
        print(f"[WARNING] Could not load models: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    success = load_all_models()
    if success:
        print("[+] All models loaded successfully")
    else:
        print("[!] Running in demo mode — models not available")
    yield
    print("[*] Shutting down NirmanAI API")


app = FastAPI(
    title="NirmanAI API",
    description=(
        "AI-powered supply chain intelligence for Indian construction. "
        "Predicts delivery delays, estimates material wastage, and generates "
        "optimized procurement plans. Built by Team Aim-Nexus, IIT Madras."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow KAYA Jarvis and any frontend to call our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Key Auth (simulated for demo) ────────────────────────

DEMO_API_KEY = "nirmanai-demo-key-2026"

async def verify_api_key(x_api_key: str = Header(default=DEMO_API_KEY)):
    """Simple API key verification. In production, use JWT/OAuth2."""
    if x_api_key != DEMO_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ── Helper Functions ─────────────────────────────────────────

def monsoon_intensity(month: int) -> float:
    profile = {1:0.0, 2:0.0, 3:0.0, 4:0.05, 5:0.15,
               6:0.7, 7:0.9, 8:0.85, 9:0.6, 10:0.2, 11:0.05, 12:0.0}
    return profile.get(month, 0.0)


def _predict_delay_internal(req: DelayPredictionRequest) -> dict:
    """Run delay prediction using loaded models."""
    from train_delay_model import predict_delay

    m_int = monsoon_intensity(req.order_month)

    # Auto-fill weather from live API if not provided
    temperature = req.temperature
    humidity = req.humidity
    weather_source = "user_provided"

    if temperature is None or humidity is None:
        try:
            from weather import get_live_weather
            wx = get_live_weather(req.destination_state)
            temperature = temperature or wx.get("temperature", 30.0)
            humidity = humidity or wx.get("humidity", 65.0)
            weather_source = wx.get("source", "live_api")
        except Exception:
            temperature = temperature or 30.0
            humidity = humidity or 65.0
            weather_source = "fallback"

    inp = {
        "month": req.order_month,
        "day_of_week": 0,
        "quarter": (req.order_month - 1) // 3 + 1,
        "is_festival_period": 0,
        "material_type": req.material_type,
        "supplier_tier": req.supplier_tier,
        "origin_state": req.origin_state,
        "destination_state": req.destination_state,
        "distance_km": req.distance_km,
        "order_quantity": req.order_quantity,
        "promised_lead_days": req.promised_lead_days,
        "monsoon_intensity": m_int,
        "monsoon_sensitivity": 0.5,
        "dest_logistics_score": 0.6,
        "orig_logistics_score": 0.7,
        "dest_monsoon_severity": 0.65,
        "supplier_reliability": 1 - req.past_delay_rate * 0.8,
        "past_delay_rate": req.past_delay_rate,
        "vehicle_type": req.vehicle_type,
        "temperature": temperature,
        "humidity": humidity,
        "traffic_status": req.traffic_status,
        "waiting_time": req.waiting_time,
        "inventory_level": req.inventory_level,
        "asset_utilization": 80.0,
        "demand_forecast": 400,
        "order_value_inr": req.order_quantity * 1500,
        "road_quality": 0.65,
        "supplier_capacity": 80,
        "fuel_price_index": 105.0,
        "driver_experience": 10,
    }

    conformal = MODELS.get("conformal")
    conformal_type = MODELS.get("conformal_type", "fixed_q_hat")

    result = predict_delay(
        MODELS["clf_delay"], MODELS["reg_delay"], conformal,
        MODELS["enc_delay"], MODELS["feat_delay"], inp
    )
    result["weather_source"] = weather_source
    return result


def _predict_wastage_internal(req: WastagePredictionRequest) -> dict:
    """Run wastage prediction using loaded models."""
    from train_wastage_model import predict_wastage

    m_int = monsoon_intensity(req.month_of_construction)

    wast_inp = {
        "project_type": req.project_type,
        "state": req.state,
        "project_size_sqft": req.project_size_sqft,
        "project_duration_months": req.project_duration_months,
        "month_of_construction": req.month_of_construction,
        "contractor_experience_yrs": req.contractor_experience_yrs,
        "num_workers": req.num_workers,
        "workforce_skill_level": req.workforce_skill_level,
        "supervision_quality": req.supervision_quality,
        "material_type": req.material_type,
        "blueprint_quantity": req.blueprint_quantity,
        "logistics_score": 0.6,
        "monsoon_intensity": m_int,
        "monsoon_sensitivity": 0.5,
    }

    return predict_wastage(
        MODELS["reg_wast"], MODELS["reg_wast_lo"], MODELS["reg_wast_hi"],
        MODELS["enc_wast"], MODELS["feat_wast"], wast_inp
    )


# ── Endpoints ────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        models_loaded=bool(MODELS),
        version="2.0.0",
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/api/v1/predict/delay",
          response_model=DelayPredictionResponse,
          tags=["Predictions"],
          summary="Predict delivery delay risk",
          description="Predicts delay probability, magnitude, and confidence intervals for a construction material order.")
async def predict_delay_endpoint(
    req: DelayPredictionRequest,
    api_key: str = Depends(verify_api_key),
):
    if not MODELS:
        raise HTTPException(503, "Models not loaded. Run setup.py first.")

    try:
        result = _predict_delay_internal(req)
        return DelayPredictionResponse(
            order_id=f"ORD-{uuid.uuid4().hex[:8].upper()}",
            delay_probability=result["delay_probability"],
            is_delayed=result["is_delayed"],
            predicted_delay_days=result["predicted_delay_days"],
            ci_lower=result["ci_lower"],
            ci_upper=result["ci_upper"],
            risk_score=result["risk_score"],
            risk_label=result["risk_label"],
            top_risk_factors=result["top_risk_factors"],
            weather_source=result.get("weather_source", "simulated"),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@app.post("/api/v1/predict/wastage",
          response_model=WastagePredictionResponse,
          tags=["Predictions"],
          summary="Estimate material wastage",
          description="Estimates wastage percentage, cost overrun, and risk factors for a construction material.")
async def predict_wastage_endpoint(
    req: WastagePredictionRequest,
    api_key: str = Depends(verify_api_key),
):
    if not MODELS:
        raise HTTPException(503, "Models not loaded. Run setup.py first.")

    try:
        result = _predict_wastage_internal(req)
        return WastagePredictionResponse(
            predicted_wastage_pct=result["predicted_wastage_pct"],
            wastage_range_low=result["wastage_range_low"],
            wastage_range_high=result["wastage_range_high"],
            actual_qty_estimate=result["actual_qty_estimate"],
            blueprint_quantity=result["blueprint_quantity"],
            estimated_cost_overrun_inr=result["estimated_cost_overrun_inr"],
            wastage_category=result["wastage_category"],
            risk_factors=result["risk_factors"],
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(500, f"Wastage prediction failed: {str(e)}")


@app.post("/api/v1/optimize/procurement",
          response_model=ProcurementPlanResponse,
          tags=["Optimization"],
          summary="Generate smart procurement plan",
          description="Creates an AI-optimized, week-by-week procurement schedule with wastage buffers.")
async def optimize_procurement(
    req: ProcurementPlanRequest,
    api_key: str = Depends(verify_api_key),
):
    if not MODELS:
        raise HTTPException(503, "Models not loaded. Run setup.py first.")

    try:
        from train_delay_model import predict_delay
        from train_wastage_model import predict_wastage

        m_int = monsoon_intensity(req.current_month)
        schedule = []
        total_overrun = 0.0

        for item in req.materials:
            mat = item["material_type"]
            qty = item["quantity"]

            delay_inp = DelayPredictionRequest(
                material_type=mat,
                supplier_tier="Tier 2 (Regional Distributor)",
                origin_state="Maharashtra",
                destination_state=req.state,
                distance_km=800,
                order_quantity=qty,
                order_month=req.current_month,
                past_delay_rate=0.35,
            )
            delay_res = _predict_delay_internal(delay_inp)

            wast_inp = WastagePredictionRequest(
                project_type=req.project_type,
                state=req.state,
                project_size_sqft=req.project_size_sqft,
                month_of_construction=req.current_month,
                contractor_experience_yrs=req.contractor_experience_yrs,
                workforce_skill_level=req.workforce_skill_level,
                supervision_quality=req.supervision_quality,
                material_type=mat,
                blueprint_quantity=qty,
            )
            wast_res = _predict_wastage_internal(wast_inp)

            risk_level = delay_res["risk_label"]
            order_week = {
                "Critical": "Week 1", "High": "Week 1",
                "Medium": "Week 2", "Low": "Week 3"
            }.get(risk_level, "Week 2")

            total_overrun += wast_res["estimated_cost_overrun_inr"]

            schedule.append(ProcurementPlanItem(
                material=mat,
                order_by=order_week,
                risk_level=risk_level,
                delay_probability=f"{delay_res['delay_probability']:.0%}",
                blueprint_qty=qty,
                order_qty_with_buffer=wast_res["actual_qty_estimate"],
                wastage_estimate=f"{wast_res['predicted_wastage_pct']:.1f}%",
                top_risk_factor=delay_res["top_risk_factors"][0] if delay_res["top_risk_factors"] else "Standard risk",
            ))

        # Sort by urgency
        order_priority = {"Week 1": 0, "Week 2": 1, "Week 3": 2}
        schedule.sort(key=lambda x: order_priority.get(x.order_by, 9))

        return ProcurementPlanResponse(
            schedule=schedule,
            total_wastage_overrun_inr=round(total_overrun, 0),
            recommendation=(
                f"Order high-risk materials immediately to prevent delays. "
                f"Total projected wastage overrun: INR {total_overrun:,.0f}. "
                f"Already included as buffer in recommended quantities."
            ),
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(500, f"Procurement optimization failed: {str(e)}")


@app.get("/api/v1/weather/{state}",
         tags=["Weather Intelligence"],
         summary="Get live weather for a state",
         description="Returns real-time weather data and construction risk assessment for an Indian state.")
async def get_weather(state: str):
    try:
        from weather import get_live_weather, get_weather_risk_summary
        weather = get_live_weather(state)
        risk = get_weather_risk_summary(state)
        return {
            "state": state,
            "weather": weather,
            "risk": risk,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(500, f"Weather fetch failed: {str(e)}")


@app.post("/api/v1/report/generate",
          response_model=ReportResponse,
          tags=["Reports"],
          summary="Generate PDF risk report (async)",
          description="Triggers background generation of a branded PDF risk report. Use the report_id to download.")
async def generate_report(
    req: ReportRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key),
):
    report_id = f"RPT-{uuid.uuid4().hex[:8].upper()}"

    def _generate(rid, request):
        try:
            from report_generator import generate_pdf_report
            generate_pdf_report(
                report_id=rid,
                project_name=request.project_name,
                project_type=request.project_type,
                state=request.state,
                current_month=request.current_month,
                output_dir=REPORTS_DIR,
                models=MODELS,
            )
        except Exception as e:
            print(f"[ERROR] Report generation failed: {e}")

    background_tasks.add_task(_generate, report_id, req)

    return ReportResponse(
        report_id=report_id,
        status="generating",
        message=f"Report {report_id} is being generated. Use GET /api/v1/report/{report_id} to download.",
    )


@app.get("/api/v1/report/{report_id}",
         tags=["Reports"],
         summary="Download generated PDF report")
async def download_report(report_id: str):
    filepath = os.path.join(REPORTS_DIR, f"{report_id}.pdf")
    if os.path.exists(filepath):
        return FileResponse(
            filepath,
            media_type="application/pdf",
            filename=f"NirmanAI_{report_id}.pdf",
        )
    # Check if still generating
    return JSONResponse(
        status_code=202,
        content={"status": "generating", "message": "Report is still being generated. Try again in a few seconds."},
    )


@app.get("/api/v1/suppliers/{material_type}",
         tags=["Supplier Intelligence"],
         summary="Find suppliers for a material",
         description="Search the supplier database for available suppliers of a specific material type.")
async def find_suppliers_endpoint(
    material_type: str,
    destination_state: Optional[str] = None,
    min_reliability: float = 0.0,
    top_n: int = 5,
):
    try:
        from suppliers_db import find_suppliers, find_alternate_suppliers
        if destination_state:
            results = find_suppliers(material_type, destination_state, min_reliability)[:top_n]
        else:
            results = find_alternate_suppliers(material_type, top_n=top_n)
        return {
            "material_type": material_type,
            "suppliers": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(500, f"Supplier search failed: {str(e)}")


@app.post("/api/v1/agent/chat",
          tags=["AI Agent"],
          summary="Chat with KAYA Jarvis",
          description="Send natural language messages to the AI procurement assistant.")
async def chat_with_agent(
    req: ChatRequest,
    api_key: str = Depends(verify_api_key),
):
    try:
        from agent import process_agent_message
        return process_agent_message(req.message, req.conversation_id)
    except Exception as e:
        raise HTTPException(500, f"Agent interaction failed: {str(e)}")


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("  NirmanAI API v2.0 — Starting...")
    print("  Swagger Docs: http://localhost:8000/docs")
    print("=" * 55 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
