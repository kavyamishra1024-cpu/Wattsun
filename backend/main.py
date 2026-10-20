from datetime import date

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from model import load_or_train_model, predict_day, simulate_weather
from recommend import (
    estimate_co2_reduction_kg,
    estimate_savings,
    estimate_surplus_kwh,
    find_alerts,
    recommend_appliances,
    solar_health_score,
    total_expected_consumption_kwh,
)
from weather import WeatherLookupError, get_live_data

app = FastAPI(title="WATTSUN API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only -- lock this down for production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = load_or_train_model()


class Appliance(BaseModel):
    name: str
    watts: float = Field(gt=0)
    hours_per_day: float = Field(default=1.5, gt=0, le=24)


class PredictRequest(BaseModel):
    location: str
    system_size_kw: float = Field(gt=0)
    appliances: list[Appliance]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(req: PredictRequest):
    today = date.today()

    try:
        hourly_weather, monthly_avg_ghi = get_live_data(req.location, today.month)
        data_source = "live"
    except Exception:
        # Covers WeatherLookupError, network failures, timeouts, bad
        # responses -- the demo keeps working on simulated data instead of
        # crashing the whole request.
        hourly_weather, monthly_avg_ghi = simulate_weather(req.location, today)
        data_source = "simulated (live lookup failed -- check location spelling or internet connection)"

    hourly = predict_day(MODEL, hourly_weather, monthly_avg_ghi, req.system_size_kw)

    peak_kw = max(h["predicted_kw"] for h in hourly)
    total_kwh = round(sum(h["predicted_kw"] for h in hourly), 2)

    appliance_dicts = [a.model_dump() for a in req.appliances]
    recommendations = recommend_appliances(hourly, appliance_dicts)
    alerts = find_alerts(hourly, peak_kw)
    health_score = solar_health_score(hourly, req.system_size_kw)
    savings = estimate_savings(recommendations)
    co2_reduction_kg = estimate_co2_reduction_kg(recommendations)
    expected_consumption_kwh = total_expected_consumption_kwh(appliance_dicts)
    surplus_kwh = estimate_surplus_kwh(hourly, recommendations)

    return {
        "location": req.location,
        "date": today.isoformat(),
        "data_source": data_source,
        "monthly_avg_ghi": round(monthly_avg_ghi, 2),
        "hourly": hourly,
        "total_predicted_kwh": total_kwh,
        "peak_kw": round(peak_kw, 2),
        "recommendations": recommendations,
        "alerts": alerts,
        "solar_health_score": health_score,
        "estimated_savings": savings,
        "co2_reduction_kg": co2_reduction_kg,
        "expected_consumption_kwh": expected_consumption_kwh,
        "surplus_kwh": surplus_kwh,
    }
