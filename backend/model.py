"""
WATTSUN prediction model.

For this demo build, there's no live NASA POWER / Weather API key wired up yet,
so `simulate_weather()` stands in for that data source. It generates a
plausible hourly cloud-cover and temperature profile for a given location and
date (deterministic per location+date, so refreshing doesn't change the
answer). Swap `simulate_weather()` for real API calls later --
`predict_day()` and everything downstream doesn't need to change.

The model itself IS real: a scikit-learn RandomForestRegressor trained on a
synthetic-but-physically-shaped dataset (solar output rises and falls with
the sun's daily arc, and is suppressed by cloud cover). Replace
`generate_training_data()` with real historical readings when you have them,
and the rest of the pipeline (train/predict/save/load) stays the same.
"""

import hashlib
import math
from datetime import date as date_cls

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

MODEL_PATH = "solar_model.joblib"

# ---------------------------------------------------------------------------
# 1. Synthetic training data (stand-in for the Kaggle historical dataset)
# ---------------------------------------------------------------------------

def _solar_curve(hour: float) -> float:
    """Bell-shaped daylight curve, 0 outside ~6am-6pm, peak at noon."""
    if hour < 6 or hour > 18:
        return 0.0
    return math.sin(math.pi * (hour - 6) / 12) ** 1.3


REFERENCE_GHI = 5.5  # kWh/m^2/day, a rough global-average baseline


def generate_training_data(n_days: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_days):
        cloud_base = rng.uniform(0, 90)
        temp_base = rng.uniform(18, 40)
        system_size = rng.choice([2, 3, 5, 7, 10])
        # monthly average solar radiation for this simulated location/month,
        # standing in for what NASA POWER climatology provides in production
        monthly_avg_ghi = rng.uniform(3.0, 7.5)
        for hour in range(24):
            cloud = min(100, max(0, cloud_base + rng.normal(0, 12)))
            temp = temp_base + rng.normal(0, 2) - 0.15 * abs(hour - 13)
            ghi_factor = monthly_avg_ghi / REFERENCE_GHI
            base = _solar_curve(hour) * system_size * ghi_factor
            cloud_factor = 1 - (cloud / 100) * 0.75
            # slight heat de-rating above 35C, typical of PV panels
            heat_derate = 1 - max(0, temp - 35) * 0.004
            noise = rng.normal(0, 0.05)
            generation_kw = max(0, base * cloud_factor * heat_derate + noise)
            rows.append(
                dict(
                    hour=hour,
                    cloud_cover=cloud,
                    temperature=temp,
                    system_size_kw=system_size,
                    monthly_avg_ghi=monthly_avg_ghi,
                    generation_kw=generation_kw,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Train / load
# ---------------------------------------------------------------------------

FEATURES = ["hour", "cloud_cover", "temperature", "system_size_kw", "monthly_avg_ghi"]


def train_model() -> RandomForestRegressor:
    df = generate_training_data()
    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(df[FEATURES], df["generation_kw"])
    joblib.dump(model, MODEL_PATH)
    return model


def load_or_train_model() -> RandomForestRegressor:
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return train_model()


# ---------------------------------------------------------------------------
# 3. Simulated weather (replace with NASA POWER + Weather API later)
# ---------------------------------------------------------------------------

def simulate_weather(location: str, on_date: date_cls) -> tuple[list[dict], float]:
    """Deterministic-per-day synthetic hourly weather for a location, used
    as a fallback when live APIs are unreachable. Returns (hourly, monthly_avg_ghi)."""
    seed_str = f"{location.lower().strip()}-{on_date.isoformat()}"
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    cloud_base = rng.uniform(5, 70)
    temp_base = rng.uniform(20, 38)
    monthly_avg_ghi = float(rng.uniform(3.5, 6.5))

    hourly = []
    for hour in range(24):
        cloud = float(min(100, max(0, cloud_base + rng.normal(0, 10))))
        temp = float(temp_base + rng.normal(0, 1.5) - 0.15 * abs(hour - 13))
        hourly.append({"hour": hour, "cloud_cover": round(cloud, 1), "temperature": round(temp, 1)})
    return hourly, monthly_avg_ghi


# ---------------------------------------------------------------------------
# 4. Day-ahead prediction
# ---------------------------------------------------------------------------

def predict_day(model, hourly_weather: list[dict], monthly_avg_ghi: float, system_size_kw: float) -> list[dict]:
    """Run the model over a day's hourly weather. `hourly_weather` is a list
    of {hour, cloud_cover, temperature} dicts -- either fetched live
    (weather.py) or produced by simulate_weather() as a fallback."""
    df = pd.DataFrame(hourly_weather)
    df["system_size_kw"] = system_size_kw
    df["monthly_avg_ghi"] = monthly_avg_ghi
    preds = model.predict(df[FEATURES])
    result = []
    for row, pred in zip(hourly_weather, preds):
        result.append(
            {
                "hour": row["hour"],
                "cloud_cover": row["cloud_cover"],
                "temperature": row["temperature"],
                "predicted_kw": round(max(0.0, float(pred)), 3),
            }
        )
    return result
