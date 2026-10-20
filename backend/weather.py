"""
Live data sources for WATTSUN.

- Geocoding: Open-Meteo's free geocoding API (no key needed) turns a place
  name into latitude/longitude.
- Live weather: Open-Meteo's free forecast API (no key needed) gives today's
  actual hourly temperature and cloud cover.
- Solar radiation: NASA POWER's climatology endpoint gives the long-term
  average daily solar radiation (GHI) for that location and month. NASA
  POWER's near-real-time hourly product actually lags several days behind,
  so it can't tell you "today's" radiation directly -- climatology (the
  typical value for this location in this month, averaged over ~20 years of
  satellite data) is the honest way to use NASA POWER as an input to a
  same-day forecast. It's combined with today's live cloud cover above to
  get a same-day estimate.

All three calls are wrapped so that if any of them fail (bad location, no
internet, API hiccup), the caller can fall back to simulated data instead of
crashing the whole prediction.
"""

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

MONTH_ABBR = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

TIMEOUT = 8  # seconds


class WeatherLookupError(Exception):
    """Raised when live data can't be fetched for a location."""


def geocode_location(location: str) -> tuple[float, float]:
    resp = requests.get(GEOCODE_URL, params={"name": location, "count": 1}, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results")
    if not results:
        raise WeatherLookupError(f"Could not find a location matching '{location}'.")
    top = results[0]
    return float(top["latitude"]), float(top["longitude"])


def fetch_hourly_weather(lat: float, lon: float) -> list[dict]:
    """Today's hourly temperature (°C) and cloud cover (%) from Open-Meteo."""
    resp = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,cloudcover",
            "forecast_days": 1,
            "timezone": "auto",
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    hourly = data.get("hourly")
    if not hourly or "time" not in hourly:
        raise WeatherLookupError("Weather API returned no hourly data.")

    result = []
    for i, timestamp in enumerate(hourly["time"]):
        hour = int(timestamp[11:13])
        result.append(
            {
                "hour": hour,
                "cloud_cover": float(hourly["cloudcover"][i]),
                "temperature": float(hourly["temperature_2m"][i]),
            }
        )
    return result


def fetch_monthly_avg_ghi(lat: float, lon: float, month: int) -> float:
    """Long-term average daily solar radiation (kWh/m^2/day) for this
    location in this calendar month, from NASA POWER climatology."""
    resp = requests.get(
        NASA_POWER_URL,
        params={
            "parameters": "ALLSKY_SFC_SW_DWN",
            "community": "RE",
            "longitude": lon,
            "latitude": lat,
            "format": "JSON",
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    try:
        monthly = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    except (KeyError, TypeError):
        raise WeatherLookupError("NASA POWER returned an unexpected response.")

    key = MONTH_ABBR[month - 1]
    value = monthly.get(key)
    if value is None or value < 0:
        raise WeatherLookupError("NASA POWER had no climatology value for this location.")
    return float(value)


def get_live_data(location: str, month: int) -> tuple[list[dict], float]:
    """Fetch hourly weather + monthly GHI for a location. Raises
    WeatherLookupError (or a requests exception) if anything fails --
    callers should catch that and fall back to simulated data."""
    lat, lon = geocode_location(location)
    hourly = fetch_hourly_weather(lat, lon)
    monthly_ghi = fetch_monthly_avg_ghi(lat, lon, month)
    return hourly, monthly_ghi
