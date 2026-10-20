"""
Recommendation engine.

Takes the hourly generation prediction from model.py and a list of
appliances, and works out: the best hour(s) to run each appliance, low
generation alerts, an overall solar health score, and a rough savings
estimate versus running everything on grid power.
"""

GRID_RATE_PER_KWH = 8.0  # currency units per kWh, adjust to your local tariff
GRID_EMISSION_FACTOR_KG_PER_KWH = 0.82  # approx. average for India's grid mix
DEFAULT_RUN_HOURS = 1.5  # fallback if an appliance doesn't specify hours_per_day


def recommend_appliances(hourly: list[dict], appliances: list[dict]) -> list[dict]:
    """Assign each appliance a start hour, preferring high-generation hours
    but staggering appliances that would collectively exceed what's
    generated at that hour -- so two appliances don't both get told to run
    at the exact same peak hour if that would overload the system.

    Each appliance dict may include "hours_per_day" (defaults to
    DEFAULT_RUN_HOURS if not given).
    """
    recommendations = []
    sorted_hours = sorted(hourly, key=lambda h: h["predicted_kw"], reverse=True)
    committed_kw = {h["hour"]: 0.0 for h in hourly}  # load already assigned per hour

    for appliance in appliances:
        watts = appliance["watts"]
        kw_needed = watts / 1000
        run_hours = appliance.get("hours_per_day") or DEFAULT_RUN_HOURS

        best = None
        for h in sorted_hours:
            available = h["predicted_kw"] - committed_kw[h["hour"]]
            if available >= kw_needed:
                best = h
                break

        if best is None:
            # nothing has enough spare capacity -- fall back to the single
            # best hour by raw generation, flagged as not fully covered
            best = sorted_hours[0]
            covered = False
        else:
            covered = True
            committed_kw[best["hour"]] += kw_needed

        recommendations.append(
            {
                "appliance": appliance["name"],
                "watts": watts,
                "hours_per_day": run_hours,
                "best_hour": best["hour"],
                "end_hour": best["hour"] + run_hours,
                "predicted_kw_at_hour": round(best["predicted_kw"], 2),
                "fully_solar_covered": covered,
            }
        )

    # annotate appliances that share an hour with a note about combined load,
    # so it reads as an intentional optimization rather than a coincidence
    by_hour: dict[int, list[dict]] = {}
    for r in recommendations:
        by_hour.setdefault(r["best_hour"], []).append(r)

    for hour, group in by_hour.items():
        if len(group) > 1:
            combined_kw = round(sum(r["watts"] / 1000 for r in group), 2)
            available_kw = next((h["predicted_kw"] for h in hourly if h["hour"] == hour), 0)
            for r in group:
                r["shared_hour_note"] = (
                    f"Combined load {combined_kw} kW fits within {round(available_kw, 2)} kW available"
                    if combined_kw <= available_kw
                    else f"Combined load {combined_kw} kW exceeds {round(available_kw, 2)} kW available at this hour"
                )

    return recommendations


def total_expected_consumption_kwh(appliances: list[dict]) -> float:
    """Rough estimate of today's total appliance energy use, using each
    appliance's own hours_per_day (or the default)."""
    return round(
        sum(
            a["watts"] / 1000 * (a.get("hours_per_day") or DEFAULT_RUN_HOURS)
            for a in appliances
        ),
        2,
    )


def find_alerts(hourly: list[dict], peak_kw: float) -> list[dict]:
    """Flag daytime hours (6am-6pm) where predicted generation drops below
    20% of today's peak -- i.e. a low-generation warning window."""
    threshold = max(0.05, peak_kw * 0.2)
    alerts = []
    in_dip = False
    dip_start = None

    for h in hourly:
        if 6 <= h["hour"] <= 18:
            low = h["predicted_kw"] < threshold
            if low and not in_dip:
                in_dip = True
                dip_start = h["hour"]
            elif not low and in_dip:
                end = h["hour"] if h["hour"] > dip_start else dip_start + 1
                alerts.append({"start_hour": dip_start, "end_hour": end})
                in_dip = False
    if in_dip:
        alerts.append({"start_hour": dip_start, "end_hour": max(19, dip_start + 1)})

    return alerts


def solar_health_score(hourly: list[dict], system_size_kw: float) -> int:
    """Compare today's predicted total generation against a theoretical
    max of ~5 peak-sun-hours for the system size."""
    total_kwh = sum(h["predicted_kw"] for h in hourly)
    theoretical_max = system_size_kw * 5.0
    score = min(100, round((total_kwh / theoretical_max) * 100)) if theoretical_max else 0
    return max(0, score)


def estimate_savings(recommendations: list[dict]) -> float:
    """Rough estimate: for every appliance-hour fully covered by solar,
    that's grid electricity you didn't have to buy."""
    kwh_saved = sum(
        r["watts"] / 1000 * r["hours_per_day"] for r in recommendations if r["fully_solar_covered"]
    )
    return round(kwh_saved * GRID_RATE_PER_KWH, 2)


def estimate_co2_reduction_kg(recommendations: list[dict]) -> float:
    """CO2 avoided by running solar-covered appliances instead of drawing
    that energy from the grid, using an approximate grid emission factor."""
    kwh_saved = sum(
        r["watts"] / 1000 * r["hours_per_day"] for r in recommendations if r["fully_solar_covered"]
    )
    return round(kwh_saved * GRID_EMISSION_FACTOR_KG_PER_KWH, 2)


def estimate_surplus_kwh(hourly: list[dict], recommendations: list[dict]) -> float:
    """Solar generated today but not used by any scheduled appliance --
    exportable to the grid or available for battery storage."""
    total_generated = sum(h["predicted_kw"] for h in hourly)
    solar_used = sum(
        r["watts"] / 1000 * r["hours_per_day"] for r in recommendations if r["fully_solar_covered"]
    )
    return round(max(0.0, total_generated - solar_used), 2)
