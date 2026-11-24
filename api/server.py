"""WheelValue FastAPI server — JSON endpoints + static frontend."""
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import DATA, MODELS, validate_car_input, build_inference_dataframe
from src.explain import explain_single_prediction

FRONTEND = ROOT / "frontend"

app = FastAPI(title="WheelValue", version="1.0.0")

# ---------------------------------------------------------------------------
# Startup: load all heavy assets once
# ---------------------------------------------------------------------------
_pipeline = joblib.load(MODELS / "wheelvalue_pipeline.joblib")
_metadata: dict = json.loads((MODELS / "ui_metadata.json").read_text())
_metrics: dict = json.loads((MODELS / "model_metrics.json").read_text())
_feature_importance: dict = json.loads((MODELS / "feature_importance.json").read_text())
_train_df: pd.DataFrame = pd.read_csv(DATA / "train.csv")

# ---------------------------------------------------------------------------
# Static assets — mount AFTER route declarations so / is not shadowed
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
def index():
    """Serve the SPA shell."""
    return FileResponse(FRONTEND / "index.html", media_type="text/html")


@app.get("/api/metadata")
def metadata():
    """Return form dropdowns, numeric bounds, model metrics, and feature importance."""
    return {
        "categorical_options": _metadata["categorical_options"],
        "numeric_ranges": _metadata["numeric_ranges"],
        "model_metrics": _metrics,
        "feature_importance": _feature_importance,
    }


@app.post("/api/predict")
def predict(payload: dict[str, Any]):
    """Validate, predict, and explain a single car's resale price."""
    valid, errors = validate_car_input(payload, _metadata)
    if not valid:
        raise HTTPException(status_code=422, detail={"errors": errors})

    df = build_inference_dataframe(payload)
    # TransformedTargetRegressor already applies expm1 — predict() returns raw INR
    price = float(_pipeline.predict(df)[0])

    validation_mae = _metrics["validation_mae"]
    car_age = int(df["car_age"].iloc[0])

    # SHAP local explanation
    effects = explain_single_prediction(_pipeline, df)
    shap_contributions = [
        {
            "feature": row["feature"],
            "label": row["label"],
            "impact": float(row["contribution"]),
            "direction": "positive" if row["contribution"] >= 0 else "negative",
        }
        for _, row in effects.iterrows()
    ]

    submitted_details = {
        "brand": payload.get("brand"),
        "year": int(float(payload.get("year", 0))),
        "car_age": car_age,
        "km_driven": int(float(payload.get("km_driven", 0))),
        "fuel": payload.get("fuel"),
        "transmission": payload.get("transmission"),
        "owner": payload.get("owner"),
        "seller_type": payload.get("seller_type"),
        "mileage": float(payload.get("mileage", 0)),
        "engine": float(payload.get("engine", 0)),
        "max_power": float(payload.get("max_power", 0)),
        "seats": int(float(payload.get("seats", 0))),
    }

    return {
        "predicted_price_inr": price,
        "error_range_low": max(0.0, price - validation_mae),
        "error_range_high": price + validation_mae,
        "validation_mae": validation_mae,
        "car_age": car_age,
        "submitted_details": submitted_details,
        "shap_contributions": shap_contributions,
    }


@app.get("/api/analytics")
def analytics(
    fuel: str = Query(default="All"),
    transmission: str = Query(default="All"),
):
    """Return aggregated training-data chart data, optionally filtered."""
    df = _train_df.copy()
    if fuel != "All":
        df = df[df["fuel"] == fuel]
    if transmission != "All":
        df = df[df["transmission"] == transmission]

    record_count = len(df)

    # Top 15 brands by average price (INR Lakhs)
    brand_avg = (
        df.groupby("brand")["selling_price"]
        .mean()
        .nlargest(15)
        .reset_index()
        .rename(columns={"selling_price": "avg_price"})
    )
    brand_avg_prices = [
        {"brand": r["brand"], "avg_price_lakhs": round(r["avg_price"] / 1e5, 2)}
        for _, r in brand_avg.iterrows()
    ]

    # Average price by fuel type (INR Lakhs)
    fuel_avg = (
        df.groupby("fuel")["selling_price"]
        .mean()
        .reset_index()
        .rename(columns={"selling_price": "avg_price"})
    )
    fuel_avg_prices = [
        {"fuel": r["fuel"], "avg_price_lakhs": round(r["avg_price"] / 1e5, 2)}
        for _, r in fuel_avg.iterrows()
    ]

    # Year price trend (INR Lakhs)
    year_avg = (
        df.groupby("year")["selling_price"]
        .mean()
        .reset_index()
        .sort_values("year")
        .rename(columns={"selling_price": "avg_price"})
    )
    year_price_trend = [
        {"year": int(r["year"]), "avg_price_lakhs": round(r["avg_price"] / 1e5, 2)}
        for _, r in year_avg.iterrows()
    ]

    return {
        "brand_avg_prices": brand_avg_prices,
        "fuel_avg_prices": fuel_avg_prices,
        "year_price_trend": year_price_trend,
        "record_count": record_count,
    }


# Mount static files for CSS and JS (must come after route definitions)
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")
