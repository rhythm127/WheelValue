"""FastAPI endpoint and static-serving tests."""
import pytest
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)

VALID_CAR = {
    "brand": "Maruti",
    "year": 2014,
    "km_driven": 70000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": 20.0,
    "engine": 1248.0,
    "max_power": 80.0,
    "seats": 5,
}


# ---------------------------------------------------------------------------
# Static serving
# ---------------------------------------------------------------------------

def test_root_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "WheelValue" in r.text


# ---------------------------------------------------------------------------
# GET /api/metadata
# ---------------------------------------------------------------------------

def test_metadata_shape():
    r = client.get("/api/metadata")
    assert r.status_code == 200
    body = r.json()
    assert "categorical_options" in body
    assert "numeric_ranges" in body
    assert "model_metrics" in body
    assert "feature_importance" in body


def test_metadata_categorical_options():
    body = client.get("/api/metadata").json()
    cats = body["categorical_options"]
    for key in ("brand", "fuel", "seller_type", "transmission", "owner"):
        assert key in cats
        assert len(cats[key]) > 0


def test_metadata_numeric_ranges():
    body = client.get("/api/metadata").json()
    for key in ("year", "km_driven", "mileage", "engine", "max_power", "seats"):
        assert key in body["numeric_ranges"]


def test_metadata_model_metrics():
    body = client.get("/api/metadata").json()
    metrics = body["model_metrics"]
    assert "test" in metrics
    assert "validation_mae" in metrics
    assert "comparison" in metrics
    assert metrics["test"]["mae"] > 0
    assert 0 < metrics["test"]["r2"] < 1


def test_metadata_feature_importance():
    body = client.get("/api/metadata").json()
    fi = body["feature_importance"]
    assert isinstance(fi, dict)
    assert len(fi) > 0


# ---------------------------------------------------------------------------
# POST /api/predict — valid input
# ---------------------------------------------------------------------------

def test_predict_valid_returns_price():
    r = client.post("/api/predict", json=VALID_CAR)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_price_inr"] > 0
    assert body["error_range_low"] >= 0
    assert body["error_range_high"] > body["error_range_low"]
    assert body["validation_mae"] > 0
    assert body["car_age"] == 2020 - VALID_CAR["year"]


def test_predict_shap_contributions():
    body = client.post("/api/predict", json=VALID_CAR).json()
    shap = body["shap_contributions"]
    assert isinstance(shap, list)
    assert len(shap) > 0
    first = shap[0]
    assert "feature" in first
    assert "label" in first
    assert "impact" in first
    assert first["direction"] in ("positive", "negative")


def test_predict_submitted_details():
    body = client.post("/api/predict", json=VALID_CAR).json()
    det = body["submitted_details"]
    assert det["brand"] == "Maruti"
    assert det["car_age"] == 6
    assert det["fuel"] == "Petrol"


def test_predict_error_range_uses_validation_mae():
    body = client.post("/api/predict", json=VALID_CAR).json()
    price = body["predicted_price_inr"]
    mae = body["validation_mae"]
    assert abs(body["error_range_high"] - (price + mae)) < 1
    assert body["error_range_low"] == max(0.0, price - mae)


# ---------------------------------------------------------------------------
# POST /api/predict — invalid inputs → 422
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_field,bad_val,desc", [
    ("year", 1800, "year too old"),
    ("year", 2025, "year in future"),
    ("km_driven", -50, "negative km"),
    ("mileage", 0, "zero mileage"),
    ("engine", None, "null engine"),
    ("brand", "", "empty brand"),
    ("seats", 2.5, "fractional seats"),
])
def test_predict_invalid_returns_422(bad_field, bad_val, desc):
    payload = dict(VALID_CAR, **{bad_field: bad_val})
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 422, f"Expected 422 for {desc}"
    body = r.json()
    # detail may come from FastAPI or from our validation
    assert body.get("detail") is not None


# ---------------------------------------------------------------------------
# GET /api/analytics
# ---------------------------------------------------------------------------

def test_analytics_unfiltered():
    r = client.get("/api/analytics")
    assert r.status_code == 200
    body = r.json()
    assert len(body["brand_avg_prices"]) > 0
    assert len(body["fuel_avg_prices"]) > 0
    assert len(body["year_price_trend"]) > 0
    assert body["record_count"] > 0


def test_analytics_top_brands_capped():
    body = client.get("/api/analytics").json()
    assert len(body["brand_avg_prices"]) <= 15


def test_analytics_filtered_by_fuel():
    r = client.get("/api/analytics?fuel=Petrol")
    assert r.status_code == 200
    body = r.json()
    assert body["record_count"] > 0
    # Fuel breakdown should only contain Petrol
    for item in body["fuel_avg_prices"]:
        assert item["fuel"] == "Petrol"


def test_analytics_filtered_by_transmission():
    r = client.get("/api/analytics?transmission=Manual")
    assert r.status_code == 200
    body = r.json()
    assert body["record_count"] > 0


def test_analytics_combined_filter():
    r = client.get("/api/analytics?fuel=Diesel&transmission=Manual")
    assert r.status_code == 200
    body = r.json()
    assert body["record_count"] > 0


def test_analytics_has_lakh_prices():
    body = client.get("/api/analytics").json()
    for item in body["brand_avg_prices"]:
        assert "avg_price_lakhs" in item
        assert item["avg_price_lakhs"] > 0
