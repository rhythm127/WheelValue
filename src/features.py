"""Shared training and inference schema."""
from pathlib import Path
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REFERENCE_YEAR = 2020
NUM = ["car_age", "km_driven", "mileage", "engine", "max_power", "seats"]
CAT = ["brand", "fuel", "seller_type", "transmission", "owner"]
FEATURES = NUM + CAT


def extract_brand(name_series):
    return name_series.str.split().str[0].replace({"Land": "Land Rover", "Ashok": "Ashok Leyland"})


def compute_car_age(year_series, reference_year=REFERENCE_YEAR):
    if reference_year != REFERENCE_YEAR:
        raise ValueError("Use the fixed CarDekho reference year 2020.")
    return reference_year - year_series


def split_data():
    from sklearn.model_selection import train_test_split
    cars = pd.read_csv(DATA / "cardekho_cleaned.csv")
    train, test = train_test_split(cars, test_size=0.2, random_state=42)
    for name, frame in [("train", train), ("test", test)]:
        frame.assign(brand=extract_brand(frame.name), car_age=compute_car_age(frame.year)).to_csv(
            DATA / f"{name}.csv", index=False)


def validate_car_input(payload, metadata):
    errors = []
    for key in CAT:
        if not isinstance(payload.get(key), str) or not payload[key].strip():
            errors.append(f"Choose a value for {key.replace('_', ' ')}.")
    bounds = {"year": (1983, 2020), "seats": (2, 14),
              "km_driven": (0, metadata["numeric_ranges"]["km_driven"]["max"])}
    for key in ["year", *NUM[1:]]:
        try:
            value = float(payload[key])
        except (KeyError, TypeError, ValueError):
            value = math.nan
        if key in bounds:
            low, high = bounds[key]
            if not low <= value <= high or (key in ("seats", "year") and not value.is_integer()):
                errors.append(f"Enter {key} between {low:g} and {high:g}" +
                              (" as a whole number." if key in ("year", "seats") else "."))
        elif not math.isfinite(value) or value <= 0:
            errors.append(f"Enter a positive finite value for {key}.")
    return not errors, errors


def build_inference_dataframe(payload, reference_year=REFERENCE_YEAR):
    row = dict(payload, car_age=compute_car_age(float(payload["year"]), reference_year))
    return pd.DataFrame([row], columns=FEATURES).astype({key: float for key in NUM})

