import pandas as pd
import pytest
from src.features import FEATURES, build_inference_dataframe, compute_car_age, extract_brand, validate_car_input

CAR = dict(year=2014, brand="Maruti", km_driven=70000, seats=5, mileage=20, engine=1248,
           max_power=80, fuel="Diesel", transmission="Manual", owner="First Owner", seller_type="Individual")
META = {"numeric_ranges": {"km_driven": {"max": 2400000}}}


def test_brand_and_age():
    assert list(extract_brand(pd.Series([" Land Rover Evoque", "Ashok Leyland Stile", "Maruti Swift"]))) == [
        "Land Rover", "Ashok Leyland", "Maruti"]
    assert list(compute_car_age(pd.Series([1983, 2014, 2020]))) == [37, 6, 0]


def test_inference_schema():
    assert validate_car_input(CAR, META) == (True, [])
    row = build_inference_dataframe(CAR)
    assert row.columns.tolist() == FEATURES
    assert row.car_age.item() == 6


@pytest.mark.parametrize(("key", "value"), [("year", 2021), ("year", 1982), ("seats", 2.5), ("km_driven", -1),
                                             ("km_driven", 2400001), ("mileage", 0), ("engine", float("nan")),
                                             ("max_power", float("inf")), ("brand", " ")])
def test_invalid_input(key, value):
    ok, errors = validate_car_input(dict(CAR, **{key: value}), META)
    assert not ok and errors
