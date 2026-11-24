"""WheelValue historical resale price estimator."""
import json
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.explain import explain_single_prediction
from src.features import DATA, MODELS, build_inference_dataframe, validate_car_input

st.set_page_config(page_title="WheelValue", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODELS / "wheelvalue_pipeline.joblib")


@st.cache_data
def load_json(name):
    return json.loads((MODELS / name).read_text())


@st.cache_data
def training_data():
    return pd.read_csv(DATA / "train.csv")


def inr(value):
    digits = str(round(value))
    head, tail = digits[:-3], digits[-3:]
    groups = []
    while head:
        groups.insert(0, head[-2:])
        head = head[:-2]
    return "₹" + ",".join(groups + [tail])


st.title("WheelValue")
st.warning("Estimates reflect 2019–2020 CarDekho market prices. Use them as historical guidance. "
           "They do not constitute an official valuation certificate.")
meta = load_json("ui_metadata.json")
metrics = load_json("model_metrics.json")
predict, analytics, performance = st.tabs(["Price predictor", "Market analytics", "Model performance"])

with predict:
    with st.form("car"):
        left, right = st.columns(2)
        payload = {}
        for field, options in meta["categorical_options"].items():
            payload[field] = left.selectbox(field.replace("_", " ").title(), options)
        labels = {"year": "Manufacturing year", "km_driven": "Distance driven (km)",
                  "engine": "Engine capacity (cc)", "mileage": "Mileage (listed value)",
                  "max_power": "Maximum power (bhp)", "seats": "Seats"}
        for field, bounds in meta["numeric_ranges"].items():
            cast = int if field in ("year", "seats", "km_driven") else float
            payload[field] = right.number_input(labels[field], value=cast(bounds["median"]), step=cast(1) if cast is int else 0.01)
        submitted = st.form_submit_button("Predict Resale Price")
    if submitted:
        valid, errors = validate_car_input(payload, meta)
        if not valid:
            for error in errors:
                st.error(error)
        else:
            model = load_model()
            row = build_inference_dataframe(payload)
            price = model.predict(row)[0]
            band = metrics["validation_mae"]
            st.metric("Estimated resale price", inr(price))
            st.write(f"Typical error range · {inr(max(0, price-band))} to {inr(price+band)}")
            st.caption("The range extends one cross-validation mean absolute error on either side of the estimate.")
            st.subheader("Submitted car details")
            st.dataframe(pd.DataFrame([payload]), hide_index=True)
            st.subheader("Why this price?")
            effects = explain_single_prediction(model, row)
            fig, ax = plt.subplots(figsize=(7, 3))
            shown = effects.head(6).iloc[::-1]
            ax.barh(shown.label, shown.contribution, color=["#19734b" if v >= 0 else "#ad3c45" for v in shown.contribution])
            ax.set_xlabel("Contribution relative to the model baseline (log-price units)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            for driver in effects.head(3).itertuples():
                direction = "raised" if driver.contribution > 0 else "lowered"
                st.markdown(f"- {driver.label} ({row.iloc[0][driver.feature]}) {direction} the estimate.")

with analytics:
    train = training_data()
    fuel = st.selectbox("Filter listings by fuel", ["All fuels", *meta["categorical_options"]["fuel"]])
    filtered = train if fuel == "All fuels" else train[train.fuel == fuel]
    for field, title in [("brand", "Top 15 brands by average price"), ("fuel", "Average price by fuel"),
                         ("year", "Average price by manufacturing year")]:
        st.subheader(title)
        series = filtered.groupby(field).selling_price.mean().rename("Average price (INR)")
        if field == "year":
            st.line_chart(series)
        else:
            st.bar_chart(series.nlargest(15), sort=False)

with performance:
    st.subheader("Five-fold cross-validation")
    st.dataframe(pd.DataFrame(metrics["comparison"]), hide_index=True)
    st.caption("MAE and RMSE are in INR. MAE gap equals validation MAE minus training MAE.")
    st.subheader(f"Untouched test set · {metrics['winner']}")
    for col, (key, value) in zip(st.columns(3), metrics["test"].items()):
        col.metric(key.upper().replace("R2", "R²"), f"{value:.4f}" if key == "r2" else inr(value))
    st.subheader("Global permutation importance")
    st.caption("Increase in training MAE (INR) after shuffling one feature, averaged over ten repeats.")
    st.bar_chart(pd.Series(load_json("feature_importance.json"), name="MAE increase (INR)"), horizontal=True, sort=False)
