"""Reproduce the split, model selection and final holdout evaluation."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, RandomizedSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.features import DATA, MODELS, NUM, CAT, FEATURES, REFERENCE_YEAR, split_data
from src.explain import global_importance

SCORING = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
CV = KFold(5, shuffle=True, random_state=42)


def make_model(regressor):
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), CAT)])
    pipeline = make_pipeline(prep, regressor)
    if isinstance(regressor, DummyRegressor):
        return pipeline
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)


def summary(scores, name, index=None):
    row = {"model": name}
    for split in ["train", "test"]:
        for metric in SCORING:
            key = f"{split}_{metric}"
            mean = scores[key].mean() if index is None else scores["mean_" + key][index]
            row[key.replace("test_", "val_")] = float(mean if metric == "r2" else -mean)
    row["mae_gap"] = row["val_mae"] - row["train_mae"]
    return row


def save(name, data):
    (MODELS / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")


def main():
    split_data()
    train = pd.read_csv(DATA / "train.csv")
    x, y = train[FEATURES], train.selling_price
    MODELS.mkdir(exist_ok=True)
    metadata = {"reference_year": REFERENCE_YEAR, "feature_columns": FEATURES,
                "categorical_options": {k: sorted(train[k].dropna().unique()) for k in CAT},
                "numeric_ranges": train[["year", *NUM[1:]]].agg(["min", "max", "median"]).to_dict()}
    save("ui_metadata.json", metadata)
    rows, candidates = [], {}
    for estimator in [DummyRegressor(strategy="mean"), LinearRegression(), DecisionTreeRegressor(random_state=42),
                      RandomForestRegressor(random_state=42, n_jobs=-1), GradientBoostingRegressor(random_state=42)]:
        name = type(estimator).__name__
        candidates[name] = make_model(estimator)
        scores = cross_validate(candidates[name], x, y, cv=CV, scoring=SCORING,
                                return_train_score=True, error_score="raise")
        rows.append(summary(scores, name))
        print(rows[-1], flush=True)
    best = min(rows, key=lambda row: row["val_mae"])
    name = best["model"]
    model = clone(candidates[name])
    prefix = "regressor__" + name.lower() + "__"
    space = {"max_depth": [5, 8, 12, 16, None], "min_samples_leaf": [1, 2, 3, 5]}
    search = RandomizedSearchCV(model, {prefix+k: v for k,v in space.items()}, n_iter=15,
                                scoring=SCORING, refit="mae", cv=CV, random_state=42,
                                return_train_score=True, error_score="raise")
    search.fit(x, y)
    tuned = summary(search.cv_results_, "Tuned " + name, search.best_index_)
    rows.append(tuned)
    params = {}
    if tuned["val_mae"] <= best["val_mae"]:
        best, model = tuned, search.best_estimator_
        params = {k.removeprefix(prefix): v for k,v in search.best_params_.items()}
    else:
        model.fit(x, y)
    if best["val_mae"] >= rows[0]["val_mae"]:
        raise ValueError("Winner must improve on the mean baseline")
    if name == "LinearRegression":
        model.shap_background_ = model.regressor_[0].transform(x.sample(100, random_state=42))
    path = MODELS / "wheelvalue_pipeline.joblib"
    joblib.dump(model, path, compress=3)
    if path.stat().st_size >= 100_000_000:
        raise ValueError("Pipeline exceeds delivery size limit")
    save("feature_importance.json", global_importance(model, x, y))
    holdout = pd.read_csv(DATA / "test.csv")
    predicted = model.predict(holdout[FEATURES])
    actual = holdout.selling_price
    results = {"comparison": rows, "winner": best["model"], "parameters": params,
               "search_params": search.best_params_, "validation_mae": best["val_mae"],
               "test": {"mae": mean_absolute_error(actual, predicted),
                        "rmse": root_mean_squared_error(actual, predicted), "r2": r2_score(actual, predicted)}}
    save("model_metrics.json", results)
    return results


if __name__ == "__main__":
    main()
