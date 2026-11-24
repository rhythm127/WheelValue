"""Feature effects in INR error (global) and log-price units (local)."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from src.features import FEATURES, NUM, CAT


def global_importance(model, x, y):
    result = permutation_importance(model, x, y, n_repeats=10, random_state=42,
                                    scoring="neg_mean_absolute_error")
    return dict(sorted(zip(FEATURES, result.importances_mean), key=lambda pair: -pair[1]))


def explain_single_prediction(pipeline, input_df):
    import shap
    prep, regressor = pipeline.regressor_.steps
    prep, regressor = prep[1], regressor[1]
    vector = prep.transform(input_df)
    if isinstance(regressor, LinearRegression):
        explainer = shap.LinearExplainer(regressor, pipeline.shap_background_)
    else:
        explainer = shap.TreeExplainer(regressor)
    values = np.asarray(explainer.shap_values(vector))[0]
    effects = dict(zip(NUM, values))
    offset = len(NUM)
    encoder = prep.named_transformers_["cat"][-1]
    for feature, categories in zip(CAT, encoder.categories_):
        end = offset + len(categories)
        effects[feature] = values[offset:end].sum()
        offset = end
    table = pd.DataFrame({"feature": effects.keys(), "contribution": effects.values()})
    table["label"] = table.feature.str.replace("_", " ").str.capitalize()
    return table.sort_values("contribution", key=abs, ascending=False).reset_index(drop=True)
