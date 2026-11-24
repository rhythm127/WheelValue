import joblib
import numpy as np
import pandas as pd
from src.features import FEATURES, DATA, MODELS
from src.explain import explain_single_prediction


def test_export_and_inference():
    path = MODELS / "wheelvalue_pipeline.joblib"
    assert 0 < path.stat().st_size < 100_000_000
    model = joblib.load(path)
    row = pd.read_csv(DATA / "train.csv", nrows=1)[FEATURES]
    for brand in [row.brand.item(), "Unseen brand"]:
        sample = row.assign(brand=brand)
        price = model.predict(sample)
        assert np.isfinite(price).all() and price.item() > 0
        np.testing.assert_allclose(price, model.predict(sample), rtol=1e-12)
    effects = explain_single_prediction(model, row)
    assert sorted(effects.feature) == sorted(FEATURES)
    assert np.isfinite(effects.contribution).all()
