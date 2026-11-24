# WheelValue

**Market limitation — WheelValue estimates resale prices in the 2019–2020 Indian CarDekho market. Use the result as historical guidance. An official valuation or a current sale price requires an independent assessment.**

WheelValue estimates a used car's selling price in rupees from its specifications, ownership and seller details. The predictor shows the estimate, its typical error range and the strongest model drivers. Two additional tabs explore market patterns and model performance.

## Data and methodology

CarDekho's *Car details v3.csv* contains 8,128 listings. Cleaning produced 6,921 rows after removing exact duplicates and test-drive cars, dropping torque, stripping text units and marking missing or impossible measurements for imputation. Vehicle age always equals `2020 - year`. The first token of the name supplies the brand, with special handling for Land Rover and Ashok Leyland.

A seeded 80/20 split (42) reserves 1,385 listings for final evaluation and 5,536 for development. EDA, widget categories and defaults, importance calculations, and model selection use only the development split. Within each of five shuffled folds, the pipeline fits median imputation and scaling for six numeric features, and most-frequent imputation and dense one-hot encoding for five categorical features. Unknown categories produce an all-zero group. Learned regressors fit `log1p(selling_price)` and return INR through `expm1`. The dummy baseline estimates the arithmetic mean price in INR.

## Evaluation

Five-fold means below are rounded to whole rupees. Gap is validation MAE minus training MAE. R² is dimensionless.

| Model | Train MAE | Validation MAE | Train RMSE | Validation RMSE | Train R² | Validation R² | MAE gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dummy mean | 293,511 | 293,531 | 522,541 | 521,041 | 0.0000 | −0.0003 | 20 |
| Linear regression | 93,764 | 95,876 | 178,190 | 187,102 | 0.8837 | 0.8690 | 2,112 |
| Decision tree | 2,687 | 103,717 | 12,608 | 227,684 | 0.9994 | 0.8065 | 101,030 |
| Random forest | 31,034 | 79,506 | 72,189 | 171,818 | 0.9809 | 0.8891 | 48,472 |
| Gradient boosting | 77,706 | 84,370 | 143,637 | 171,583 | 0.9244 | 0.8906 | 6,663 |
| Tuned random forest | 31,034 | 79,506 | 72,189 | 171,818 | 0.9809 | 0.8891 | 48,472 |

Random forest won on validation MAE. A 15-draw randomized search over depth and minimum leaf size selected the original configuration (`max_depth=None`, `min_samples_leaf=1`, 100 trees, seed 42). On the single final holdout evaluation, **MAE was ₹72,615.75, RMSE ₹1,31,855.07 and R² 0.906265**. The compressed pipeline occupies 11,058,163 bytes.

The predictor's error range adds and subtracts the validation MAE of ₹79,506.45, with a zero lower floor. It describes typical absolute error and has no calibrated coverage probability. Local SHAP contributions group every encoded category into its original feature and describe changes in log price. Global importance reports the average increase in training MAE over ten permutations per feature.

## Run locally

Use Python 3.14 and create an environment from the repository root.

```sh
python -m venv .venv
```

Activate with `source .venv/bin/activate` on macOS/Linux or `.\.venv\Scripts\Activate.ps1` in PowerShell. Then run:

```sh
python -m pip install -r requirements.txt
python -m src.train
pytest -v
```

### HTML/CSS/JS web app (FastAPI + uvicorn)

```sh
uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000> in your browser. The server loads the trained pipeline and all JSON metadata once at startup and exposes:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Serves `frontend/index.html` |
| `/api/metadata` | GET | Form dropdowns, numeric bounds, model metrics, feature importance |
| `/api/predict` | POST | Validate input, predict INR price, return SHAP contributions |
| `/api/analytics` | GET | Aggregated training-data charts (`?fuel=&transmission=` filters) |

### Streamlit app (alternative)

```sh
streamlit run app/app.py
```

The included artifacts also support launching the app directly. Each training invocation regenerates the split, runs cross-validation and tuning, exports the selected pipeline with joblib compression level 3 and evaluates the holdout once.

## Repository guide

| Location | Contents |
|---|---|
| `data/raw/` | Original CarDekho listings |
| `data/processed/` | Cleaned listings and reproducible train/test split |
| `notebooks/` | Inspection, cleaning, training-only EDA, modeling and explanation |
| `src/` | Shared features, validation, training and explainability |
| `models/` | Pipeline, metrics, UI metadata and permutation importance |
| `api/server.py` | FastAPI server: JSON endpoints + static frontend serving |
| `frontend/` | HTML5/CSS3/Vanilla JS SPA (index.html, styles.css, app.js) |
| `app/app.py` | Streamlit predictor, analytics and performance views |
| `tests/` | Feature validation, artifact integration, and API endpoint tests |

## Screenshots

| View | Capture |
|---|---|
| Predictor and local explanation | Screenshot pending |
| Filtered market analytics | Screenshot pending |
| Model performance and importance | Screenshot pending |
