# Walmart Weekly Sales — MLOps Pipeline

Training, experiment tracking, model versioning, and deployment for the
Random Forest / XGBoost weekly-sales forecasters from `modeling.ipynb`,
containerized with Docker.

## Architecture

```
┌─────────────┐      logs runs, params,      ┌──────────────────────┐
│   trainer   │ ───► metrics, model artifacts ─► │  mlflow (tracking +  │
│ (one-shot)  │      registers model versions │  registry) :5000     │
└─────────────┘      promotes "champion"      └──────────┬───────────┘
                                                          │ loads models:/…@champion
                                                          ▼
                                               ┌──────────────────────┐
                                               │   api (FastAPI)      │
                                               │   /predict  :8000    │
                                               └──────────────────────┘
```

- **Tracking & versioning**: every training run logs its hyperparameters,
  MAE/RMSE/R2, and the fitted model to MLflow. Both Random Forest and
  XGBoost are registered as versions of the same model,
  `walmart-sales-forecaster`, in the MLflow Model Registry.
- **Promotion**: after both models finish, `train.py` compares R2 and
  points a `champion` alias at whichever version won. The API always
  serves whatever `champion` currently points to.
- **Deployment**: the FastAPI service loads `models:/walmart-sales-forecaster@champion`
  from the registry and exposes `POST /predict`.

## 1. Add the data

Download the Walmart Kaggle dataset (`train.csv`, `features.csv`,
`stores.csv`) and place all three files in `./data/`.

## 2. Start MLflow

```bash
docker compose up -d mlflow
```

MLflow UI: http://localhost:5000

## 3. Run training

```bash
docker compose run --rm trainer
```

This builds the trainer image, prepares the dataset, trains both models,
logs everything to MLflow, and promotes the better one to `champion`.
Re-run this any time you want to retrain (e.g. new data drop) — it will
register new versions and re-promote the champion automatically.

## 4. Start the API

```bash
docker compose up -d api
```

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "Store": 1, "Dept": 1, "IsHoliday": false,
        "Temperature": 42.31, "Fuel_Price": 2.572,
        "MarkDown1": 0, "MarkDown2": 0, "MarkDown3": 0,
        "MarkDown4": 0, "MarkDown5": 0,
        "CPI": 211.096358, "Unemployment": 8.106, "Size": 151315,
        "Year": 2010, "Month": 2, "Week": 5, "Quarter": 1, "Day": 5,
        "Type_B": false, "Type_C": false
      }'
```

If you retrain later and want the running API to pick up the new
champion without restarting the container:

```bash
curl -X POST http://localhost:8000/reload
```

## Project layout

```
config.yaml          # paths, split ratio, model hyperparameters
src/
  config.py           # loads config.yaml
  data.py              # merge / impute / feature-engineer / split (mirrors the notebook)
  train.py             # trains RF + XGBoost, logs to MLflow, promotes champion
  serve.py             # FastAPI app serving the champion model
Dockerfile.train        # training container
Dockerfile.serve        # serving container
docker-compose.yml       # mlflow + trainer + api
data/                    # put train.csv / features.csv / stores.csv here
```

## Notes

- SARIMA, ETS, and LSTM from the original notebook are exploratory
  time-series/deep-learning experiments, not part of this production
  pipeline — they either require `statsmodels` (not installed in your
  original environment, per the earlier notebook error) or add
  significant image weight (`tensorflow`) for a marginal gain over the
  tuned XGBoost/Random Forest models the notebook itself selected between
  at the end. Add a similar `train_*` function in `src/train.py` and a
  matching `requirements-train.txt` entry if you want to bring one of them
  into the pipeline.
- Hyperparameters in `config.yaml` are the tuned values your
  `RandomizedSearchCV` runs found in the notebook. The registry keeps
  every past run/version if you want to roll back — just repoint the
  `champion` alias in the MLflow UI (Models tab).
