"""
Training entrypoint.

Trains the two production candidate models (Random Forest, XGBoost) using
the hyperparameters found by the RandomizedSearchCV tuning in the original
notebook, logs params/metrics/artifacts to MLflow, registers both models
in the MLflow Model Registry, and promotes whichever has the higher R2
to the "champion" alias that src/serve.py loads at inference time.

Run inside the `trainer` container (see docker-compose.yml), or locally
with MLFLOW_TRACKING_URI and CONFIG_PATH set appropriately.
"""

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.config import load_config
from src.data import build_dataset


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def train_random_forest(cfg, X_train, y_train, X_test, y_test):
    params = cfg["models"]["random_forest"]
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae, rmse, r2 = evaluate_regression(y_test, pred)

    with mlflow.start_run(run_name="random_forest") as run:
        mlflow.log_params({f"rf_{k}": v for k, v in params.items()})
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["registry_model_name"],
        )
        run_id = run.info.run_id

    print(f"[random_forest] MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")
    return run_id, r2


def train_xgboost(cfg, X_train, y_train, X_test, y_test):
    params = cfg["models"]["xgboost"]
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae, rmse, r2 = evaluate_regression(y_test, pred)

    with mlflow.start_run(run_name="xgboost") as run:
        mlflow.log_params({f"xgb_{k}": v for k, v in params.items()})
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=cfg["mlflow"]["registry_model_name"],
        )
        run_id = run.info.run_id

    print(f"[xgboost] MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.4f}")
    return run_id, r2


def promote_champion(cfg, client: MlflowClient, best_run_id: str):
    """Point the 'champion' alias at whichever run just won on R2."""
    model_name = cfg["mlflow"]["registry_model_name"]
    versions = client.search_model_versions(f"name='{model_name}'")
    matching = [v for v in versions if v.run_id == best_run_id]
    if not matching:
        raise RuntimeError(
            f"Could not find a registered model version for run {best_run_id}"
        )
    best_version = matching[0].version
    client.set_registered_model_alias(model_name, "champion", best_version)
    print(f"Promoted {model_name} version {best_version} to alias 'champion'")


def main():
    cfg = load_config()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X_train, X_test, y_train, y_test = build_dataset(cfg)
    print(f"X_train {X_train.shape}, X_test {X_test.shape}")

    rf_run_id, rf_r2 = train_random_forest(cfg, X_train, y_train, X_test, y_test)
    xgb_run_id, xgb_r2 = train_xgboost(cfg, X_train, y_train, X_test, y_test)

    best_run_id = xgb_run_id if xgb_r2 >= rf_r2 else rf_run_id
    winner = "xgboost" if xgb_r2 >= rf_r2 else "random_forest"
    print(f"Champion: {winner} (R2={max(rf_r2, xgb_r2):.4f})")

    client = MlflowClient(tracking_uri=cfg["mlflow"]["tracking_uri"])
    promote_champion(cfg, client, best_run_id)


if __name__ == "__main__":
    main()
