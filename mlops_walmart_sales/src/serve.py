"""
FastAPI serving layer.

Loads the current "champion" model version from the MLflow Model
Registry on startup and exposes it via a /predict endpoint. Because a
newly promoted champion only takes effect on the next container start,
POST /reload lets you hot-swap to the latest champion without a restart.
"""

from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import load_config
from src.data import FEATURE_COLUMNS

cfg = load_config()
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
MODEL_URI = f"models:/{cfg['mlflow']['registry_model_name']}@champion"

_state = {"model": None}


def _load_model():
    _state["model"] = mlflow.pyfunc.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
    except Exception as exc:  # noqa: BLE001 - surface at /health instead of crashing
        print(f"Warning: could not load champion model at startup: {exc}")
    yield


app = FastAPI(title="Walmart Weekly Sales Forecaster", lifespan=lifespan)


class PredictRequest(BaseModel):
    Store: int
    Dept: int
    IsHoliday: bool
    Temperature: float
    Fuel_Price: float
    MarkDown1: float = Field(default=0.0)
    MarkDown2: float = Field(default=0.0)
    MarkDown3: float = Field(default=0.0)
    MarkDown4: float = Field(default=0.0)
    MarkDown5: float = Field(default=0.0)
    CPI: float
    Unemployment: float
    Size: int
    Year: int
    Month: int
    Week: int
    Quarter: int
    Day: int
    Type_B: bool = Field(default=False)
    Type_C: bool = Field(default=False)


class PredictResponse(BaseModel):
    predicted_weekly_sales: float


@app.get("/health")
def health():
    return {
        "status": "ok" if _state["model"] is not None else "model_not_loaded",
        "model_uri": MODEL_URI,
    }


@app.post("/reload")
def reload_model():
    """Re-fetch whichever model version the 'champion' alias currently points to."""
    try:
        _load_model()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc))
    return {"status": "reloaded", "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    row = pd.DataFrame([request.model_dump()])[FEATURE_COLUMNS]
    prediction = _state["model"].predict(row)
    return PredictResponse(predicted_weekly_sales=float(prediction[0]))
