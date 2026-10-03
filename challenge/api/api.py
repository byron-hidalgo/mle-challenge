"""Pipeline control FastAPI — triggers Cloud Run Job training/serving pipelines."""

import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, run_v2, storage

from challenge import serving, training
from challenge.config import get_settings

logger = logging.getLogger(__name__)


# Par+ametros de los Cloud Run Jobs en GCP
TRAINING_JOB = os.environ.get("TRAINING_JOB_NAME", "delay-training")
SERVING_JOB = os.environ.get("SERVING_JOB_NAME", "delay-serving")

# Las predicciones se guardan en memoria durante 60 segundos, para no consultar
# BigQuery en cada request.
CACHE_SECONDS = 60
_cache = {"data": None, "loaded_at": 0.0}

# Evita que se lancen dos entrenamientos al mismo tiempo.
_training_lock = threading.Lock()


def _is_local() -> bool:
    return get_settings().deployment_mode == "local"


def _artifacts_dir() -> Path:
    return Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))


def _bq_table(name: str) -> str:
    settings = get_settings()
    return f"{settings.gcp_project_id}.{settings.bq_dataset}.{name}"


def _gcs_bucket() -> storage.Bucket:
    return storage.Client().bucket(get_settings().gcp_bucket_artifacts)


def _check_raw_flights() -> int:
    """Cuenta cuántos vuelos hay disponibles para entrenar."""
    if _is_local():
        path = Path(os.environ.get("DATA_DIR", "data")) / "data.csv"
        return len(pd.read_csv(path, usecols=[0])) if path.exists() else 0

    query = f"SELECT COUNT(*) AS n FROM `{_bq_table(get_settings().bq_raw_table)}`"
    try:
        rows = bigquery.Client().query(query).result()
    except NotFound:
        return 0
    return next(iter(rows))["n"]


def _check_model_exists() -> bool:
    """Revisa si ya hay al menos un modelo entrenado (en latest.txt)."""
    if _is_local():
        return (_artifacts_dir() / "latest.txt").exists()
    return _gcs_bucket().blob("latest.txt").exists()


def _submit_and_wait(pipeline: str) -> SimpleNamespace:
    """Ejecuta el pipeline ("training" o "serving") y espera a que termine."""
    if _is_local():
        # En local, el pipeline se ejecuta directamente dentro de este proceso.
        (training if pipeline == "training" else serving).main()
        return SimpleNamespace(name=f"local-{pipeline}-{uuid.uuid4().hex[:8]}")

    # En GCP, se lanza el Cloud Run Job y se espera a que termine
    settings = get_settings()
    job = TRAINING_JOB if pipeline == "training" else SERVING_JOB
    job_path = f"projects/{settings.gcp_project_id}/locations/{settings.gcp_region}/jobs/{job}"
    execution = run_v2.JobsClient().run_job(name=job_path).result(timeout=3600)
    if execution.failed_count:
        raise RuntimeError(f"{job} execution failed: {execution.name}")
    return execution


def _read_metrics() -> dict:
    """Lee las métricas del último entrenamiento. Si algo falla, devuelve {}."""
    try:
        if _is_local():
            run_id = (_artifacts_dir() / "latest.txt").read_text().strip()
            return json.loads((_artifacts_dir() / run_id / "metrics.json").read_text())
        bucket = _gcs_bucket()
        run_id = bucket.blob("latest.txt").download_as_text().strip()
        return json.loads(bucket.blob(f"{run_id}/metrics.json").download_as_text())
    except Exception:
        logger.warning("Could not read training metrics", exc_info=True)
        return {}


def _load_predictions() -> pd.DataFrame:
    """Devuelve la tabla de predicciones, usando la copia en memoria si es reciente."""
    if _cache["data"] is not None and time.time() - _cache["loaded_at"] < CACHE_SECONDS:
        return _cache["data"]

    empty = pd.DataFrame(columns=["OPERA", "TIPOVUELO", "MES", "prediction"])
    if _is_local():
        path = _artifacts_dir() / "predictions.csv"
        data = pd.read_csv(path) if path.exists() else empty
    else:
        query = f"SELECT * FROM `{_bq_table(get_settings().bq_predictions_table)}`"
        try:
            data = bigquery.Client().query(query).to_dataframe()
        except NotFound:
            data = empty

    _cache["data"] = data
    _cache["loaded_at"] = time.time()
    return data


def _query_predictions(
    page: int,
    page_size: int,
    opera: list[str] | None,
    tipovuelo: list[str] | None,
    mes: list[int] | None,
) -> dict:
    """Filtra y avanza con las predicciones."""
    data = _load_predictions()
    if opera:
        data = data[data["OPERA"].isin(opera)]
    if tipovuelo:
        data = data[data["TIPOVUELO"].isin(tipovuelo)]
    if mes:
        data = data[data["MES"].isin(mes)]

    start = (page - 1) * page_size
    rows = data.iloc[start : start + page_size].rename(columns={"prediction": "predicted_delay"})
    return {"total_predictions": len(data), "predictions": rows.to_dict(orient="records")}


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — placeholder for startup/shutdown logic."""
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    """Return liveness status."""
    return {"status": "ok"}


@app.post("/pipeline/train", status_code=200)
async def post_pipeline_train() -> dict:
    """Trigger the training Cloud Run Job and return model metadata."""
    if _check_raw_flights() == 0:
        raise HTTPException(status_code=422, detail="raw_flights table is empty")

    if not _training_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="a training run is already in progress")
    try:
        job = await run_in_threadpool(_submit_and_wait, "training")
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={"error": "training pipeline failed", "details": str(error)},
        ) from error
    finally:
        _training_lock.release()

    metrics = _read_metrics()
    return {
        "model_display_name": "delay-classifier",
        "version": metrics.get("run_id"),
        "pipeline_job_id": job.name,
        "metrics": metrics,
    }



@app.post("/pipeline/predict", status_code=200)
async def post_pipeline_predict() -> dict:
    """Trigger the serving Cloud Run Job and return first 10 predictions."""
    if not _check_model_exists():
        raise HTTPException(
            status_code=422, detail="no trained model found, run /pipeline/train first"
        )

    try:
        job = await run_in_threadpool(_submit_and_wait, "serving")
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail={"error": "serving pipeline failed", "details": str(error)},
        ) from error

    # si es que hay predicciones nuevas -> descartamos la copia en memoria.
    _cache["data"] = None
    result = _query_predictions(1, 10, None, None, None)
    return {"pipeline_job_id": job.name, **result}


@app.get("/pipeline/predict/results", status_code=200)
async def get_prediction_results(
    page: int = 1,
    page_size: int = 10,
    opera: str | None = None,
    tipovuelo: str | None = None,
    mes: str | None = None,
) -> dict:
    """Return a paginated page of predictions, optionally filtered by OPERA/TIPOVUELO/MES."""
    try:
        mes_list = [int(m) for m in _split(mes)] if mes else None
    except ValueError as error:
        raise HTTPException(
            status_code=422, detail="mes must be a comma-separated list of integers (1-12)"
        ) from error

    result = _query_predictions(page, page_size, _split(opera), _split(tipovuelo), mes_list)
    return {"page": page, "page_size": page_size, **result}
