"""Cloud Run Job entry point for model training."""

import json
import logging
import sys
import uuid
from pathlib import Path

import pandas as pd
from google.cloud import storage
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from challenge.config import get_settings
from challenge.connectors import gcs
from challenge.connectors.bigquery import BigQueryClient
from challenge.connectors.local import LocalCSVClient
from challenge.model import DelayModel

logger = logging.getLogger(__name__)


### Define here all the step for the pipeline implementation ###

def step_load_data(model: DelayModel, project:str, dataset:str) -> pd.DataFrame:
    """El primer paso es leer los vuelos (BigQuery en GCP, data/data.scsv en local)"""
    return model.load_data(project=project, bq_dataset=dataset)

def step_preprocess(model:DelayModel, data:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Luego, convertir vuelos en variables y calcular si se atrasaron o no"""
    return model.preprocess(data, target_column="delay")

def step_train(model:DelayModel, features: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Se prueba el moodelo con datos no vistos y luego se entrena con todo
    Separo un 33 por ciento para medir el modelo como en el notebook"""
    x_train, x_val, y_train, y_val = train_test_split(

        features, target, test_size=0.33, random_state=42
    )
    model.fit(x_train, y_train)

    report = classification_report(y_val, model.predict(x_val), output_dict=True)

    metrics = {

        "recall_delay": round(report["1"]["recall"], 4),
        "precision_delay": round(report["1"]["precision"], 4),
        "f1_delay": round(report["1"]["f1-score"], 4),
        "accuracy": round(report["accuracy"], 4),
    }
    logger.info("Validation metrics: %s", metrics)

    # El modelo final se entrena con todo
    model.fit(features, target)
    return metrics


def step_save(model:DelayModel, bucket: str, run_id:str, metrics:dict) -> str:
    """Último paso es guardar el modelo y métricas"""
    uri = model.save(bucket=bucket, run_id=run_id)

    metrics_json = json.dumps({"run_id":run_id, **metrics})
    # Se decide local o GCS
    if uri.startswith("gs://"):
        storage.Client().bucket(bucket).blob(f"{run_id}/metrics.json").upload_from_string(
            metrics_json
        )
    else:
        (Path(uri).parent / "metrics.json").write_text(metrics_json)

    logger.info(f"Model saved to {uri}")

    return uri


def main() -> str:
    """Orchestrate the full training pipeline."""
    settings = get_settings()

    # En GCP se lee de BigQuery y se guarda en GCS, en local, se usa path.
    if settings.deployment_mode == "gcp":
        model = DelayModel(reader=BigQueryClient(), store=gcs.GCSArtifactStore())
    else:
        model = DelayModel(reader=LocalCSVClient("data.csv"))

    run_id = uuid.uuid4().hex[:12]

    data = step_load_data(model, settings.gcp_project_id, settings.bq_dataset)
    features, target = step_preprocess(model, data)
    metrics = step_train(model, features, target)
    return step_save(model, settings.gcp_bucket_artifacts, run_id, metrics)
    


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        main()
    except Exception:
        logger.exception("Training job failed")
        sys.exit(1)
