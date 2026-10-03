"""Cloud Run Job entry point for model serving."""

import logging
import sys

import pandas as pd

from challenge.config import get_settings
from challenge.connectors import gcs
from challenge.connectors.local import LocalCSVClient
from challenge.model import DelayModel

logger = logging.getLogger(__name__)

### Define here all the step for the pipeline implementation ###

SERVING_FILE = "serving_input.csv"

def step_load_model(model: DelayModel) -> None:
    """Primero, cargar el último modelo entrenado"""
    model.load()


def step_load_data(model: DelayModel) -> pd.DataFrame:
    """Segundo leer los vuelos a predecir"""
    return model.load_data()


def step_preprocess(model: DelayModel, data: pd.DataFrame) -> pd.DataFrame:
    """Luego, convertir los vuelos en las variables del modelo"""
    return model.preprocess(data)


def step_predict(model: DelayModel, features: pd.DataFrame) -> list[int]:
    """El siguiente paso es predecir si cada vuelo se atrasará (1) o no (0)."""
    return model.predict(features)


def main() -> str:
    """Orchestrate the full serving pipeline."""
    settings = get_settings()

    # En GCP se lee de GCS y se escribe en BigQuery, en local, se usa otro path.
    if settings.deployment_mode == "gcp":
        reader = gcs.GCSCSVClient(blob_name=SERVING_FILE)
        model = DelayModel(reader=reader, store=gcs.GCSArtifactStore())
    else:
        reader = LocalCSVClient(SERVING_FILE)
        model = DelayModel(reader=reader)

    step_load_model(model)
    data = step_load_data(model)
    features = step_preprocess(model, data)
    predictions = step_predict(model, features)

    # Al final -> guardar cada predicción junto a su aerolínea, tipo de vuelo y mes.
    reader.write(predictions, data[["OPERA", "TIPOVUELO", "MES"]])
    logger.info("%d predictions written", len(predictions))
    return f"{len(predictions)} predictions written"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        main()
    except Exception:
        logger.exception("Serving job failed")
        sys.exit(1)
