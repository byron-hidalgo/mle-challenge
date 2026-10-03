"""Artifact-store protocol and local/GCS filesystem implementations."""

from __future__ import annotations

import io
from typing import Any

import joblib
import pandas as pd
from google.cloud import storage

from challenge.config import get_settings
from challenge.connectors import bigquery


class GCSArtifactStore:
    """Save and load model artifacts on Google Cloud Storage.

    The bucket name is read from ``Settings.gcp_bucket_artifacts`` at call time.

    Layout::

        gs://<bucket>/
            <run_id>/model.joblib
            latest.txt           ← contains the most-recent run_id
    """

    def _bucket_name(self) -> str:
        return get_settings().gcp_bucket_artifacts

    def _bucket(self) -> storage.Bucket:
        # El cliente se crea al usarlo (no al crear la clase), así no pide
        # credenciales de GCP hasta que realmente se necesitan
        return storage.Client().bucket(self._bucket_name())
    
    def save(self, model: Any, run_id: str) -> str:
        """Serialize and upload model to GCS; update latest.txt."""
        bucket = self._bucket()

        # Basta con convertir el modelo a bytes
        buffer = io.BytesIO()
        joblib.dump(model._model, buffer)
        bucket.blob(f"{run_id}/model.joblib").upload_from_string(buffer.getvalue())


        # Se anota cual es el último modelo entrenado
        bucket.blob("latest.txt").upload_from_string(run_id)
        return f"gs://{self._bucket_name()}/{run_id}/model.joblib"

    def load(self, model: Any, run_id: str | None) -> None:
        """Download and deserialize model from GCS into model._model."""
        bucket = self._bucket()

        # Si run_id, entonces usamos el último modelo
        if run_id is None:
            latest = bucket.blob("latest.txt")
            if not latest.exists():
                raise FileNotFoundError(f"latest.txt not found in gs://{self._bucket_name()}")
            run_id = latest.download_as_text().strip()

        data = bucket.blob(f"{run_id}/model.joblib").download_as_bytes()
        model._model = joblib.load(io.BytesIO(data))


class GCSCSVClient:
    """Read flight data as a CSV object from the GCS input bucket.

    Predictions are written to BigQuery (not GCS) by delegating to an
    internal ``BigQueryClient`` — only the read side bypasses BigQuery.
    """

    def __init__(self, blob_name: str, _bq_writer: Any | None = None) -> None:
        """Initialise with the input-bucket object name and an optional injected writer."""
        self._blob_name = blob_name
        self._bq_writer = _bq_writer

    def read(self) -> pd.DataFrame:
        """Download and parse the CSV object from the configured input bucket."""
        bucket = storage.Client().bucket(get_settings().gcp_bucket_input)
        data = bucket.blob(self._blob_name).download_as_bytes()
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to the configured BigQuery predictions table."""
        # AHora las predicciones se guardan en BigQuery
        if self._bq_writer is None:
            self._bq_writer = bigquery.BigQueryClient()
        self._bq_writer.write(predictions, identifiers)
