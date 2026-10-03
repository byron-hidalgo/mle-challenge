"""Data-reader and Artifactor local implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class LocalCSVClient:
    """Read a CSV from the local data directory and write predictions to disk.

    The data directory is taken from the ``DATA_DIR`` environment variable,
    defaulting to ``data/`` relative to the current working directory.

    When ``filename`` is ``None`` (default): globs for the single ``.csv`` in
    the directory, raising ``ValueError`` on multiples and ``FileNotFoundError``
    on none — preserving original training behaviour.

    When ``filename`` is provided: reads that specific file directly, raising
    ``FileNotFoundError`` if it does not exist.
    """

    def __init__(self, filename: str | None = None) -> None:
        """Initialise the reader with an optional specific filename."""
        self._filename = filename

    def read(self) -> pd.DataFrame:
        """Return a DataFrame from the target CSV file."""
        # Directorio asociado a los datos
        data_dir = Path(os.environ.get("DATA_DIR", "data"))
        # Lectura con validación, en caso de que no exista
        if self._filename:
            path = data_dir / self._filename
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            return pd.read_csv(path, low_memory=False)

        csv_files = list(data_dir.glob("*.csv"))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        if len(csv_files) > 1:
            raise ValueError(f"Multiple CSV files found in {data_dir}, pass a filename")
        return pd.read_csv(csv_files[0], low_memory=False)

    def write(self, predictions: list[int], identifiers: pd.DataFrame) -> None:
        """Write predictions to ``<ARTIFACTS_DIR>/predictions.csv``.

        Creates the artifacts directory if it does not exist.
        Output columns: ``OPERA``, ``TIPOVUELO``, ``MES`` (row-aligned with
        predictions) followed by ``prediction`` (0|1).
        """
        out_dir = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
        out_dir.mkdir(parents=True, exist_ok=True)

        df = identifiers[["OPERA", "TIPOVUELO", "MES"]].reset_index(drop=True)
        df["prediction"] = predictions
        df.to_csv(out_dir / "predictions.csv", index=False)


class LocalArtifactStore:
    """Save and load model artifacts on the local filesystem.

    The root directory is taken from the ``ARTIFACTS_DIR`` environment variable,
    defaulting to ``artifacts/`` relative to the current working directory.

    Layout::

        <ARTIFACTS_DIR>/
            <run_id>/model.joblib
            latest.txt           ← contains the most-recent run_id
    """

    def _root(self) -> Path:
        return Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))

    def save(self, model: Any, run_id: str) -> str:
        """Save model to <ARTIFACTS_DIR>/<run_id>/model.joblib and update latest.txt."""
        run_dir = self._root() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        path = run_dir / "model.joblib"
        joblib.dump(model._model, path)
        (self._root() / "latest.txt").write_text(run_id)
        return str(path)

    def load(self, model: Any, run_id: str | None) -> None:
        """Load model from <ARTIFACTS_DIR>/<run_id>/model.joblib into model._model."""
        if run_id is None:
            latest = self._root() / "latest.txt"
            if not latest.exists():
                raise FileNotFoundError(f"latest.txt not found in {self._root()}")
            run_id = latest.read_text().strip()
        model._model = joblib.load(self._root() / run_id / "model.joblib")