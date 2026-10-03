from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import xgboost as xgb

from challenge.connectors.local import LocalArtifactStore, LocalCSVClient
from challenge.connectors.protocols import (
    ArtifactStore,
    DataReader,
)

logger = logging.getLogger(__name__)

# Las 10 variables más importantes según el notebook.
FEATURES_COLS = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]

# Un vuelo está atrasado si sale más de 15 minutos tarde.
# Según el notebook (threshold_in_minutes = 15).
DELAY_MINUTES = 15

class BaseDelayModel(ABC):
    """Abstract interface — defines the obligatory ML contract.

    Subclasses must implement preprocess, fit, and predict.
    Attempting to instantiate DelayModel without all three raises TypeError.
    """

    def __init__(self) -> None:
        self._model = None  # fitted XGBClassifier, populated by fit()

    @abstractmethod
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference.

        Parameters
        ----------
        data : pd.DataFrame
            Raw flight records with columns needed for training and/or inference.
        target_column : str, optional
            Name of the target column to return (e.g. "delay").

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Features and target DataFrames when target_column is provided.
        pd.DataFrame
            Features DataFrame only when target_column is None.
        """

    @abstractmethod
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train the selected model.

        Parameters
        ----------
        features : pd.DataFrame
            Preprocessed feature matrix (columns = FEATURES_COLS).
        target : pd.DataFrame
            Binary delay labels (column = "delay").
        """

    @abstractmethod
    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions for each row in features.

        Parameters
        ----------
        features : pd.DataFrame
            Preprocessed feature matrix (columns = FEATURES_COLS).

        Returns
        -------
        list[int]
            0 (no delay) or 1 (delayed) for each input row.
        """


class GCPModelMixin:
    """Optional GCP I/O operations with injectable client support.

    Accepts ``reader`` and ``store`` keyword arguments so local and GCP
    implementations can be swapped without changing the step-function signatures.
    Defaults to ``LocalCSVReader`` and ``LocalArtifactStore`` for zero-credential
    local operation.
    """

    def __init__(
        self,
        reader: DataReader | None = None,
        store: ArtifactStore | None = None,
    ) -> None:
        """Initialise with optional injectable clients.

        Parameters
        ----------
        reader : DataReader, optional
            Data-loading client; defaults to ``LocalCSVReader()``.
        store : ArtifactStore, optional
            Artifact-persistence client; defaults to ``LocalArtifactStore()``.
        """
        # Ejecutamos el __init__ de BaseDelayModel, que crea self._model.
        super().__init__()

        # Si no recibimos lo necesario, usamos los locales.
        self._reader = reader if reader is not None else LocalCSVClient()
        self._store = store if store is not None else LocalArtifactStore()


    def load_data(self, project: str = "", bq_dataset: str = "") -> pd.DataFrame:
        """Load raw flight data via the injected DataReader.

        Parameters
        ----------
        project : str
            GCP project ID (unused in local mode; accepted for API compatibility).
        bq_dataset : str
            BigQuery dataset name (unused in local mode; accepted for API compatibility).

        Returns
        -------
        pd.DataFrame
            Raw flight records ready to pass to preprocess().
        """
        return self._reader.read()

    def save(self, bucket: str = "", run_id: str = "") -> str:
        """Persist the fitted model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str
            Unique identifier for this training run.

        Returns
        -------
        str
            Path or URI of the saved artifact.
        """
        return self._store.save(self, run_id)

    def load(self, bucket: str = "", run_id: str | None = None) -> None:
        """Restore a model via the injected ArtifactStore.

        Parameters
        ----------
        bucket : str
            GCS bucket name (unused in local mode; accepted for API compatibility).
        run_id : str, optional
            Specific run to load; latest artifact when omitted.
        """
        self._store.load(self, run_id)


class DelayModel(GCPModelMixin, BaseDelayModel):
    """Concrete class to implement for the challenge.

    Required ML methods are declared abstract in BaseDelayModel — omitting
    any of them raises TypeError at instantiation time.

    GCP I/O methods from GCPModelMixin are optional: implement them to
    connect preprocess/fit/predict to BigQuery and Cloud Storage.
    """

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | pd.DataFrame:
        """Prepare raw data for training or inference."""
        # Primero, pasamos aerolinea, tipo de vuelo, mes a columnas de 0
        # y 1 como en el notebook.
        features = pd.concat(

            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1


        )
        # Se debe dejar las 10 columnas exactas, se llena con ceros si
        # falta alguna
        features = features.reindex(columns=FEATURES_COLS, fill_value=0).astype(int)


        # PAra predecir:
        if target_column is None:
            return features

        # Para entrenar, calculamos si cada vuelo se atrasó
        fecha_programada = pd.to_datetime(data["Fecha-I"])
        fecha_real = pd.to_datetime(data["Fecha-O"])

        minutos_atraso = (fecha_real - fecha_programada).dt.total_seconds()/60

        target = pd.DataFrame(

            {target_column: np.where(minutos_atraso > DELAY_MINUTES, 1, 0)},
            index=data.index,
        )

        return features, target
        
    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Train model with class-imbalance weighting."""
        y = target.iloc[:, 0]

        # Hay muchos menos vuelos atrasados, así que les damos más peso
        # Peso de los vuelos atrasados = vuelos sin atraso / vuelos atrasados.
        #  Aquí se calcula sobre los datos que recibe fit, en vez
        # de sobre un split fijo como en el notebook.
        peso_atrasados = (y == 0).sum() / (y == 1).sum()

        # Mismo modelo que eligió el notebook.
        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=peso_atrasados
        )
        self._model.fit(features, y)

    def predict(self, features: pd.DataFrame) -> list[int]:
        """Return binary delay predictions."""
        # Si no hay modelo entrenado, respondemos "sin atraso" (0) para todo
        if self._model is None:
            logger.warning("No trained model: returning 0 for every row")
            return [0] * len(features)

        # El modelo devuelve números de numpy-> los pasamos a int normales.
        return [int(p) for p in self._model.predict(features[FEATURES_COLS])]
