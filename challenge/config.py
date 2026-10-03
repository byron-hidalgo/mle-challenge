"""Centralised, lazily-loaded configuration for the challenge package."""

import os
from dataclasses import dataclass
from functools import lru_cache

_VALID_MODES = {"local", "gcp"}

# Settings attribute -> environment variable required in gcp mode
_GCP_REQUIRED = {
    "gcp_project_id": "GCP_PROJECT_ID",
    "gcp_bucket_artifacts": "GCP_BUCKET_ARTIFACTS",
}

@dataclass(frozen=True)
class Settings:
    """All environment-variable configuration in one place."""

    deployment_mode: str
    data_dir: str
    artifacts_dir: str
    gcp_project_id: str
    gcp_region: str
    gcp_bucket_artifacts: str
    gcp_bucket_input: str
    bq_dataset: str
    bq_raw_table: str
    bq_predictions_table: str

    def __post_init__(self) -> None:
        """Validate required GCP vars when deployment_mode is 'gcp'."""
        if self.deployment_mode not in _VALID_MODES:
            raise ValueError(
                f"DEPLOYMENT_MODE must be one of {sorted(_VALID_MODES)}, "
                f"got {self.deployment_mode!r}"
            )
        if self.deployment_mode == "gcp":
            missing = [env for attr, env in _GCP_REQUIRED.items() if not getattr(self, attr)]
            if missing:
                raise ValueError(f"DEPLOYMENT_MODE=gcp requires: {', '.join(missing)}")



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings read from environment variables."""
    env = os.environ.get
    return Settings(
        deployment_mode=env("DEPLOYMENT_MODE", "local").lower(),
        data_dir=env("DATA_DIR", "data"),
        artifacts_dir=env("ARTIFACTS_DIR", "artifacts"),
        gcp_project_id=env("GCP_PROJECT_ID", ""),
        gcp_region=env("GCP_REGION", "us-central1"),
        gcp_bucket_artifacts=env("GCP_BUCKET_ARTIFACTS", ""),
        gcp_bucket_input=env("GCP_BUCKET_INPUT", ""),
        bq_dataset=env("BQ_DATASET", "mle_challenge"),
        bq_raw_table=env("BQ_RAW_TABLE", "raw_flights"),
        bq_predictions_table=env("BQ_PREDICTIONS_TABLE", "predictions"),
    )


def clear_settings_cache() -> None:
    """Invalidate the cached settings — call in test teardown after patching env vars."""
    get_settings.cache_clear()
