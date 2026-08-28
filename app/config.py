"""Environment-backed configuration for the demand forecasting API."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ApiSettings:
    database_url: str | None
    demand_data_mode: str
    synthetic_batch: str | None
    model_path: Path
    metadata_path: Path
    cors_origins: tuple[str, ...]
    log_level: str

    @classmethod
    def from_environment(cls) -> "ApiSettings":
        mode = os.getenv("DEMAND_DATA_MODE", "production").strip().lower()
        if mode not in {"production", "synthetic"}:
            raise ValueError("DEMAND_DATA_MODE doit valoir production ou synthetic.")
        synthetic_batch = os.getenv("DEMAND_SYNTHETIC_BATCH") if mode == "synthetic" else None
        if mode == "synthetic" and not synthetic_batch:
            raise ValueError("DEMAND_SYNTHETIC_BATCH est obligatoire quand DEMAND_DATA_MODE=synthetic.")
        origins = tuple(
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
            if origin.strip()
        )
        if "*" in origins:
            raise ValueError("CORS_ORIGINS ne peut pas contenir '*'.")
        return cls(
            database_url=os.getenv("DATABASE_URL"),
            demand_data_mode=mode,
            synthetic_batch=synthetic_batch,
            model_path=Path(os.getenv("DEMAND_MODEL_PATH", ROOT / "artifacts" / "models" / "demand_model_deployment.joblib")),
            metadata_path=Path(
                os.getenv(
                    "DEMAND_MODEL_METADATA_PATH",
                    ROOT / "artifacts" / "models" / "demand_model_deployment_metadata.json",
                ),
            ),
            cors_origins=origins,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
