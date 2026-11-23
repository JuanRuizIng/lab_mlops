"""Configuración del servicio por entorno, independiente del entrenamiento histórico."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

from src.config import ROOT_DIR

SERVING_DIR = ROOT_DIR / "outputs" / "models" / "serving"
MLOPS_DIR = ROOT_DIR / "outputs" / "models" / "mlops"


def active_serving_dir() -> Path:
    """Usa la versión promovida si existe; conserva el artefacto histórico como respaldo."""
    pointer = MLOPS_DIR / "current.json"
    if not pointer.exists():
        return SERVING_DIR
    version = json.loads(pointer.read_text(encoding="utf-8"))["version"]
    if type(version) is not int or version < 1:
        raise ValueError("La versión MLOps activa no es válida")
    return MLOPS_DIR / "releases" / f"v{version}"


@dataclass(frozen=True)
class Settings:
    model_path: Path | None = None
    manifest_path: Path | None = None
    max_batch_size: int = 1000
    max_request_bytes: int = 2 * 1024 * 1024
    log_level: str = "INFO"

    def __post_init__(self):
        if self.model_path is None or self.manifest_path is None:
            serving_dir = active_serving_dir()
            if self.model_path is None:
                object.__setattr__(self, "model_path", serving_dir / "logistic_regression_drybean.joblib")
            if self.manifest_path is None:
                object.__setattr__(self, "manifest_path", serving_dir / "model_manifest.json")
        if self.max_batch_size < 1 or self.max_batch_size > 1000:
            raise ValueError("MAX_BATCH_SIZE debe estar entre 1 y 1000")
        if self.max_request_bytes < 1:
            raise ValueError("MAX_REQUEST_BYTES debe ser positivo")
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("LOG_LEVEL no es válido")

    @classmethod
    def from_env(cls):
        model_path = os.getenv("MODEL_PATH")
        manifest_path = os.getenv("MODEL_MANIFEST_PATH")
        return cls(
            model_path=Path(model_path) if model_path else None,
            manifest_path=Path(manifest_path) if manifest_path else None,
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "1000")),
            max_request_bytes=int(os.getenv("MAX_REQUEST_BYTES", str(2 * 1024 * 1024))),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
