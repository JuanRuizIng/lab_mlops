"""Selección, publicación y observación local del modelo servido."""

from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import platform
import sqlite3
from tempfile import TemporaryDirectory

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deploy.web.api.model_store import load_bundle
from deploy.web.api.schemas import BeanFeatures
from deploy.web.api.settings import MLOPS_DIR, Settings
from src import config
from src.data import clean_data, load_data, split_data
from src.models import build_baseline
from src.training import EXPERIMENT, LOCAL_DATA, log_training, split_digest

MODEL_NAME = "drybean-classifier"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
CURRENT = MLOPS_DIR / "current.json"
RUNTIME_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
MIN_F1 = 0.90
MODEL_FILE = "logistic_regression_drybean.joblib"
MANIFEST_FILE = "model_manifest.json"


def client() -> MlflowClient:
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient(tracking_uri=TRACKING_URI)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_release() -> dict | None:
    return json.loads(CURRENT.read_text(encoding="utf-8")) if CURRENT.exists() else None


def local_split():
    return split_data(clean_data(load_data(prefer_uci=False, save_raw=False), save=False))


def train() -> str:
    X_train, X_test, y_train, y_test = local_split()
    model = build_baseline().fit(X_train, y_train)
    predictions = model.predict(X_test)
    return log_training(
        model, X_train, y_train, X_test, y_test, predictions,
        {"accuracy": accuracy_score(y_test, predictions), "f1_macro": f1_score(y_test, predictions, average="macro")},
        run_name="logistic-regression", source="mlops", dataset_source=LOCAL_DATA, tracking_uri=TRACKING_URI,
    )


def require_local_partitions(run, X_train, X_test, y_train, y_test) -> None:
    expected = {"training": split_digest(X_train, y_train), "evaluation": split_digest(X_test, y_test)}
    inputs = {
        tag.value: item.dataset.digest
        for item in run.inputs.dataset_inputs
        for tag in item.tags
        if tag.key == "mlflow.data.context"
    }
    if (run.data.tags.get("dataset_sha256") != digest(config.RAW_XLSX)
            or inputs != {context: value[:32] for context, value in expected.items()}
            or any(run.data.tags.get(f"{context}_sha256") != value for context, value in expected.items())):
        raise ValueError("El run debe incluir las particiones del Excel local. Genera uno con el comando train de MLOps.")


def register(run_id: str) -> int:
    registry = client()
    run = registry.get_run(run_id)
    experiment = registry.get_experiment(run.info.experiment_id)
    if (experiment.name != EXPERIMENT or run.info.status != "FINISHED"
            or run.data.tags.get("mlflow.runName") != "logistic-regression"):
        raise ValueError("Selecciona un run terminado de regresión logística")
    require_local_partitions(run, *local_split())
    if not run.data.metrics.get("f1_macro_test", 0) >= MIN_F1:
        raise ValueError("El run no supera el umbral mínimo de F1 macro")
    if len(run.outputs.model_outputs) != 1:
        raise ValueError("El run debe contener un único modelo guardado")
    (output,) = run.outputs.model_outputs
    for registered in registry.search_model_versions(f"run_id = '{run.info.run_id}'"):
        if registered.name == MODEL_NAME and registered.status == "READY":
            break
    else:
        registered = mlflow.register_model(f"models:/{output.model_id}", MODEL_NAME, await_registration_for=60)
    if registered.status != "READY":
        raise ValueError("La versión todavía no está lista en Model Registry")
    registry.set_registered_model_alias(MODEL_NAME, "candidate", registered.version)
    return int(registered.version)


def make_release(version: int) -> dict:
    release_dir = MLOPS_DIR / "releases" / f"v{version}"
    if release_dir.exists():
        manifest = json.loads((release_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        verified_bundle(release_dir, manifest)
        return manifest

    registry = client()
    registered = registry.get_model_version(MODEL_NAME, str(version))
    source_run = registry.get_run(registered.run_id)
    X_train, X_test, y_train, y_test = local_split()
    require_local_partitions(source_run, X_train, X_test, y_train, y_test)
    model_uri = f"models:/{MODEL_NAME}/{version}"
    model = mlflow.sklearn.load_model(model_uri)
    if (not isinstance(model, Pipeline) or tuple(model.named_steps) != ("scaler", "classifier")
            or not isinstance(model.named_steps["scaler"], StandardScaler)
            or not isinstance(model.named_steps["classifier"], LogisticRegression)):
        raise ValueError("El modelo seleccionado debe ser el pipeline de regresión logística del laboratorio")
    if list(model.feature_names_in_) != list(BeanFeatures.model_fields):
        raise ValueError("Las 16 variables del modelo no coinciden con la API")
    if set(model.classes_) != set(y_train.unique()):
        raise ValueError("Las clases del modelo no coinciden con el dataset")
    predictions = model.predict(X_test)
    f1 = f1_score(y_test, predictions, average="macro")
    if f1 < MIN_F1:
        raise ValueError(f"F1 macro insuficiente para servir: {f1:.4f}")
    probabilities = model.predict_proba(X_test)
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1)
            or not np.array_equal(model.classes_[probabilities.argmax(axis=1)], predictions)):
        raise ValueError("El artefacto no cumple el contrato de probabilidades de la API")

    manifest = {
        "manifest_version": 1,
        "algorithm": "StandardScaler + LogisticRegression",
        "features": list(model.feature_names_in_),
        "classes": list(model.classes_),
        "python_version": platform.python_version(),
        "packages": {name: package_version(name) for name in RUNTIME_PACKAGES},
        "metrics": {"accuracy": accuracy_score(y_test, predictions), "f1_macro": f1},
        "evaluation": {
            "source": "Excel local; deduplicado; partición estratificada 80/20",
            "rows": len(X_test),
            "random_state": config.RANDOM_STATE,
            "classification_report": classification_report(y_test, predictions, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=model.classes_).tolist(),
        },
        "training": {
            "run_id": registered.run_id,
            "registered_model": MODEL_NAME,
            "registered_version": version,
            "model_uri": model_uri,
            "dataset_sha256": digest(config.RAW_XLSX),
            "train_sha256": split_digest(X_train, y_train),
            "test_sha256": split_digest(X_test, y_test),
        },
    }
    release_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=release_dir.parent) as staging:
        export = Path(staging) / "bundle"
        export.mkdir()
        model_path = export / MODEL_FILE
        joblib.dump(model, model_path)
        model_digest = digest(model_path)
        manifest["model_sha256"] = model_digest
        manifest["model_version"] = f"drybean-mlflow-v{version}-{model_digest[:12]}"
        (export / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
        )
        verified_bundle(export, manifest)
        export.rename(release_dir)
    return manifest


def verified_bundle(directory: Path, manifest: dict):
    if manifest["training"]["registered_model"] != MODEL_NAME:
        raise ValueError("El artefacto local pertenece a otro modelo")
    return load_bundle(Settings(model_path=directory / MODEL_FILE, manifest_path=directory / MANIFEST_FILE))


def promote(version: int | None = None) -> dict:
    registry = client()
    if version is None:
        version = int(registry.get_model_version_by_alias(MODEL_NAME, "candidate").version)
    manifest = make_release(version)
    registered = registry.get_model_version(MODEL_NAME, str(version))
    if (registered.run_id != manifest["training"]["run_id"]
            or int(manifest["training"]["registered_version"]) != version):
        raise ValueError("El artefacto local no corresponde a la versión solicitada en MLflow")
    previous = current_release()
    previous_version = previous["version"] if previous else None
    old_alias = None
    try:
        old_alias = registry.get_model_version_by_alias(MODEL_NAME, "champion").version
    except mlflow.exceptions.MlflowException as error:
        if error.error_code != "RESOURCE_DOES_NOT_EXIST":
            raise
    pointer = {
        "registered_model": MODEL_NAME,
        "version": version,
        "previous_version": previous_version if previous_version != version else (previous or {}).get("previous_version"),
        "model_version": manifest["model_version"],
        "model_sha256": manifest["model_sha256"],
        "run_id": registered.run_id,
    }
    CURRENT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=CURRENT.parent) as staging:
        temporary = Path(staging) / "current.json"
        temporary.write_text(json.dumps(pointer, indent=2) + "\n", encoding="utf-8")
        registry.set_registered_model_alias(MODEL_NAME, "champion", version)
        try:
            os.replace(temporary, CURRENT)
        except OSError:
            if old_alias:
                registry.set_registered_model_alias(MODEL_NAME, "champion", old_alias)
            else:
                registry.delete_registered_model_alias(MODEL_NAME, "champion")
            raise
    return pointer


def rollback() -> dict:
    current = current_release()
    if current is None or current.get("previous_version") is None:
        raise ValueError("No existe una versión anterior para revertir")
    return promote(int(current["previous_version"]))


def monitor(csv_path: Path | None = None) -> dict:
    from src.monitor import build_report

    current = current_release()
    if current is None:
        raise ValueError("Promueve primero un modelo")
    release = MLOPS_DIR / "releases" / f"v{current['version']}"
    manifest = json.loads((release / MANIFEST_FILE).read_text(encoding="utf-8"))
    bundle = verified_bundle(release, manifest)
    provenance = manifest["training"]
    if digest(config.RAW_XLSX) != provenance["dataset_sha256"]:
        raise ValueError("El Excel local ya no coincide con la referencia de la versión activa")
    X_train, X_test, y_train, y_test = local_split()
    if "train_sha256" in provenance and (
        split_digest(X_train, y_train) != provenance["train_sha256"]
        or split_digest(X_test, y_test) != provenance["test_sha256"]
    ):
        raise ValueError("Las particiones locales no coinciden con la referencia de la versión activa")
    report = build_report(bundle, X_train, X_test, csv_path)
    output = config.REPORTS_DIR / "mlops_monitor.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    commands.add_parser("register").add_argument("--run-id", required=True)
    commands.add_parser("promote").add_argument("--version", type=int)
    commands.add_parser("rollback")
    commands.add_parser("status")
    commands.add_parser("monitor").add_argument("--csv", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = {"run_id": train()}
        elif args.command == "monitor":
            result = monitor(args.csv)
        elif args.command == "status":
            result = current_release() or {"status": "no_active_release"}
        else:
            MLOPS_DIR.mkdir(parents=True, exist_ok=True)
            # Serializa los cambios locales del registro y del puntero entre comandos.
            with closing(sqlite3.connect(MLOPS_DIR / "operations.db")) as lock:
                lock.execute("BEGIN IMMEDIATE")
                if args.command == "register":
                    result = {"candidate_version": register(args.run_id)}
                elif args.command == "promote":
                    result = promote(args.version)
                else:
                    result = rollback()
    except (ValueError, OSError, mlflow.exceptions.MlflowException, sqlite3.OperationalError) as error:
        parser.exit(1, f"MLOps: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
