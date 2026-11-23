"""Entrenamiento compartido por el pipeline y el notebook con seguimiento opcional."""

from __future__ import annotations

import hashlib
import os

from pandas.util import hash_pandas_object
from sklearn.metrics import classification_report, confusion_matrix

from src import config
from src.evaluate import compute_metrics

EXPERIMENT = "drybean-laboratorio"
LOCAL_DATA = config.RAW_XLSX.relative_to(config.ROOT_DIR).as_posix()
UCI_DATA = "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset"


def split_digest(features, labels) -> str:
    """Identifica contenido, orden de filas, etiquetas y esquema de una partición."""
    frame = features.astype("float64").assign(**{config.TARGET: labels.to_numpy()})
    digest = hashlib.sha256(",".join(frame.columns).encode("utf-8"))
    digest.update(hash_pandas_object(frame, index=False).to_numpy(dtype="<u8").tobytes())
    return digest.hexdigest()


def log_training(
    model, X_train, y_train, X_test, y_test, predictions, metrics, *,
    run_name: str, source: str, dataset_source: str, tracking_uri: str, tuning=None,
) -> str:
    """Guarda un modelo entrenado y devuelve el ID de su run."""
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        tags = {"source": source, "dataset_source": dataset_source}
        if dataset_source == LOCAL_DATA:
            tags["dataset_sha256"] = hashlib.sha256(config.RAW_XLSX.read_bytes()).hexdigest()
        for context, features, labels in (
            ("training", X_train, y_train), ("evaluation", X_test, y_test),
        ):
            fingerprint = split_digest(features, labels)
            tags[f"{context}_sha256"] = fingerprint
            partition = mlflow.data.from_pandas(
                features.assign(**{config.TARGET: labels.to_numpy()}),
                source=dataset_source,
                targets=config.TARGET,
                name=f"drybean-{context}",
                digest=fingerprint[:32],
            )
            mlflow.log_input(partition, context=context)
        mlflow.set_tags(tags)
        params = {key: value for key, value in model.get_params().items()
                  if value is None or isinstance(value, (str, int, float, bool))}
        mlflow.log_params({
            **params,
            "model": type(model).__name__,
            "split_random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        })
        mlflow.log_metrics({f"{name}_test": float(value) for name, value in metrics.items()})
        if tuning is not None:
            mlflow.log_metric("f1_macro_cv_best", float(tuning.best_score_))
            mlflow.log_dict({"best_params": tuning.best_params_, "param_grid": tuning.param_grid}, "grid_search.json")
        mlflow.log_dict(
            classification_report(y_test, predictions, output_dict=True, zero_division=0),
            "classification_report.json",
        )
        mlflow.log_dict({
            "classes": model.classes_.tolist(),
            "matrix": confusion_matrix(y_test, predictions, labels=model.classes_).tolist(),
        }, "confusion_matrix.json")
        example = X_train.head(2).astype("float64")
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(example, model.predict(example)),
            input_example=example,
        )
    print(f"MLflow run: {tracking_uri.rstrip('/')}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
    return run.info.run_id


def fit_and_evaluate(
    model, X_train, y_train, X_test, y_test, *, run_name: str, source: str,
    tracking_uri: str | None = None, dataset_source: str = LOCAL_DATA,
):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        log_training(
            model, X_train, y_train, X_test, y_test, predictions, metrics,
            run_name=run_name, source=source, dataset_source=dataset_source, tracking_uri=uri,
        )
    return model, predictions, metrics


def log_grid_search(
    grid, X_train, y_train, X_test, y_test, *, source: str,
    tracking_uri: str | None = None, dataset_source: str = LOCAL_DATA,
) -> None:
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return
    model = grid.best_estimator_
    predictions = model.predict(X_test)
    log_training(
        model, X_train, y_train, X_test, y_test, predictions, compute_metrics(y_test, predictions),
        run_name="random-forest-grid-search", source=source, dataset_source=dataset_source,
        tracking_uri=uri, tuning=grid,
    )
