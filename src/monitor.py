"""Resume predicciones locales y compara un lote nuevo con la referencia."""

from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import accuracy_score, f1_score

from deploy.web.api.schemas import BeanFeatures
from src import config

MIN_ROWS = 30
SHARE_THRESHOLD = 0.10
KS_THRESHOLD = 0.20


def observed_predictions(path: Path, model_version: str, classes: list[str]) -> list[tuple[str, float]]:
    if not path.exists():
        return []
    observations = []
    with path.open(encoding="utf-8") as log:
        for line in log:
            try:
                event = json.loads(line)
                if not isinstance(event, dict) or event.get("model_version") != model_version:
                    continue
                label = event["predicted_class"]
                confidence = float(event["confidence"])
                if label in classes and math.isfinite(confidence) and 0 <= confidence <= 1:
                    observations.append((label, confidence))
            except (KeyError, TypeError, ValueError):
                continue
    return observations


def evaluate_csv(csv_path: Path, bundle, X_train) -> dict:
    incoming = pd.read_csv(csv_path, encoding="utf-8")
    labels = incoming.pop(config.TARGET) if config.TARGET in incoming else None
    if incoming.empty or set(incoming.columns) != set(bundle.info.features):
        raise ValueError("El CSV debe incluir datos y las 16 columnas de la API")
    features = pd.DataFrame(
        [BeanFeatures.model_validate(row).model_dump() for row in incoming.to_dict("records")],
        columns=bundle.info.features,
    )
    if labels is not None and not labels.isin(bundle.info.classes).all():
        raise ValueError("La columna Class tiene etiquetas inválidas o vacías")
    distances = {
        name: round(float(ks_2samp(X_train[name], features[name], method="asymp").statistic), 4)
        for name in features
    }
    flagged = [name for name, statistic in distances.items() if statistic > KS_THRESHOLD]
    status = "insufficient_data" if len(features) < MIN_ROWS else ("review" if flagged else "no_signal")
    report = {"rows": len(features), "feature_shift_status": status, "ks_by_feature": distances}
    if len(features) >= MIN_ROWS:
        report["features_to_review"] = flagged
    if labels is not None:
        predictions = bundle.model.predict(features)
        report["labeled_performance"] = {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
            "rows": len(labels),
            "classes_present": sorted(labels.unique()),
        }
    return report


def build_report(bundle, X_train, X_test, csv_path: Path | None = None) -> dict:
    events = observed_predictions(
        config.ROOT_DIR / "logs" / "predictions.jsonl", bundle.info.model_version, bundle.info.classes,
    )
    reference = Counter(bundle.model.predict(X_test))
    observed = Counter(label for label, _ in events)
    count = len(events)
    differences = {
        name: round(observed[name] / count - reference[name] / len(X_test), 4)
        for name in bundle.info.classes
    } if count else {}
    status = "insufficient_data"
    if count >= MIN_ROWS:
        status = "review" if any(abs(delta) > SHARE_THRESHOLD for delta in differences.values()) else "no_signal"
    report = {
        "model_version": bundle.info.model_version,
        "prediction_count": count,
        "mean_confidence": round(mean(confidence for _, confidence in events), 4) if count else None,
        "low_confidence_rate_below_0_8": round(mean(confidence < 0.8 for _, confidence in events), 4) if count else None,
        "class_share_delta_vs_holdout": differences,
        "output_mix_status": status,
    }
    if csv_path is not None:
        report["csv"] = evaluate_csv(csv_path, bundle, X_train)
    return report
