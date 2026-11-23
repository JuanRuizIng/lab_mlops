"""Registra resultados de inferencia local sin guardar las medidas de entrada."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src import config


def record_predictions(
    predictions: list[dict],
    *,
    source: str,
    model_version: str | None = None,
    request_id: str | None = None,
    log_path: Path = config.ROOT_DIR / "logs" / "predictions.jsonl",
) -> None:
    """Añade una línea JSON por resultado; un ID común agrupa cada lote."""
    if not predictions:
        return

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    batch_id = request_id or str(uuid4())
    lines = []
    for index, prediction in enumerate(predictions):
        lines.append(json.dumps({
            "timestamp_utc": timestamp,
            "source": source,
            "batch_id": batch_id,
            "index": index,
            "predicted_class": prediction["predicted_class"],
            "confidence": float(prediction["confidence"]),
            "model_version": model_version,
        }, ensure_ascii=False))

    with log_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
