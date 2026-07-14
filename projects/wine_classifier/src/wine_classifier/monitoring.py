"""Small in-process monitoring example with explicit limitations."""
from __future__ import annotations

from collections import Counter
from threading import Lock
from typing import Any


class RequestMonitor:
    def __init__(self, reference_statistics: dict[str, dict[str, float]]) -> None:
        self._reference = reference_statistics
        self._requests = 0
        self._predictions: Counter[str] = Counter()
        self._feature_warnings: Counter[str] = Counter()
        self._lock = Lock()

    def record(self, features: dict[str, float], predicted_class: str) -> list[str]:
        warnings = []
        for name, value in features.items():
            stats = self._reference[name]
            std = stats["std"]
            z_score = 0.0 if std == 0 else abs(value - stats["mean"]) / std
            if value < stats["min"] or value > stats["max"]:
                warnings.append(f"{name} is outside the training range")
            elif z_score > 4:
                warnings.append(f"{name} is more than four reference standard deviations away")

        with self._lock:
            self._requests += 1
            self._predictions[predicted_class] += 1
            for warning in warnings:
                self._feature_warnings[warning.split(" is ", 1)[0]] += 1
        return warnings

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            total = self._requests
            distribution = {
                name: count / total for name, count in self._predictions.items()
            } if total else {}
            return {
                "requests": total,
                "prediction_distribution": distribution,
                "feature_warning_counts": dict(self._feature_warnings),
                "limitations": (
                    "In-process counters reset on restart and are not a replacement for "
                    "durable telemetry, delayed-label performance, or statistical drift tests."
                ),
            }
