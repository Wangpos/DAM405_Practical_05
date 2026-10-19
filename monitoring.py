"""
monitoring.py  —  Task 4 (monitoring signals) and Task 5 (drift check)
"""
from collections import deque

import numpy as np

from logger_config import get_logger

logger = get_logger()

# ---------------- Task 4: monitoring signals ----------------
# Simple in-memory counters. Latencies keep only the last 100 values (rolling window).
stats = {"requests": 0, "errors": 0, "latencies": deque(maxlen=100), "predictions": {}}


def record(latency_ms, ok=True, prediction=None):
    """Update counters after every request."""
    stats["requests"] += 1
    if not ok:
        stats["errors"] += 1
    stats["latencies"].append(latency_ms)
    if prediction is not None:
        # Count how often each class is predicted (helps spot prediction drift)
        stats["predictions"][prediction] = stats["predictions"].get(prediction, 0) + 1


def summary():
    """Log one summary line and return the numbers (also served by /stats)."""
    lat = stats["latencies"]
    avg = sum(lat) / len(lat) if lat else 0
    err_rate = stats["errors"] / stats["requests"] if stats["requests"] else 0
    logger.info(
        f"SUMMARY requests={stats['requests']} errors={stats['errors']} "
        f"error_rate={err_rate:.0%} avg_latency_ms={avg:.1f} "
        f"prediction_counts={stats['predictions']}"
    )
    return {
        "requests": stats["requests"],
        "errors": stats["errors"],
        "error_rate": round(err_rate, 3),
        "avg_latency_ms": round(avg, 2),
        "prediction_counts": stats["predictions"],
    }


# ---------------- Task 5: basic drift check ----------------
def drift_check(train_mean, live_values, tol=0.25, feature_name="feature"):
    """Compare the live mean of one feature with its training mean.
    Logs a WARNING if the relative change is greater than `tol` (25% by default)."""
    live_mean = float(np.mean(live_values))
    rel = abs(live_mean - train_mean) / (abs(train_mean) + 1e-9)  # 1e-9 avoids divide-by-zero
    if rel > tol:
        logger.warning(
            f"possible drift on {feature_name}: live={live_mean:.3f} "
            f"train={train_mean:.3f} change={rel:.0%}"
        )
    else:
        logger.info(
            f"no drift on {feature_name}: live={live_mean:.3f} "
            f"train={train_mean:.3f} change={rel:.0%}"
        )
    return rel
