"""Vector-wind error metrics for AMV validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


METRIC_COLUMNS = (
    "u_bias",
    "v_bias",
    "signed_speed_bias",
    "mean_absolute_speed_error",
    "speed_rmse",
    "mean_vector_difference",
    "vector_difference_rmse",
    "mean_absolute_direction_difference",
    "mean_signed_direction_difference",
    "speed_correlation",
)


def direction_difference(u1, v1, u2, v2) -> np.ndarray:
    """Return signed angular difference in degrees in [-180, 180]."""

    dot = np.asarray(u1) * np.asarray(u2) + np.asarray(v1) * np.asarray(v2)
    cross = np.asarray(u2) * np.asarray(v1) - np.asarray(v2) * np.asarray(u1)
    return np.degrees(np.arctan2(cross, dot))


def _safe_corr(x, y) -> float:
    sx = pd.Series(np.asarray(x, dtype=float))
    sy = pd.Series(np.asarray(y, dtype=float))
    valid = np.isfinite(sx) & np.isfinite(sy)
    sx = sx[valid]
    sy = sy[valid]
    if len(sx) <= 1 or sx.std() <= 0 or sy.std() <= 0:
        return np.nan
    return float(sx.corr(sy))


def summary_metrics(
    u1,
    v1,
    u2,
    v2,
    *,
    subset: str = "all_valid",
    comparison: str = "comparison",
) -> dict:
    """Compute compact AMV-style vector-wind error metrics.

    Parameters
    ----------
    u1, v1:
        First wind vector, normally AMV/PIV.
    u2, v2:
        Reference wind vector, normally ERA5, radiosonde, or official AMV.

    Returned metrics are intentionally limited to commonly used AMV validation
    errors: component bias, speed bias/error, vector difference, RMSVD,
    direction error, and speed correlation.
    """

    frame = pd.DataFrame({"u1": u1, "v1": v1, "u2": u2, "v2": v2}).dropna()
    output = {"subset": subset, "comparison": comparison, "sample_count": len(frame)}
    if frame.empty:
        output.update({name: np.nan for name in METRIC_COLUMNS})
        return output

    u1 = frame.u1.astype(float)
    v1 = frame.v1.astype(float)
    u2 = frame.u2.astype(float)
    v2 = frame.v2.astype(float)

    du = u1 - u2
    dv = v1 - v2
    speed1 = np.hypot(u1, v1)
    speed2 = np.hypot(u2, v2)
    speed_error = speed1 - speed2
    vector_difference = np.hypot(du, dv)
    direction_error = direction_difference(u1, v1, u2, v2)

    output.update(
        {
            "u_bias": float(np.nanmean(du)),
            "v_bias": float(np.nanmean(dv)),
            "signed_speed_bias": float(np.nanmean(speed_error)),
            "mean_absolute_speed_error": float(np.nanmean(np.abs(speed_error))),
            "speed_rmse": float(np.sqrt(np.nanmean(speed_error**2))),
            "mean_vector_difference": float(np.nanmean(vector_difference)),
            # RMSVD: root-mean-square vector difference.
            "vector_difference_rmse": float(np.sqrt(np.nanmean(vector_difference**2))),
            "mean_absolute_direction_difference": float(np.nanmean(np.abs(direction_error))),
            "mean_signed_direction_difference": float(np.nanmean(direction_error)),
            "speed_correlation": _safe_corr(speed1, speed2),
        }
    )
    return output
