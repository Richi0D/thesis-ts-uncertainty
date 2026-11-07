import numpy as np
from sklearn.metrics import mean_absolute_error


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    sp: int | np.ndarray = 1,
    multioutput='raw_values',
    eps: float = 1e-8,
) -> list[float]:
    """
    MASE consistent with Hyndman & Koehler and the sktime implementation

    y_true, y_pred: shape (H, n_targets), y_train: shape (n_past, n_targets)
    sp: seasonal period of the naive scale, one int or one per target (n_targets,);
        see seasonal_scale for the fallbacks when that scale is missing or (near) zero
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    if y_train.shape[1:] != y_true.shape[1:]:
        raise ValueError("y_train must have the same number of targets as y_true")

    # Seasonal naive on training: ŷ_t = y_{t-sp}
    y_train_clean = y_train[~np.isnan(y_train).any(axis=1)] # remove all nan values
    if y_train_clean.shape[0] < 2:
        raise ValueError("y_train needs at least 2 rows without NaN")
    mae_naive = seasonal_scale(y_train_clean, sp)  # (n_targets,)
    if multioutput != "raw_values":
        mae_naive = np.average(mae_naive)
   
    # MAE of forecast (on horizon)
    mae_pred = mean_absolute_error(y_true, y_pred, multioutput=multioutput)

    # Avoid division by 0 (flat series => huge MASE)
    denom = np.maximum(mae_naive, eps)
    return mae_pred / denom


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles: list,
    multioutput: str = 'raw_values',
) -> np.ndarray:
    """
    Calculates the (pinball) quantile loss.

    y_true: shape (n_samples, n_targets)
    y_pred: shape (n_samples, n_targets, n_quantiles)
    quantiles: list of floats (e.g., [0.1, 0.2, ..., 0.9])
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    q = np.asarray(quantiles, dtype=float).reshape(1, 1, -1)

    if y_true.shape != y_pred.shape[:-1]:
        raise ValueError(
            f"y_true shape {y_true.shape} must match y_pred shape "
            f"{y_pred.shape} without the quantile dimension"
        )

    diff = y_true[..., np.newaxis] - y_pred  # shape: (n_samples, n_targets, n_quantiles)
    ql = np.maximum(q * diff, (q - 1) * diff)

    # average over quantiles, then over samples -> shape (n_targets,)
    ql_per_sample = ql.mean(axis=-1)
    output_errors = np.average(ql_per_sample, axis=0)

    if multioutput == "raw_values":
        return output_errors
    return np.average(output_errors)


def _naive_scale(y_past: np.ndarray, m: int) -> np.ndarray:
    """Mean absolute lag-m difference per target, NaN if the history is not longer than m."""
    if y_past.shape[0] <= m:
        return np.full(y_past.shape[1], np.nan)
    return np.nanmean(np.abs(y_past[m:] - y_past[:-m]), axis=0)  # NaNs in history are ignored


def seasonal_scale(
    y_past: np.ndarray,
    seasonal_period: int | np.ndarray = 1,
    rel_tol: float = 1e-3,
) -> np.ndarray:
    """
    In-sample mean absolute seasonal-naive error per target:
        a_j = 1/(T-m_j) * sum_{t=m_j+1}^{T} |y_{t,j} - y_{t-m_j,j}|

    Fallbacks per target, when a scale is missing (history not longer than m) or degenerate
    (<= rel_tol * std of the history, e.g. a noise-free periodic series that the seasonal naive
    forecasts perfectly), so a tiny forecast error doesn't blow up the scaled metric:
        seasonal naive (m) -> naive (m = 1) -> std of the history -> 1 (series scale ≈ 1)

    y_past: shape (n_past, n_targets), history BEFORE the forecast start
    seasonal_period: one int m for all targets or one per target, shape (n_targets,)
    rel_tol: a scale <= rel_tol * std(history) counts as degenerate; 0 disables the fallback
        (except for a scale of exactly 0)
    returns: shape (n_targets,), always > 0
    """
    y_past = np.asarray(y_past, dtype=float)
    n_targets = y_past.shape[1]
    periods = np.broadcast_to(np.asarray(seasonal_period, dtype=int), (n_targets,))
    if np.any(periods < 1):
        raise ValueError(f"seasonal_period must be >= 1, got {seasonal_period}")

    std = np.nanstd(y_past, axis=0) if y_past.shape[0] > 0 else np.full(n_targets, np.nan)  # (n_targets,)
    scale = np.full(n_targets, np.nan)
    for m in np.unique(periods):
        cols = periods == m
        scale[cols] = _naive_scale(y_past[:, cols], m)

    def degenerate(s):
        return np.isnan(s) | (s <= rel_tol * std)

    if (bad := degenerate(scale)).any():  # seasonal naive is perfect or undefined: plain naive
        scale[bad] = _naive_scale(y_past[:, bad], 1)
    if (bad := degenerate(scale)).any():  # naive is perfect too: spread of the history
        scale[bad] = std[bad]
    if (bad := degenerate(scale) | (scale <= 0)).any():  # constant history: no data scale at all
        scale[bad] = 1.0
    return scale


def scaled_quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    quantiles: list,
    seasonal_period: int | np.ndarray = 1,
    multioutput: str = "raw_values",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Scaled quantile loss (SQL), AutoGluon / fev-bench convention:
 
        SQL_j = 1/a_j * 1/H * sum_t 1/|Q| * sum_q 2 * rho_q(y_tj, f^q_tj)
 
    y_true: shape (H, n_targets)                 future values
    y_pred: shape (H, n_targets, n_quantiles)    quantile forecasts
    y_train: shape (n_past, n_targets)           history used for the scale
    seasonal_period: one int or one per target, shape (n_targets,), see seasonal_scale
    eps: lower bound of the scale a_j (only a guard, seasonal_scale is already > 0)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    q = np.asarray(quantiles, dtype=float).reshape(1, 1, -1)
 
    if y_true.shape != y_pred.shape[:-1]:
        raise ValueError(
            f"y_true shape {y_true.shape} must match y_pred shape "
            f"{y_pred.shape} without the quantile dimension"
        )
    if np.asarray(y_train).shape[1:] != y_true.shape[1:]:
        raise ValueError("y_train must have the same number of targets as y_true")
 
    diff = y_true[..., np.newaxis] - y_pred
    ql = 2.0 * np.maximum(q * diff, (q - 1) * diff)  # (H, n_targets, n_quantiles)
 
    # mean over quantiles and horizon (NaN targets in the horizon are skipped)
    ql_per_target = np.nanmean(ql.mean(axis=-1), axis=0)  # (n_targets,)
 
    scale = seasonal_scale(y_train, seasonal_period)
    output_errors = ql_per_target / np.maximum(scale, eps)
 
    if multioutput == "raw_values":
        return output_errors
    return np.nanmean(output_errors)