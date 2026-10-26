import numpy as np
from sklearn.metrics import mean_absolute_error


def mean_absolute_scaled_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    sp: int = 1,
    multioutput='raw_values',
    eps: float = 1e-8,
) -> list[float]:
    """
    MASE consistent with Hyndman & Koehler and the sktime implementation
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    if y_train.size <= sp:
        raise ValueError(
            f"Length of y_train must be > sp. Got len={y_train.size}, sp={sp}."
        )

    # Seasonal naive on training: ŷ_t = y_{t-sp}
    y_train_clean = y_train[~np.isnan(y_train).any(axis=1)] # remove all nan values
    y_pred_naive_train = y_train_clean[:-sp]
    mae_naive = mean_absolute_error(y_train_clean[sp:], y_pred_naive_train, multioutput=multioutput)
   
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