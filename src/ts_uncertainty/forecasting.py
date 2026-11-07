"""One wrapper around the forecasting models compared in the thesis, with a unified output.

`ForecastModel(name, ...)` loads one of `MODELS` through its original library
(statsforecast, chronos-forecasting, tirex-2, timestral) and `forecast(samples)` turns
Samples into a `QuantileForecast`: quantiles of shape (B, Q, H) on a common grid
of quantile levels, plus the context and ground truth, so every model's output can go
straight into the metrics and `plot_sample`.

    model = ForecastModel("timestral", size="28M")
    fc = model.forecast([gen.sample() for _ in range(8)])   # or a single sample
    plot_sample(s, prediction=fc.quantiles[0], quantiles=fc.quantile_levels)
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from ts_uncertainty.sample import (
    Sample,
    samples_to_chronos2_inputs,
    samples_to_sf_inputs,
    samples_to_timestral_inputs,
    samples_to_tirex2_inputs,
)

MODELS = ("autoarima", "chronos2", "tirex2", "timestral")
TIMESTRAL_SIZES = ("5M", "28M", "123M", "209M")
# TiRex-2's native levels; every other model outputs these too.
DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class QuantileForecast:
    """Unified forecast of B univariate series.

    quantiles:       (B, Q, H) forecast values, non-decreasing along Q
    quantile_levels: (Q,)      probability levels, contains 0.5
    context:         (B, T)    history the forecast was made from
    target:          (B, H)    ground truth over the horizon
    """

    model: str
    quantiles: np.ndarray
    quantile_levels: np.ndarray
    context: np.ndarray
    target: np.ndarray

    @property
    def median(self) -> np.ndarray:
        """(B, H) median forecast, used as the point forecast."""
        return self.quantile(0.5)

    def quantile(self, level: float) -> np.ndarray:
        """(B, H) forecast at one of the quantile levels."""
        hits = np.flatnonzero(np.isclose(self.quantile_levels, level))
        if len(hits) == 0:
            raise ValueError(f"level {level} is not one of {self.quantile_levels.tolist()}")
        return self.quantiles[:, hits[0]]

    def interval(self, coverage: float) -> tuple[np.ndarray, np.ndarray]:
        """(B, H) lower and upper bound of the central interval, e.g. 0.8 = q0.1–q0.9."""
        return self.quantile((1 - coverage) / 2), self.quantile((1 + coverage) / 2)

    def metric_inputs(self) -> dict:
        """Arrays in the time-first layout of `metrics.py` (rows = time steps, columns = series).

        y_true: (H, B), y_pred: (H, B) median, y_quantiles: (H, B, Q), y_train: (T, B),
        quantiles: list of the Q levels. Usage:
            m = fc.metric_inputs()
            mean_absolute_scaled_error(m["y_true"], m["y_pred"], m["y_train"])
            quantile_loss(m["y_true"], m["y_quantiles"], m["quantiles"])
        """
        return {
            "y_true": self.target.T,
            "y_pred": self.median.T,
            "y_quantiles": self.quantiles.transpose(2, 0, 1),
            "y_train": self.context.T,
            "quantiles": self.quantile_levels.tolist(),
        }


class ForecastModel:
    """Load one forecasting model by name and forecast Samples with it.

    Args:
        name: one of MODELS.
        size: Timestral checkpoint size, one of TIMESTRAL_SIZES; ignored by the other models.
        quantile_levels: common levels picked out of every forecast; each must be one of the
            model's native levels, otherwise forecast raises a ValueError. None keeps the
            model's native levels (AutoARIMA then uses DEFAULT_QUANTILE_LEVELS).
        device: torch device for the neural models, defaults to cuda if available.
        checkpoint: override the Hugging Face repo id / local path of a neural model.
        season_length: AutoARIMA season length for all samples; None uses each sample's own
            Sample.season_length (samples with different season lengths are fitted in groups).
        batch_size: series per forward pass of the neural models.
    """

    def __init__(
        self,
        name: str,
        size: str = "209M",
        quantile_levels: Sequence[float] | None = DEFAULT_QUANTILE_LEVELS,
        device: str | None = None,
        checkpoint: str | None = None,
        season_length: int | None = None,
        batch_size: int = 256,
    ) -> None:
        if name not in MODELS:
            raise ValueError(f"name must be one of {MODELS}, got {name!r}")
        if name == "timestral" and size not in TIMESTRAL_SIZES:
            raise ValueError(f"Timestral size must be one of {TIMESTRAL_SIZES}, got {size!r}")
        if quantile_levels is not None:
            q = np.asarray(quantile_levels, dtype=np.float64)
            if q.ndim != 1 or np.any(q <= 0) or np.any(q >= 1) or np.any(np.diff(q) <= 0):
                raise ValueError(f"quantile_levels must be strictly increasing in (0, 1), got {quantile_levels}")
            if not np.isclose(q, 0.5).any():
                raise ValueError("quantile_levels must contain the median 0.5")
        if season_length is not None and season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.name = name
        self.size = size if name == "timestral" else None
        self.quantile_levels = None if quantile_levels is None else tuple(float(level) for level in quantile_levels)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.season_length = season_length
        self.batch_size = batch_size
        self.checkpoint = checkpoint or {
            "autoarima": None,
            "chronos2": "amazon/chronos-2",
            "tirex2": "NX-AI/TiRex-2",
            "timestral": f"mistralai/Timestral-{size}",
        }[name]
        self.model = self._load()

    def __repr__(self) -> str:
        return f"ForecastModel({self.name!r}, checkpoint={self.checkpoint!r}, device={self.device!r})"

    @property
    def config(self):
        """The native model config (None for AutoARIMA)."""
        return {
            "autoarima": lambda: None,
            "chronos2": lambda: self.model.model.chronos_config,
            "tirex2": lambda: self.model.model.stack_config,
            "timestral": lambda: self.model.config,
        }[self.name]()

    @property
    def native_quantile_levels(self) -> tuple[float, ...]:
        """Quantile levels the model outputs itself."""
        if self.name == "autoarima":
            return DEFAULT_QUANTILE_LEVELS
        if self.name == "chronos2":
            return tuple(self.model.quantiles)
        if self.name == "tirex2":
            return tuple(round(float(q), 6) for q in self.model.model.quantiles)
        return tuple(self.model.quantile_levels)

    def _load(self):
        if self.name == "autoarima":
            return None  # one StatsForecast per season length, built in _forecast_autoarima
        if self.name == "chronos2":
            from chronos import Chronos2Pipeline

            return Chronos2Pipeline.from_pretrained(self.checkpoint, device_map=self.device)
        if self.name == "tirex2":
            from tirex2 import load_model

            return load_model(self.checkpoint, device=self.device)
        from timestral import TimestralForecaster

        return TimestralForecaster.from_pretrained(self.checkpoint, device=self.device)

    def forecast(
        self,
        samples: Sample | Sequence[Sample],
        prediction_length: int | None = None,
        **predict_kwargs,
    ) -> QuantileForecast:
        """Forecast the horizon of each sample from its context (and covariates, if any).

        prediction_length defaults to the samples' horizon H (must be the same for all).
        predict_kwargs go to the underlying call, e.g. cross_learning (Chronos-2),
        tta_sign_flip (TiRex-2), symmetric_averaging (Timestral).
        """
        if isinstance(samples, Sample):
            samples = [samples]
        samples = list(samples)
        if not samples:
            raise ValueError("need at least one sample")
        horizons = {len(s.target) for s in samples}
        if len(horizons) != 1:
            raise ValueError(f"all samples need the same horizon, got {sorted(horizons)}")
        H = prediction_length or horizons.pop()
        levels = self.quantile_levels or self.native_quantile_levels

        quantiles = {  # (B, Q, H) at `levels`
            "autoarima": self._forecast_autoarima,
            "chronos2": self._forecast_chronos2,
            "tirex2": self._forecast_tirex2,
            "timestral": self._forecast_timestral,
        }[self.name](samples, H, levels, **predict_kwargs)

        T = min(len(s.context) for s in samples)  # contexts may differ in length; keep the common tail
        return QuantileForecast(
            model=self.name if self.size is None else f"{self.name}-{self.size}",
            quantiles=np.asarray(quantiles, dtype=np.float32),
            quantile_levels=np.asarray(levels, dtype=np.float64),
            context=np.stack([np.asarray(s.context[-T:], dtype=np.float32) for s in samples]),
            target=np.stack([np.asarray(s.target[:H], dtype=np.float32) for s in samples]),
        )

    def _forecast_autoarima(self, samples, H, levels, **predict_kwargs) -> np.ndarray:
        # Gaussian prediction intervals: quantile q is a bound of the (|1 - 2q|·100)% interval,
        # the median is the mean forecast.
        coverages = {}
        for q in levels:
            if not np.isclose(q, 0.5):
                c = abs(1 - 2 * q) * 100
                if not np.isclose(c, round(c)):
                    raise ValueError(f"AutoARIMA needs quantile levels on a 0.005 grid, got {q}")
                coverages[q] = round(c)
        from statsforecast import StatsForecast
        from statsforecast.models import AutoARIMA

        columns = [
            "AutoARIMA" if q not in coverages else f"AutoARIMA-{'lo' if q < 0.5 else 'hi'}-{coverages[q]}"
            for q in levels
        ]
        seasons = np.array([self.season_length or s.season_length for s in samples])  # (B,)
        out = np.empty((len(samples), len(levels), H))  # (B, Q, H)
        for m in np.unique(seasons):
            idx = np.flatnonzero(seasons == m)
            df, X_df, _ = samples_to_sf_inputs([samples[i] for i in idx])
            sf = StatsForecast(models=[AutoARIMA(season_length=int(m))], freq=1)
            fc = sf.forecast(df=df, h=H, X_df=X_df, level=sorted(set(coverages.values())), **predict_kwargs)
            # statsforecast sorts unique_id as strings ("0", "1", "10", ...): restore sample order
            fc = fc.sort(fc["unique_id"].cast(int), "ds")
            values = fc.select(columns).to_numpy()  # (B_m * H, Q)
            out[idx] = values.reshape(len(idx), H, len(levels)).transpose(0, 2, 1)
        return out

    def _forecast_chronos2(self, samples, H, levels, **predict_kwargs) -> np.ndarray:
        inputs, _ = samples_to_chronos2_inputs(samples)
        predict_kwargs.setdefault("batch_size", self.batch_size)
        out = self.model.predict(inputs, prediction_length=H, **predict_kwargs)  # B x (1, Q, H)
        quantiles = torch.stack([o[0] for o in out]).float().cpu().numpy()
        return select_quantiles(quantiles, self.native_quantile_levels, levels)

    def _forecast_tirex2(self, samples, H, levels, **predict_kwargs) -> np.ndarray:
        inputs, _ = samples_to_tirex2_inputs(samples)
        predict_kwargs.setdefault("batch_size", self.batch_size)
        out = self.model.forecast(inputs, prediction_length=H, output_type="numpy", **predict_kwargs)  # B x (1, Q, H)
        return select_quantiles(np.stack([o[0] for o in out]), self.native_quantile_levels, levels)

    def _forecast_timestral(self, samples, H, levels, **predict_kwargs) -> np.ndarray:
        context, past, known, _ = samples_to_timestral_inputs(samples)  # (B, 1, T), None, (B, V, T + H)
        chunks = []
        for i in range(0, len(context), self.batch_size):
            batch = slice(i, i + self.batch_size)
            out = self.model.forecast(
                context[batch],
                prediction_length=H,
                past_covariates=past,
                known_covariates=None if known is None else known[batch],
                **predict_kwargs,
            )
            chunks.append(out.quantiles[:, 0].transpose(1, 2).float().cpu().numpy())  # (b, Q, H)
        return select_quantiles(np.concatenate(chunks), self.native_quantile_levels, levels)


def select_quantiles(
    quantiles: np.ndarray, levels: Sequence[float], new_levels: Sequence[float]
) -> np.ndarray:
    """Pick the (B, Q', H) quantiles at `new_levels` out of (B, Q, H) quantiles at `levels`."""
    levels = np.asarray(levels, dtype=np.float64)
    index = []
    for level in new_levels:
        hits = np.flatnonzero(np.isclose(levels, level))
        if len(hits) == 0:
            raise ValueError(f"quantile level {level} is not in the model output {levels.tolist()}")
        index.append(hits[0])
    return quantiles[:, index]
