"""One wrapper around the forecasting models compared in the thesis, with a unified output.

`ForecastModel(name, ...)` loads one of `MODELS` through its original library
(statsforecast, chronos-forecasting, tirex-2, timestral, timesfm,
continuousintelligence) and `forecast(samples)` turns
Samples into a `QuantileForecast`: quantiles of shape (B, Q, H) on a common grid
of quantile levels, plus the context and ground truth, so every model's output can go
straight into the metrics and `plot_sample`.

    model = ForecastModel("timestral", size="28M")
    fc = model.forecast([gen.sample() for _ in range(8)])   # or a single sample
    plot_sample(s, prediction=fc.quantiles[0], quantiles=fc.quantile_levels)
"""

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ts_uncertainty.sample import (
    Sample,
    samples_to_ciproto_inputs,
    samples_to_chronos2_inputs,
    samples_to_sf_inputs,
    samples_to_timesfm3_inputs,
    samples_to_timestral_inputs,
    samples_to_tirex2_inputs,
)

# statsforecast models: ForecastModel name -> statsforecast class (= its output column)
STAT_MODELS = {"autoarima": "AutoARIMA", "autotheta": "AutoTheta", "seasonalnaive": "SeasonalNaive"}
MODELS = (*STAT_MODELS, "chronos2", "tirex2", "timestral", "timesfm3", "ciproto")
TIMESTRAL_SIZES = ("5M", "28M", "123M", "209M")
# TiRex-2's native levels; every other model outputs these too.
DEFAULT_QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _use_venv_cuda_toolkit() -> None:
    """Build torch extensions (TiRex-2's flashrnn sLSTM kernel) with the venv's CUDA 13 toolkit.

    Without CUDA_HOME, torch falls back to the system nvcc (/usr/bin, CUDA 12) and flashrnn links
    the system libcudart.so.12 next to torch's CUDA 13 runtime. Once ci-proto has been loaded,
    TileLang's global libcudart stub then serves those CUDA 12 calls, can't resolve them and
    aborts the process (kills the Jupyter kernel). flashrnn links with `-L<CUDA_HOME>/lib
    -lcudart -lcublas`, so the pip toolkit also needs the unversioned symlinks a CUDA install has.
    """
    if os.environ.get("CUDA_HOME") or sys.platform != "linux":
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in map(Path, nvidia.__path__):
        home = root / "cu13"
        if not (home / "bin" / "nvcc").is_file():
            continue
        for name in ("libcudart.so", "libcublas.so"):
            link = home / "lib" / name
            if not link.exists() and (home / "lib" / f"{name}.13").exists():
                try:
                    link.symlink_to(f"{name}.13")
                except OSError:
                    return  # read-only venv: keep torch's default toolkit
        os.environ["CUDA_HOME"] = str(home)
        if "torch.utils.cpp_extension" in sys.modules:  # it reads CUDA_HOME once, at import
            sys.modules["torch.utils.cpp_extension"].CUDA_HOME = str(home)
        return


_use_venv_cuda_toolkit()


@dataclass
class QuantileForecast:
    """Unified forecast of B univariate series.

    quantiles:       (B, Q, H) forecast values, non-decreasing along Q
    quantile_levels: (Q,)      probability levels, contains 0.5
    context:         (B, T)    history the forecast was made from
    target:          (B, H)    ground truth over the horizon
    season_length:   (B,)      seasonal period of each series (Sample.season_length) for scaled metrics
    """

    model: str
    quantiles: np.ndarray
    quantile_levels: np.ndarray
    context: np.ndarray
    target: np.ndarray
    season_length: np.ndarray | None = None

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
        quantiles: list of the Q levels, season_length: (B,) seasonal period per series. Usage:
            m = fc.metric_inputs()
            mean_absolute_scaled_error(m["y_true"], m["y_pred"], m["y_train"], sp=m["season_length"])
            scaled_quantile_loss(m["y_true"], m["y_quantiles"], m["y_train"], m["quantiles"], m["season_length"])
        """
        return {
            "y_true": self.target.T,
            "y_pred": self.median.T,
            "y_quantiles": self.quantiles.transpose(2, 0, 1),
            "y_train": self.context.T,
            "quantiles": self.quantile_levels.tolist(),
            "season_length": np.ones(len(self.target), dtype=int) if self.season_length is None else self.season_length,
        }


class ForecastModel:
    """Load one forecasting model by name and forecast Samples with it.

    Args:
        name: one of MODELS.
        size: Timestral checkpoint size, one of TIMESTRAL_SIZES; ignored by the other models.
        quantile_levels: common levels picked out of every forecast; each must be one of the
            model's native levels, otherwise forecast raises a ValueError. None keeps the
            model's native levels (the statistical models then use DEFAULT_QUANTILE_LEVELS).
        device: torch device for the neural models, defaults to cuda if available.
        checkpoint: override the Hugging Face repo id / local path of a neural model.
        season_length: season length of the statistical models (STAT_MODELS) for all samples;
            None uses each sample's own
            Sample.season_length (samples with different season lengths are fitted in groups).
        n_jobs: statistical-model worker processes (one series per process), -1 = all CPU cores.
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
        n_jobs: int = -1,
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
        if n_jobs == 0 or n_jobs < -1:
            raise ValueError(f"n_jobs must be -1 or >= 1, got {n_jobs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.name = name
        self.size = size if name == "timestral" else None
        self.quantile_levels = None if quantile_levels is None else tuple(float(level) for level in quantile_levels)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.season_length = season_length
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.checkpoint = checkpoint or {
            **dict.fromkeys(STAT_MODELS),
            "chronos2": "amazon/chronos-2",
            "tirex2": "NX-AI/TiRex-2",
            "timestral": f"mistralai/Timestral-{size}",
            "timesfm3": "google/timesfm-3.0-pytorch",
            "ciproto": "ContinuousIntelligence/ci-proto",
        }[name]
        self.model = self._load()

    def __repr__(self) -> str:
        return f"ForecastModel({self.name!r}, checkpoint={self.checkpoint!r}, device={self.device!r})"

    @property
    def config(self):
        """The native model config (None for the statistical models)."""
        if self.name in STAT_MODELS:
            return None
        return {
            "chronos2": lambda: self.model.model.chronos_config,
            "tirex2": lambda: self.model.model.stack_config,
            "timestral": lambda: self.model.config,
            "timesfm3": lambda: self.model.config,
            "ciproto": lambda: self.model.info,
        }[self.name]()

    @property
    def native_quantile_levels(self) -> tuple[float, ...]:
        """Quantile levels the model outputs itself."""
        if self.name in STAT_MODELS:
            return DEFAULT_QUANTILE_LEVELS
        if self.name == "chronos2":
            return tuple(self.model.quantiles)
        if self.name == "tirex2":
            return tuple(round(float(q), 6) for q in self.model.model.quantiles)
        if self.name == "timesfm3":
            return tuple(round(float(q), 6) for q in self.model.config.quantiles)
        if self.name == "ciproto":
            return tuple(round(float(q), 6) for q in self.model.quantiles)
        return tuple(self.model.quantile_levels)

    def _load(self):
        if self.name in STAT_MODELS:
            return None  # one StatsForecast per season length, built in _forecast_statsforecast
        if self.name == "chronos2":
            from chronos import Chronos2Pipeline

            return Chronos2Pipeline.from_pretrained(self.checkpoint, device_map=self.device)
        if self.name == "tirex2":
            from tirex2 import load_model

            return load_model(self.checkpoint, device=self.device)
        if self.name == "timesfm3":
            from timesfm3 import ModelConfig, TimesFM3Evaluator

            config = ModelConfig(checkpoint_path=self.checkpoint, per_core_batch_size=self.batch_size, device=self.device)
            return TimesFM3Evaluator(config)
        if self.name == "ciproto":
            import continuousintelligence as ci

            model = ci.load_model(self.checkpoint, device=self.device)
            # Its tta.json clamps forecasts at 0 when the context is non-negative and rounds
            # them when it is integer-valued; turn both off like make_positive for TimesFM-3.
            model._tta = {**model._tta, "clip_nonnegative": 0, "integer_round": 0}
            return model
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
        tta_sign_flip (TiRex-2), symmetric_averaging (Timestral),
        use_symmetric_averaging / univariate (TimesFM-3),
        categorical_future (ci-proto).
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
            **dict.fromkeys(STAT_MODELS, self._forecast_statsforecast),
            "chronos2": self._forecast_chronos2,
            "tirex2": self._forecast_tirex2,
            "timestral": self._forecast_timestral,
            "timesfm3": self._forecast_timesfm3,
            "ciproto": self._forecast_ciproto,
        }[self.name](samples, H, levels, **predict_kwargs)

        T = min(len(s.context) for s in samples)  # contexts may differ in length; keep the common tail
        return QuantileForecast(
            model=self.name if self.size is None else f"{self.name}-{self.size}",
            quantiles=np.asarray(quantiles, dtype=np.float32),
            quantile_levels=np.asarray(levels, dtype=np.float64),
            context=np.stack([np.asarray(s.context[-T:], dtype=np.float32) for s in samples]),
            target=np.stack([np.asarray(s.target[:H], dtype=np.float32) for s in samples]),
            season_length=np.array([s.season_length for s in samples]),
        )

    def _forecast_statsforecast(self, samples, H, levels, **predict_kwargs) -> np.ndarray:
        # statsforecast returns central prediction intervals: quantile q is a bound of the
        # (|1 - 2q|·100)% interval, the median is the point forecast. Covariates are only used
        # by AutoARIMA (as regressors); AutoTheta and SeasonalNaive ignore them.
        coverages = {}
        for q in levels:
            if not np.isclose(q, 0.5):
                c = abs(1 - 2 * q) * 100
                if not np.isclose(c, round(c)):
                    raise ValueError(f"{self.name} needs quantile levels on a 0.005 grid, got {q}")
                coverages[q] = round(c)
        import statsforecast.models
        from statsforecast import StatsForecast

        alias = STAT_MODELS[self.name]
        model_cls = getattr(statsforecast.models, alias)
        columns = [alias if q not in coverages else f"{alias}-{'lo' if q < 0.5 else 'hi'}-{coverages[q]}" for q in levels]
        seasons = np.array([self.season_length or s.season_length for s in samples])  # (B,)
        out = np.empty((len(samples), len(levels), H))  # (B, Q, H)
        for m in np.unique(seasons):
            idx = np.flatnonzero(seasons == m)
            df, X_df, _ = samples_to_sf_inputs([samples[i] for i in idx])
            sf = StatsForecast(models=[model_cls(season_length=int(m))], freq=1, n_jobs=self.n_jobs)
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

    def _forecast_timesfm3(self, samples, H, levels, **predict_kwargs) -> np.ndarray:
        contexts, known, _ = samples_to_timesfm3_inputs(samples)  # B x (T,) or (1, T), B x (V, T + H)
        # TimesFM3Evaluator defaults to benchmark settings; make_positive would clamp the
        # (z-normalised, often negative) series at 0, so turn it off unless asked for.
        predict_kwargs.setdefault("make_positive", False)
        predict_kwargs.setdefault("use_symmetric_averaging", False)
        out = self.model.predict_batch(
            contexts=contexts,
            horizon=H,
            past_future_covariates=known,
            return_quantiles=True,
            **predict_kwargs,
        )  # B x quantiles (H, Q), or (1, H, Q) with covariates
        native = self.native_quantile_levels
        quantiles = np.stack([np.asarray(o.quantiles).reshape(H, len(native)).T for o in out])  # (B, Q, H)
        return select_quantiles(quantiles, native, levels)

    def _forecast_ciproto(self, samples, H, levels, **predict_kwargs) -> np.ndarray:
        # All rows of one ci-proto call are forecast jointly and its `series_ids` mode pools
        # series (cross-learning), so every sample gets its own call to stay independent.
        histories, known, _ = samples_to_ciproto_inputs(samples)  # B x (1, T), B x (V, T + H)
        out = [
            self.model.forecast(x, horizon=H, future_covariates=k, **predict_kwargs)[0]  # (Q, H)
            for x, k in zip(histories, known)
        ]
        quantiles = torch.stack(out).float().cpu().numpy()  # (B, Q, H)
        return select_quantiles(quantiles, self.native_quantile_levels, levels)


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
