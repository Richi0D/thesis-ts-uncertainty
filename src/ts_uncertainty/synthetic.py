"""Synthetic univariate time series generator.

Each sample is built as

    y_t = level + slope * t + seasonal_t + noise_t + spikes_t

and split into a context window of length T and a forecast horizon of length H.
Optionally, a known-in-advance covariate describing the seasonal phase is returned
for the full window (T + H), so it is available over the forecast horizon as well.

Distribution shift (optional) changes one component abruptly at a random change
point tau, located anywhere in the window (context or horizon):

    level      y_t += delta                    for t >= tau
    trend      y_t += delta_slope * (t - tau)  for t >= tau
    noise      noise std scaled by a factor    for t >= tau
    amplitude  seasonal amplitude scaled       for t >= tau
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

SHIFT_TYPES = ("level", "trend", "noise", "amplitude")


@dataclass
class SyntheticSample:
    context: np.ndarray  # (T,)
    target: np.ndarray  # (H,)
    covariate: np.ndarray | None  # (T + H, 2 * n_periods) sin/cos of seasonal phase, or None
    periods: tuple[int, ...]  # seasonal periods used (empty if no seasonality)
    shift_point: int | None = None  # change point tau as index into the T + H window, or None
    shift_type: str | None = None  # one of SHIFT_TYPES, or None
    spike_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))  # indices into T + H


class SyntheticGenerator:
    """Infinite generator of synthetic time series.

    Args:
        context_length: T, number of observed time steps.
        horizon: H, number of time steps to forecast.
        seasonal: add a seasonal component.
        covariate: return sin/cos phase features of the seasonal periods (requires seasonal).
        noise: add Gaussian observation noise.
        spikes: add sparse random spikes (outliers).
        shift: apply a distribution shift at a random change point.
        periods: candidate seasonal periods; each sample draws n_periods of them.
        n_periods: number of superimposed seasonal patterns per sample.
        n_harmonics: Fourier harmonics per period (1 = pure sine, more = sharper shapes).
        level_range: range of the constant level.
        slope_range: range of the linear trend slope per time step.
        amplitude_range: range of the seasonal amplitude (per period).
        noise_std_range: range of the Gaussian observation noise std.
        spike_prob: per-time-step probability of a spike.
        spike_magnitude_range: range of the absolute spike height (sign is random).
        shift_types: shift types to draw from, subset of SHIFT_TYPES.
        shift_position_range: range of the change point as a fraction of T + H,
            e.g. (0.5, 1.0) covers the end of the context and the horizon.
        shift_level_range: range of the absolute level jump (sign is random).
        shift_slope_range: range of the absolute slope change (sign is random).
        shift_scale_range: range of the scale factor for noise / amplitude shifts;
            with probability 0.5 the reciprocal is used, so shifts go both up and down.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        context_length: int,
        horizon: int,
        seasonal: bool = True,
        covariate: bool = False,
        noise: bool = True,
        spikes: bool = False,
        shift: bool = False,
        periods: Sequence[int] = (7, 12, 24, 52),
        n_periods: int = 1,
        n_harmonics: int = 3,
        level_range: tuple[float, float] = (-1.0, 1.0),
        slope_range: tuple[float, float] = (-0.01, 0.01),
        amplitude_range: tuple[float, float] = (0.5, 2.0),
        noise_std_range: tuple[float, float] = (0.05, 0.3),
        spike_prob: float = 0.01,
        spike_magnitude_range: tuple[float, float] = (2.0, 5.0),
        shift_types: Sequence[str] = SHIFT_TYPES,
        shift_position_range: tuple[float, float] = (0.5, 1.0),
        shift_level_range: tuple[float, float] = (1.0, 3.0),
        shift_slope_range: tuple[float, float] = (0.01, 0.05),
        shift_scale_range: tuple[float, float] = (2.0, 4.0),
        seed: int | None = None,
    ):
        if covariate and not seasonal:
            raise ValueError("covariate=True requires seasonal=True")
        if seasonal and n_periods > len(periods):
            raise ValueError("n_periods must not exceed the number of candidate periods")
        if unknown := set(shift_types) - set(SHIFT_TYPES):
            raise ValueError(f"unknown shift types {unknown}, choose from {SHIFT_TYPES}")

        # shifts that need a component which is switched off are dropped
        shift_types = tuple(
            s for s in shift_types if not (s == "noise" and not noise) and not (s == "amplitude" and not seasonal)
        )
        if shift and not shift_types:
            raise ValueError("no applicable shift types for this configuration")

        self.context_length = context_length
        self.horizon = horizon
        self.seasonal = seasonal
        self.covariate = covariate
        self.noise = noise
        self.spikes = spikes
        self.shift = shift
        self.periods = tuple(periods)
        self.n_periods = n_periods
        self.n_harmonics = n_harmonics
        self.level_range = level_range
        self.slope_range = slope_range
        self.amplitude_range = amplitude_range
        self.noise_std_range = noise_std_range
        self.spike_prob = spike_prob
        self.spike_magnitude_range = spike_magnitude_range
        self.shift_types = shift_types
        self.shift_position_range = shift_position_range
        self.shift_level_range = shift_level_range
        self.shift_slope_range = shift_slope_range
        self.shift_scale_range = shift_scale_range
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[SyntheticSample]:
        return self

    def __next__(self) -> SyntheticSample:
        return self.sample()

    def sample(self, shift_point: float | None = None, shift_type: str | None = None) -> SyntheticSample:
        """Draw one sample.

        Passing shift_point or shift_type forces a shift, even if the generator has shift=False.
        An explicit shift_type may be any of SHIFT_TYPES, also one whose component is switched
        off: "noise" without noise adds noise only after the change point, "amplitude" without
        seasonality adds a seasonal pattern only after the change point.

        Args:
            shift_point: change point as a fraction of T + H in [0, 1]; drawn from
                shift_position_range if None.
            shift_type: one of SHIFT_TYPES; drawn from shift_types if None.
        """
        rng = self.rng
        length = self.context_length + self.horizon
        t = np.arange(length, dtype=np.float64)

        after = np.zeros(length, dtype=bool)  # mask of time steps after the change point
        if self.shift or shift_point is not None or shift_type is not None:
            if shift_point is None:
                lo, hi = (round(f * length) for f in self.shift_position_range)
                shift_point = int(rng.integers(lo, max(hi, lo + 1)))
            else:
                if not (0 <= shift_point <= 1):
                    raise ValueError(f"shift_point {shift_point} out of bounds. Valid range is [0, 1]")
                shift_point = round(shift_point * length)
            if shift_type is None:
                if not self.shift_types:
                    raise ValueError("no applicable shift types for this configuration, pass shift_type")
                shift_type = str(rng.choice(self.shift_types))
            elif shift_type not in SHIFT_TYPES:
                raise ValueError(f"shift_type {shift_type} not in {SHIFT_TYPES}")
            after = t >= shift_point

        level = rng.uniform(*self.level_range)
        slope = rng.uniform(*self.slope_range)
        y = level + slope * t
        if shift_type == "level":
            y += self._signed(self.shift_level_range) * after
        elif shift_type == "trend":
            y += self._signed(self.shift_slope_range) * (t - shift_point) * after

        periods: tuple[int, ...] = ()
        covariate = None
        if self.seasonal or shift_type == "amplitude":
            periods = tuple(int(p) for p in rng.choice(self.periods, size=self.n_periods, replace=False))
            # random time offset so samples don't all start at phase 0
            t_abs = t + rng.integers(0, max(periods))
            seasonal = np.zeros(length)
            for p in periods:
                seasonal += rng.uniform(*self.amplitude_range) * self._seasonal_shape(t_abs, p)
            if not self.seasonal:
                seasonal *= after  # seasonality switched off: pattern appears at the change point
            elif shift_type == "amplitude":
                seasonal *= np.where(after, self._scale_factor(), 1.0)
            y += seasonal
            if self.covariate:
                phase = 2 * np.pi * t_abs[:, None] / np.array(periods)[None, :]
                covariate = np.concatenate([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)

        if self.noise or shift_type == "noise":
            noise_std = np.full(length, rng.uniform(*self.noise_std_range))
            if not self.noise:
                noise_std *= after  # noise switched off: noise appears at the change point
            elif shift_type == "noise":
                noise_std *= np.where(after, self._scale_factor(), 1.0)
            y += rng.normal(0.0, noise_std)

        spike_idx = np.empty(0, dtype=np.int64)
        if self.spikes:
            spike_idx = np.flatnonzero(rng.random(length) < self.spike_prob)
            y[spike_idx] += [self._signed(self.spike_magnitude_range) for _ in spike_idx]

        y = y.astype(np.float32)

        return SyntheticSample(
            context=y[: self.context_length],
            target=y[self.context_length :],
            covariate=covariate,
            periods=periods,
            shift_point=shift_point,
            shift_type=shift_type,
            spike_idx=spike_idx,
        )

    def sample_batch(self, batch_size: int) -> dict[str, np.ndarray | None]:
        """Stack batch_size samples.

        Returns context (B, T), target (B, H), covariate (B, T + H, C) or None,
        and shift_point (B,) with -1 where no shift was applied.
        """
        samples = [self.sample() for _ in range(batch_size)]
        return {
            "context": np.stack([s.context for s in samples]),
            "target": np.stack([s.target for s in samples]),
            "covariate": np.stack([s.covariate for s in samples]) if self.covariate else None,
            "shift_point": np.array([-1 if s.shift_point is None else s.shift_point for s in samples]),
        }

    def _seasonal_shape(self, t: np.ndarray, period: int) -> np.ndarray:
        """Random periodic shape from a truncated Fourier series, normalised to unit max amplitude."""
        shape = np.zeros_like(t)
        for k in range(1, self.n_harmonics + 1):
            a, b = self.rng.normal(size=2) / k  # decay higher harmonics
            shape += a * np.sin(2 * np.pi * k * t / period) + b * np.cos(2 * np.pi * k * t / period)
        return shape / (np.abs(shape).max() + 1e-8)

    def _signed(self, value_range: tuple[float, float]) -> float:
        """Magnitude drawn from value_range with a random sign."""
        return self.rng.choice((-1.0, 1.0)) * self.rng.uniform(*value_range)

    def _scale_factor(self) -> float:
        """Scale factor from shift_scale_range, inverted with probability 0.5."""
        factor = self.rng.uniform(*self.shift_scale_range)
        return factor if self.rng.random() < 0.5 else 1.0 / factor
