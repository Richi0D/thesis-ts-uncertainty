"""Synthetic univariate time series generator.

Each sample is a clean series

    y_t = level + slope * t + seasonal_t

split into a context window of length T and a forecast horizon of length H. Optionally,
known-in-advance sin/cos features of the seasonal phase are returned for the full window
(T + H), so they are available over the forecast horizon as well.

Noise, spikes and distribution shifts are added afterwards by an Injector
(ts_uncertainty.injection), which works on any series.
"""

from collections.abc import Iterator, Sequence

import numpy as np

from ts_uncertainty.sample import Sample


class SyntheticGenerator:
    """Infinite generator of clean synthetic time series.

    Args:
        context_length: T, number of observed time steps.
        horizon: H, number of time steps to forecast.
        seasonal: add a seasonal component.
        covariate: return sin/cos phase features of the seasonal periods (requires seasonal).
        periods: candidate seasonal periods; each sample draws n_periods of them.
        n_periods: number of superimposed seasonal patterns per sample.
        n_harmonics: Fourier harmonics per period (1 = pure sine, more = sharper shapes).
        level_range: range of the constant level.
        slope_range: range of the linear trend slope per time step.
        amplitude_range: range of the seasonal amplitude (per period).
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        context_length: int,
        horizon: int,
        seasonal: bool = True,
        covariate: bool = False,
        periods: Sequence[int] = (7, 12, 24, 52),
        n_periods: int = 1,
        n_harmonics: int = 3,
        level_range: tuple[float, float] = (-1.0, 1.0),
        slope_range: tuple[float, float] = (-0.01, 0.01),
        amplitude_range: tuple[float, float] = (0.5, 2.0),
        seed: int | None = None,
    ):
        if covariate and not seasonal:
            raise ValueError("covariate=True requires seasonal=True")
        if seasonal and n_periods > len(periods):
            raise ValueError("n_periods must not exceed the number of candidate periods")

        self.context_length = context_length
        self.horizon = horizon
        self.seasonal = seasonal
        self.covariate = covariate
        self.periods = tuple(periods)
        self.n_periods = n_periods
        self.n_harmonics = n_harmonics
        self.level_range = level_range
        self.slope_range = slope_range
        self.amplitude_range = amplitude_range
        self.rng = np.random.default_rng(seed)

    def __iter__(self) -> Iterator[Sample]:
        return self

    def __next__(self) -> Sample:
        return self.sample()

    def sample(self) -> Sample:
        """Draw one clean sample; info holds the seasonal periods used (empty if none).

        season_length is the period with the largest amplitude (1 without seasonality).
        """
        rng = self.rng
        length = self.context_length + self.horizon
        t = np.arange(length, dtype=np.float64)

        y = rng.uniform(*self.level_range) + rng.uniform(*self.slope_range) * t

        periods: tuple[int, ...] = ()
        amplitudes: list[float] = []
        covariate = None
        covariate_names: tuple[str, ...] = ()
        if self.seasonal:
            periods = tuple(int(p) for p in rng.choice(self.periods, size=self.n_periods, replace=False))
            # random time offset so samples don't all start at phase 0
            t_abs = t + rng.integers(0, max(periods))
            for p in periods:
                amplitudes.append(rng.uniform(*self.amplitude_range))
                y += amplitudes[-1] * self._seasonal_shape(t_abs, p)
            if self.covariate:
                phase = 2 * np.pi * t_abs[:, None] / np.array(periods)[None, :]
                covariate = np.concatenate([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)  # (T + H, 2 * n_periods)
                covariate_names = tuple(f"sin (p={p})" for p in periods) + tuple(f"cos (p={p})" for p in periods)

        y = y.astype(np.float32)
        # a single season for AutoARIMA: the period with the largest amplitude
        season_length = periods[int(np.argmax(amplitudes))] if periods else 1
        return Sample(
            context=y[: self.context_length],
            target=y[self.context_length :],
            covariate=covariate,
            covariate_names=covariate_names,
            season_length=season_length,
            info={"periods": periods},
        )

    def sample_batch(self, batch_size: int) -> list[Sample]:
        """List of batch_size samples; use stack_samples to get stacked arrays."""
        return [self.sample() for _ in range(batch_size)]

    def _seasonal_shape(self, t: np.ndarray, period: int) -> np.ndarray:
        """Random periodic shape from a truncated Fourier series, normalised to unit max amplitude."""
        shape = np.zeros_like(t)
        for k in range(1, self.n_harmonics + 1):
            a, b = self.rng.normal(size=2) / k  # decay higher harmonics
            shape += a * np.sin(2 * np.pi * k * t / period) + b * np.cos(2 * np.pi * k * t / period)
        return shape / (np.abs(shape).max() + 1e-8)
