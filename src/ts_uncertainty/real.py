"""Random windows from real univariate time series.

Takes a long polars DataFrame (one row per time step, several series stacked, e.g. the
splits written by notebooks/data_download.ipynb) and draws windows of length T + H:

    1. a series (uniformly, or proportional to its number of valid windows),
    2. a start position, such that the window has no missing values and no gap in the
       timestamps (every step equals the most common step of that series),
    3. optionally z-normalises the window with the statistics of the context, so the series
       has scale ≈ 1 like the synthetic and chaotic samples (and the Injector defaults fit).

Windows with a (near) constant context are rejected and redrawn. Only the value column is
used (univariate, no covariates).

The season length (for seasonal baselines like AutoARIMA) is not stored in the data: it
follows from the sampling step of each series and a seasonal period given as a duration,
by default one day (the dominant cycle of energy data), so 5 min -> 288, 15 min -> 96,
60 min -> 24 steps. A per-series column can override it.
"""

from collections.abc import Iterator, Sequence
from datetime import timedelta

import numpy as np
import polars as pl

from ts_uncertainty.sample import Sample

NORMALIZE = ("context", "none")
WEIGHTS = ("series", "windows")


class RealSampler:
    """Infinite generator of windows from real univariate time series.

    Args:
        df: long DataFrame with one row per (series, time step).
        context_length: T, number of observed time steps.
        horizon: H, number of time steps to forecast.
        series: restrict to these series ids (values of id_col); None = all.
        weight: "series" draws every series with equal probability, "windows" every valid
            window (long series are drawn more often).
        normalize: "context" z-normalises each window with the context mean / std (stored in
            info as loc / scale), "none" keeps the raw values.
        check_gaps: reject windows with a gap in time_col (step != most common step of the series).
        season: seasonal period as a duration; season_length = season / sampling step of the
            series (rounded, at least 1). None = no season (season_length 1).
        season_col: optional column with a season length in steps per series (first value of the
            series is used); overrides season where it is not null.
        time_col, value_col, id_col, name_col: column names; name_col may be None.
        max_tries: redraws per sample (constant context) before giving up.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        df: pl.DataFrame,
        context_length: int,
        horizon: int,
        series: Sequence[int | str] | None = None,
        weight: str = "series",
        normalize: str = "context",
        check_gaps: bool = True,
        season: timedelta | None = timedelta(days=1),
        season_col: str | None = None,
        time_col: str = "timestamp",
        value_col: str = "value",
        id_col: str = "series_index",
        name_col: str | None = "series_name",
        max_tries: int = 20,
        seed: int | None = None,
    ):
        if context_length <= 0 or horizon <= 0:
            raise ValueError("context_length and horizon must be positive")
        if weight not in WEIGHTS:
            raise ValueError(f"unknown weight {weight!r}, choose from {WEIGHTS}")
        if normalize not in NORMALIZE:
            raise ValueError(f"unknown normalize {normalize!r}, choose from {NORMALIZE}")
        if season is not None and season <= timedelta(0):
            raise ValueError("season must be a positive duration or None")
        cols = [id_col, time_col, value_col] + [c for c in (name_col, season_col) if c]
        if missing := set(cols) - set(df.columns):
            raise ValueError(f"df is missing columns {missing}")

        self.context_length = context_length
        self.horizon = horizon
        self.weight = weight
        self.normalize = normalize
        self.check_gaps = check_gaps
        self.max_tries = max_tries
        self.rng = np.random.default_rng(seed)

        df = df.select(cols)
        if series is not None:
            df = df.filter(pl.col(id_col).is_in(list(series)))
        df = df.sort(id_col, time_col)

        length = context_length + horizon
        self._ids: list = []
        self._names: list[str | None] = []
        self._values: list[np.ndarray] = []  # (N_i,) float64 per series
        self._times: list[np.ndarray] = []  # (N_i,) datetime64 per series
        self._starts: list[np.ndarray] = []  # valid window starts per series
        self._freqs: list[np.timedelta64] = []  # most common sampling step per series
        self._season_lengths: list[int] = []
        for part in df.partition_by(id_col, maintain_order=True):
            values = part[value_col].cast(pl.Float64).fill_null(np.nan).to_numpy()
            times = part[time_col].to_numpy()
            starts = self._valid_starts(values, times, length)
            if len(starts) == 0:
                continue
            freq = self._freq(times)
            season_length = 1
            if season is not None:
                season_length = max(1, round(np.timedelta64(season) / freq))
            if season_col and part[season_col][0] is not None:
                season_length = int(part[season_col][0])
                if season_length < 1:
                    raise ValueError(f"{season_col} must be >= 1, got {season_length} for series {part[id_col][0]!r}")
            self._ids.append(part[id_col][0])
            self._names.append(part[name_col][0] if name_col else None)
            self._values.append(values)
            self._times.append(times)
            self._starts.append(starts)
            self._freqs.append(freq)
            self._season_lengths.append(season_length)

        if not self._ids:
            raise ValueError(f"no series has a valid window of length T + H = {length}")
        n_windows = np.array([len(s) for s in self._starts], dtype=np.float64)
        self._probs = n_windows / n_windows.sum() if weight == "windows" else None

    def __iter__(self) -> Iterator[Sample]:
        return self

    def __next__(self) -> Sample:
        return self.sample()

    @property
    def series_ids(self) -> tuple:
        """Ids of the series that have at least one valid window."""
        return tuple(self._ids)

    @property
    def season_lengths(self) -> dict:
        """Season length in steps per series id."""
        return dict(zip(self._ids, self._season_lengths))

    def sample(self, series: int | str | None = None, start: int | None = None) -> Sample:
        """Draw one window; info holds series_index, series_name, start, start_time, freq, loc, scale.

        Args:
            series: series id to draw from; drawn if None.
            start: row index of the window start within the series (must be valid); drawn if None.
        """
        if series is not None and series not in self._ids:
            raise ValueError(f"series {series!r} unknown or has no valid window")
        if start is not None and series is None:
            raise ValueError("start requires series")

        T = self.context_length
        for _ in range(self.max_tries):
            i = self._ids.index(series) if series is not None else int(self.rng.choice(len(self._ids), p=self._probs))
            starts = self._starts[i]
            if start is None:
                s = int(starts[self.rng.integers(len(starts))])
            elif start in starts:
                s = int(start)
            else:
                raise ValueError(f"start {start} is not a valid window start for series {series!r}")

            y = self._values[i][s : s + T + self.horizon]  # (T + H,)
            loc, scale = float(y[:T].mean()), float(y[:T].std())
            if scale <= 1e-6 * (1.0 + abs(loc)):  # constant context, can't normalise / uninformative
                if start is not None:
                    raise ValueError("context of the requested window is constant")
                continue
            if self.normalize == "none":
                loc, scale = 0.0, 1.0
            y = ((y - loc) / scale).astype(np.float32)

            times = self._times[i]
            return Sample(
                context=y[:T],
                target=y[T:],
                season_length=self._season_lengths[i],
                info={
                    "series_index": self._ids[i],
                    "series_name": self._names[i],
                    "start": s,
                    "start_time": times[s],
                    "freq": self._freqs[i],
                    "loc": loc,
                    "scale": scale,
                },
            )
        raise RuntimeError(f"no window with non-constant context after {self.max_tries} tries")

    def sample_batch(self, batch_size: int) -> list[Sample]:
        """List of batch_size samples; use stack_samples to get stacked arrays."""
        return [self.sample() for _ in range(batch_size)]

    def _valid_starts(self, values: np.ndarray, times: np.ndarray, length: int) -> np.ndarray:
        """Starts s such that [s, s + length) has no NaN and (if check_gaps) a regular time step."""
        n = len(values)
        if n < length:
            return np.empty(0, dtype=np.int64)
        bad = np.isnan(values).astype(np.int64)  # (N,)
        bad_cum = np.concatenate([[0], np.cumsum(bad)])
        ok = bad_cum[length:] - bad_cum[:-length] == 0  # (N - length + 1,)

        if self.check_gaps:
            steps = np.diff(times)  # (N - 1,), step i lies between rows i and i + 1
            gap = (steps != self._freq(times)).astype(np.int64)
            gap_cum = np.concatenate([[0], np.cumsum(gap)])
            ok &= gap_cum[length - 1 :] - gap_cum[: -(length - 1) or None] == 0

        return np.flatnonzero(ok)

    @staticmethod
    def _freq(times: np.ndarray) -> np.timedelta64:
        """Most common step between consecutive time stamps."""
        vals, counts = np.unique(np.diff(times), return_counts=True)
        return vals[counts.argmax()]
