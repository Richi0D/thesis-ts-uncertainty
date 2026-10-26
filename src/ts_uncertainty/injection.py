"""Inject distribution shifts, observation noise and spikes into any series.

An Injector maps a clean Sample (from SyntheticGenerator, ChaoticGenerator, or a real
series via Sample.from_series) to a perturbed one, in this order:

    1. shift   one component changes abruptly while a shift episode [tau, end) is active
    2. noise   y_t += noise_t, Gaussian with a std drawn per sample
    3. spikes  y_t += spikes_t, sparse outliers with random sign

and appends known-in-advance indicator covariates over T + H: a step indicator of the
shift episodes and a pulse indicator of the spikes. The indicators say *when* something
happens, never its type or magnitude.

Shift types (all work on any series):

    level      y_t += delta                         for t in [tau, end)
    trend      y_t += delta_slope * (t - tau)       for t in [tau, end)
    noise      noise std scaled by a factor         for t in [tau, end)
    amplitude  deviation from the linear trend      for t in [tau, end)
               (fitted outside the episodes) scaled by a factor

By default there is one episode from a random change point tau to the end of the window.
A recurring shift has two episodes with the same type and magnitude: one inside the
context, followed by a return to normal, and one starting in the horizon. It tests whether
a model learns the shift from the context and anticipates it.

The episode helpers (draw_episodes, signed, scale_factor) are shared with the dynamics
shifts of ChaoticGenerator.
"""

import dataclasses
from collections.abc import Sequence

import numpy as np

from ts_uncertainty.sample import Sample

SHIFT_TYPES = ("level", "trend", "noise", "amplitude")


class Injector:
    """Callable that perturbs Samples; `map(injector, generator)` gives a perturbed stream.

    Args:
        noise: add Gaussian observation noise.
        spikes: add sparse random spikes (outliers).
        shift: apply a distribution shift at a random change point. A sample that already
            has a shift (e.g. a dynamics shift from ChaoticGenerator) keeps it and gets none.
        shift_covariate: append an indicator that is 1 while a shift episode is active
            (all zeros if the sample has no shift). Also marks shifts made by the source.
        spike_covariate: append a pulse indicator that is 1 at spike positions (requires spikes).
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
        shift_recurring_prob: probability that a drawn shift is recurring: one episode in the
            context, back to normal, then the same shift again from a random start in the horizon.
        shift_duration_range: range of the length of each recurring episode as a fraction of T;
            the normal stretch after the context episode is at least as long as the episode.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        noise: bool = True,
        spikes: bool = False,
        shift: bool = False,
        shift_covariate: bool = False,
        spike_covariate: bool = False,
        noise_std_range: tuple[float, float] = (0.05, 0.3),
        spike_prob: float = 0.01,
        spike_magnitude_range: tuple[float, float] = (2.0, 5.0),
        shift_types: Sequence[str] = SHIFT_TYPES,
        shift_position_range: tuple[float, float] = (0.5, 1.0),
        shift_level_range: tuple[float, float] = (1.0, 3.0),
        shift_slope_range: tuple[float, float] = (0.01, 0.05),
        shift_scale_range: tuple[float, float] = (2.0, 4.0),
        shift_recurring_prob: float = 0.0,
        shift_duration_range: tuple[float, float] = (0.1, 0.25),
        seed: int | None = None,
    ):
        if spike_covariate and not spikes:
            raise ValueError("spike_covariate=True requires spikes=True")
        if unknown := set(shift_types) - set(SHIFT_TYPES):
            raise ValueError(f"unknown shift types {unknown}, choose from {SHIFT_TYPES}")
        check_episode_ranges(shift_recurring_prob, shift_duration_range)
        if shift and not shift_types:
            raise ValueError("shift=True needs at least one shift type")

        self.noise = noise
        self.spikes = spikes
        self.shift = shift
        self.shift_covariate = shift_covariate
        self.spike_covariate = spike_covariate
        self.noise_std_range = noise_std_range
        self.spike_prob = spike_prob
        self.spike_magnitude_range = spike_magnitude_range
        self.shift_types = tuple(shift_types)
        self.shift_position_range = shift_position_range
        self.shift_level_range = shift_level_range
        self.shift_slope_range = shift_slope_range
        self.shift_scale_range = shift_scale_range
        self.shift_recurring_prob = shift_recurring_prob
        self.shift_duration_range = shift_duration_range
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        sample: Sample,
        shift_point: float | None = None,
        shift_type: str | None = None,
        recurring: bool | None = None,
        shift_episodes: Sequence[tuple[float, float]] | None = None,
    ) -> Sample:
        """Return a perturbed copy of sample.

        Passing any shift argument forces a shift, even if the injector has shift=False.
        An explicit shift_type may be any of SHIFT_TYPES; "noise" with noise=False adds
        noise only during the shift episodes.

        Args:
            shift_point: change point as a fraction of T + H in [0, 1]; drawn from
                shift_position_range if None.
            shift_type: one of SHIFT_TYPES; drawn from shift_types if None.
            recurring: True forces a recurring shift (one episode in the context, one in the
                horizon), False a single persistent one; drawn with shift_recurring_prob if None.
                Can't be combined with shift_point.
            shift_episodes: explicit (start, end) episodes as fractions of T + H in [0, 1],
                sorted and non-overlapping, all with the same shift type and magnitude.
                Can't be combined with shift_point or recurring.
        """
        rng = self.rng
        T, H = len(sample.context), len(sample.target)
        length = T + H
        y = sample.series.astype(np.float64)

        forced = shift_point is not None or shift_type is not None or recurring or shift_episodes is not None
        if forced and sample.shift_type is not None:
            raise ValueError(f"sample already has a {sample.shift_type} shift, only one shift per sample")

        episodes = list(sample.shift_episodes)
        shift_info = dict(sample.shift_info)
        own_shift = None  # shift type applied here, None if the sample keeps its own (or none)
        if forced or (self.shift and sample.shift_type is None):
            episodes = draw_episodes(
                rng, T, H, self.shift_position_range, self.shift_recurring_prob, self.shift_duration_range,
                shift_point=shift_point, recurring=recurring, shift_episodes=shift_episodes,
            )
            if shift_type is None:
                if not self.shift_types:
                    raise ValueError("no shift types configured, pass shift_type")
                shift_type = str(rng.choice(self.shift_types))
            elif shift_type not in SHIFT_TYPES:
                raise ValueError(f"shift_type {shift_type} not in {SHIFT_TYPES}")
            own_shift = shift_type

        active = np.zeros(length, dtype=bool)  # mask of time steps inside a shift episode
        ramp = np.zeros(length)  # steps since the start of the current episode, 0 outside
        for start, end in episodes:
            active[start:end] = True
            ramp[start:end] = np.arange(end - start)

        if own_shift == "level":
            delta = signed(rng, self.shift_level_range)
            y += delta * active
            shift_info = {"delta": delta}
        elif own_shift == "trend":
            delta = signed(rng, self.shift_slope_range)
            y += delta * ramp
            shift_info = {"delta_slope": delta}
        elif own_shift == "amplitude":
            factor = scale_factor(rng, self.shift_scale_range)
            baseline = _linear_fit(y, ~active if (~active).sum() >= 2 else np.ones(length, dtype=bool))
            y = np.where(active, baseline + factor * (y - baseline), y)
            shift_info = {"factor": factor}

        if self.noise or own_shift == "noise":
            noise_std = np.full(length, rng.uniform(*self.noise_std_range))
            if own_shift == "noise":
                factor = scale_factor(rng, self.shift_scale_range)
                noise_std *= np.where(active, factor, 1.0) if self.noise else active
                shift_info = {"factor": factor}
            y += rng.normal(0.0, noise_std)

        spike_idx = sample.spike_idx
        if self.spikes:
            spike_idx = np.flatnonzero(rng.random(length) < self.spike_prob)
            y[spike_idx] += [signed(rng, self.spike_magnitude_range) for _ in spike_idx]

        covariates = [] if sample.covariate is None else [sample.covariate]  # each (T + H, c)
        covariate_names = list(sample.covariate_names)
        if self.shift_covariate:
            covariates.append(active[:, None])
            covariate_names.append("shift")
        if self.spike_covariate:
            spike_indicator = np.zeros((length, 1))
            spike_indicator[spike_idx] = 1.0
            covariates.append(spike_indicator)
            covariate_names.append("spike")
        covariate = np.concatenate(covariates, axis=1).astype(np.float32) if covariates else None

        y = y.astype(np.float32)
        return dataclasses.replace(
            sample,
            context=y[:T],
            target=y[T:],
            covariate=covariate,
            covariate_names=tuple(covariate_names),
            shift_type=shift_type if own_shift else sample.shift_type,
            shift_episodes=tuple(episodes),
            shift_info=shift_info,
            spike_idx=spike_idx,
        )


def _linear_fit(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares line a + b * t through y[mask], evaluated at every t."""
    t = np.arange(len(y), dtype=np.float64)
    b, a = np.polyfit(t[mask], y[mask], 1)
    return a + b * t


# ---------------------------------------------------------------------- shared shift helpers


def check_episode_ranges(recurring_prob: float, duration_range: tuple[float, float]):
    if not 0 <= recurring_prob <= 1:
        raise ValueError("shift_recurring_prob must be in [0, 1]")
    if not 0 < duration_range[0] <= duration_range[1] <= 0.5:
        raise ValueError("shift_duration_range must satisfy 0 < low <= high <= 0.5")


def draw_episodes(
    rng: np.random.Generator,
    context_length: int,
    horizon: int,
    position_range: tuple[float, float],
    recurring_prob: float,
    duration_range: tuple[float, float],
    shift_point: float | None = None,
    recurring: bool | None = None,
    shift_episodes: Sequence[tuple[float, float]] | None = None,
) -> list[tuple[int, int]]:
    """Shift episodes as sorted (start, end) indices into T + H, end exclusive.

    Explicit shift_episodes or shift_point (fractions of T + H) are used as given;
    otherwise one persistent episode from a change point in position_range, or with
    recurring_prob (or recurring=True) a recurring pair.
    """
    length = context_length + horizon
    if shift_episodes is not None:
        if shift_point is not None or recurring is not None:
            raise ValueError("shift_episodes can't be combined with shift_point or recurring")
        return _parse_episodes(shift_episodes, length)
    if recurring and shift_point is not None:
        raise ValueError("recurring=True can't be combined with shift_point")
    if recurring is None:
        # only draw when enabled, so seeds stay reproducible with the default of 0
        recurring = shift_point is None and recurring_prob > 0 and rng.random() < recurring_prob
    if recurring:
        return _recurring_episodes(rng, context_length, horizon, duration_range)
    if shift_point is None:
        lo, hi = (round(f * length) for f in position_range)
        start = int(rng.integers(lo, max(hi, lo + 1)))
    else:
        if not 0 <= shift_point <= 1:
            raise ValueError(f"shift_point {shift_point} out of bounds. Valid range is [0, 1]")
        start = round(shift_point * length)
    return [(start, length)]


def _recurring_episodes(rng, context_length, horizon, duration_range) -> list[tuple[int, int]]:
    """One episode inside the context, then the same duration again from a start in the horizon."""
    T, H = context_length, horizon
    duration = max(1, int(rng.uniform(*duration_range) * T))
    # leave at least `duration` normal steps between the context episode and the forecast start
    start = int(rng.integers(0, T - 2 * duration + 1))
    second = int(rng.integers(T, T + H))
    return [(start, start + duration), (second, min(second + duration, T + H))]


def _parse_episodes(episodes: Sequence[tuple[float, float]], length: int) -> list[tuple[int, int]]:
    """Convert (start, end) fractions of T + H to sorted, non-overlapping index episodes."""
    if not episodes:
        raise ValueError("shift_episodes must contain at least one episode")
    result = []
    for lo, hi in episodes:
        if not 0 <= lo < hi <= 1:
            raise ValueError(f"shift episode {(lo, hi)} invalid, need 0 <= start < end <= 1")
        start, end = round(lo * length), round(hi * length)
        if start == end:
            raise ValueError(f"shift episode {(lo, hi)} is shorter than one time step")
        if result and start < result[-1][1]:
            raise ValueError("shift_episodes must be sorted and non-overlapping")
        result.append((start, end))
    return result


def signed(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    """Magnitude drawn from value_range with a random sign."""
    return float(rng.choice((-1.0, 1.0)) * rng.uniform(*value_range))


def scale_factor(rng: np.random.Generator, value_range: tuple[float, float]) -> float:
    """Scale factor from value_range, inverted with probability 0.5."""
    factor = rng.uniform(*value_range)
    return float(factor if rng.random() < 0.5 else 1.0 / factor)
