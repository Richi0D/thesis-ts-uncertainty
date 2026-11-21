"""Inject distribution shifts, observation noise and spikes into any series.

An Injector maps a clean Sample (from SyntheticGenerator, ChaoticGenerator, or a real
series via Sample.from_series) to a perturbed one, in this order:

    1. shift   one component changes abruptly while a shift episode [tau, end) is active
    2. noise   y_t += noise_t, Gaussian with a std drawn per sample
    3. spikes  y_t += spikes_t, sparse outliers with random sign

and appends known-in-advance covariates over T + H: shift covariates (a step indicator of
the shift episodes, a ramp of the time since the episode start, and / or the exact signed
shift profile as an oracle) and a signed pulse indicator of the spikes (+1 up, -1 down).
The step and ramp say *when* a shift happens, never its type or magnitude.

Shift types (all work on any series):

    level      y_t += delta                         for t in [tau, end)
    trend      y_t += delta_slope * (t - tau)       for t in [tau, end)
    noise      noise std scaled by a factor         for t in [tau, end)
    amplitude  deviation from the linear trend      for t in [tau, end)
               (fitted outside the episodes) scaled by a factor

Noise and spikes can be restricted to the context or the horizon (noise_region,
spike_region), and spikes can recur with a fixed period (spike_period). shift_region places
the shift only in the horizon, only in the context (the series returns to normal at the
forecast start), or in the context and persisting through the horizon.

By default there is one episode from a random change point tau to the end of the window.
A recurring shift has two episodes with the same type and magnitude: one inside the
context, followed by a return to normal, and one starting in the horizon. It tests whether
a model learns the shift from the context and anticipates it.

All additive magnitudes (noise std, spike height, level jump, slope change) are absolute by
default and assume a series scale of about 1. With relative=True they are multiplied by the
std of the clean context, so the same settings work on unnormalised real series of any scale
(models that standardise with the context then see the same perturbation strength).

Missing values (NaN, e.g. in real series) are ignored: the context std and the amplitude
baseline are computed from the observed values only, spikes are only placed on observed values,
and a missing value stays missing (no value is filled in).

The episode helpers (draw_episodes, signed, scale_factor) are shared with the dynamics
shifts of ChaoticGenerator.
"""

import dataclasses
from collections.abc import Sequence

import numpy as np

from ts_uncertainty.sample import Sample

SHIFT_TYPES = ("level", "trend", "noise", "amplitude")
REGIONS = ("both", "context", "horizon")
SHIFT_REGIONS = ("any", *REGIONS)
SHIFT_COVARIATES = ("step", "ramp", "effect")  # covariate names: shift, shift_ramp, shift_effect


class Injector:
    """Callable that perturbs Samples; `map(injector, generator)` gives a perturbed stream.

    Args:
        noise: add Gaussian observation noise.
        spikes: add sparse random spikes (outliers).
        shift: apply a distribution shift at a random change point. A sample that already
            has a shift (e.g. a dynamics shift from ChaoticGenerator) keeps it and gets none.
        shift_covariate: append shift covariates over T + H (all zeros if the sample has no
            shift). True means "step"; a string or a sequence of SHIFT_COVARIATES picks variants,
            one column each, in the given order:
            "step"   ("shift")        1 while a shift episode is active, 0 elsewhere.
            "ramp"   ("shift_ramp")   (t - start) / (T + H) inside each episode, 0 elsewhere:
                                      grows steadily with time since the shift started, but
                                      gives neither the direction nor the magnitude.
            "effect" ("shift_effect") the exact signed shift profile (oracle): delta while
                                      active for level, delta_slope * (t - start) for trend
                                      (the added trend itself), log(factor) while active for
                                      noise / amplitude. Shifts made by the source (e.g.
                                      ChaoticGenerator dynamics) fall back to the step.
            "step" and "ramp" also mark shifts made by the source.
        spike_covariate: append a signed pulse indicator that is +1 at upward and -1 at
            downward spikes, 0 elsewhere (requires spikes).
        noise_std_range: range of the Gaussian observation noise std.
        noise_region: where noise is added: "both" (T + H), "context" or "horizon".
        spike_prob: per-time-step probability of a spike (ignored if spike_period is set).
        spike_magnitude_range: range of the absolute spike height (sign is random).
        spike_region: where spikes may occur: "both" (T + H), "context" or "horizon".
        spike_period: if set, spikes occur every spike_period steps from a random phase
            instead of at random steps, all with the same height and sign per sample, so
            the pattern is learnable from the context.
        shift_types: shift types to draw from, subset of SHIFT_TYPES.
        shift_position_range: range of the change point as a fraction of the shift region
            (T + H for "any"), e.g. (0.5, 1.0) covers the end of the context and the horizon.
        shift_region: where a non-recurring shift is placed: "any" (change point anywhere in
            shift_position_range of T + H, persistent), "horizon" (change point in the
            horizon), "context" (episode inside the context, ending at the forecast start) or
            "both" (change point in the context, persisting through the horizon).
            Anything but "any" requires shift_recurring_prob=0.
        shift_level_range: range of the absolute level jump (sign is random).
        shift_slope_range: range of the absolute slope change (sign is random).
        shift_scale_range: range of the scale factor for noise / amplitude shifts;
            with probability 0.5 the reciprocal is used, so shifts go both up and down.
        shift_recurring_prob: probability that a drawn shift is recurring: one episode in the
            context, back to normal, then the same shift again from a random start in the horizon.
        shift_duration_range: range of the length of each recurring episode as a fraction of T;
            the normal stretch after the context episode is at least as long as the episode.
        relative: multiply all additive magnitudes (noise std, spike height, level jump,
            slope change) by the std of the clean context, stored as info["injection_scale"].
            Use it for unnormalised series; scale factors (shift_scale_range) are already relative.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        noise: bool = True,
        spikes: bool = False,
        shift: bool = False,
        shift_covariate: bool | str | Sequence[str] = False,
        spike_covariate: bool = False,
        noise_std_range: tuple[float, float] = (0.05, 0.3),
        noise_region: str = "both",
        spike_prob: float = 0.01,
        spike_magnitude_range: tuple[float, float] = (2.0, 5.0),
        spike_region: str = "both",
        spike_period: int | None = None,
        shift_types: Sequence[str] = SHIFT_TYPES,
        shift_position_range: tuple[float, float] = (0.5, 1.0),
        shift_region: str = "any",
        shift_level_range: tuple[float, float] = (1.0, 3.0),
        shift_slope_range: tuple[float, float] = (0.01, 0.05),
        shift_scale_range: tuple[float, float] = (2.0, 4.0),
        shift_recurring_prob: float = 0.0,
        shift_duration_range: tuple[float, float] = (0.1, 0.25),
        relative: bool = False,
        seed: int | None = None,
    ):
        if shift_covariate is True:
            shift_covariate = ("step",)
        elif not shift_covariate:
            shift_covariate = ()
        elif isinstance(shift_covariate, str):
            shift_covariate = (shift_covariate,)
        shift_covariate = tuple(shift_covariate)
        if unknown := set(shift_covariate) - set(SHIFT_COVARIATES):
            raise ValueError(f"unknown shift covariates {unknown}, choose from {SHIFT_COVARIATES}")
        if len(set(shift_covariate)) != len(shift_covariate):
            raise ValueError(f"duplicate shift covariates in {shift_covariate}")
        if spike_covariate and not spikes:
            raise ValueError("spike_covariate=True requires spikes=True")
        if unknown := set(shift_types) - set(SHIFT_TYPES):
            raise ValueError(f"unknown shift types {unknown}, choose from {SHIFT_TYPES}")
        check_episode_ranges(shift_recurring_prob, shift_duration_range)
        if shift and not shift_types:
            raise ValueError("shift=True needs at least one shift type")
        for name, region in (("noise_region", noise_region), ("spike_region", spike_region)):
            if region not in REGIONS:
                raise ValueError(f"{name} {region!r} not in {REGIONS}")
        if shift_region not in SHIFT_REGIONS:
            raise ValueError(f"shift_region {shift_region!r} not in {SHIFT_REGIONS}")
        if shift_region != "any" and shift_recurring_prob > 0:
            raise ValueError("shift_region other than 'any' requires shift_recurring_prob=0")
        if not 0 <= shift_position_range[0] <= shift_position_range[1] <= 1:
            raise ValueError("shift_position_range must satisfy 0 <= low <= high <= 1")
        if spike_period is not None and spike_period < 1:
            raise ValueError("spike_period must be a positive integer")

        self.noise = noise
        self.spikes = spikes
        self.shift = shift
        self.shift_covariate = shift_covariate  # tuple of SHIFT_COVARIATES, empty for none
        self.spike_covariate = spike_covariate
        self.noise_std_range = noise_std_range
        self.noise_region = noise_region
        self.spike_prob = spike_prob
        self.spike_magnitude_range = spike_magnitude_range
        self.spike_region = spike_region
        self.spike_period = spike_period
        self.shift_types = tuple(shift_types)
        self.shift_position_range = shift_position_range
        self.shift_region = shift_region
        self.shift_level_range = shift_level_range
        self.shift_slope_range = shift_slope_range
        self.shift_scale_range = shift_scale_range
        self.shift_recurring_prob = shift_recurring_prob
        self.shift_duration_range = shift_duration_range
        self.relative = relative
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        sample: Sample,
        shift_point: float | None = None,
        shift_type: str | None = None,
        recurring: bool | None = None,
        shift_episodes: Sequence[tuple[float, float]] | None = None,
        shift_sign: int | None = None,
        relative: bool | None = None,
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
            shift_sign: +1 for an upward shift (level/slope up, noise/amplitude factor >= 1),
                -1 for a downward one (factor <= 1); random if None.
            relative: overrides the injector's relative setting for this sample, e.g.
                True for an unnormalised real sample with an injector shared across sources.
        """
        rng = self.rng
        T, H = len(sample.context), len(sample.target)
        length = T + H
        y = sample.series.astype(np.float64)
        relative = self.relative if relative is None else relative
        scale = _context_scale(y[:T]) if relative else 1.0  # unit of all additive magnitudes

        if shift_sign not in (None, 1, -1):
            raise ValueError(f"shift_sign {shift_sign} must be +1, -1 or None")
        forced = (
            shift_point is not None or shift_type is not None or recurring
            or shift_episodes is not None or shift_sign is not None
        )
        if forced and sample.shift_type is not None:
            raise ValueError(f"sample already has a {sample.shift_type} shift, only one shift per sample")

        episodes = list(sample.shift_episodes)
        shift_info = dict(sample.shift_info)
        own_shift = None  # shift type applied here, None if the sample keeps its own (or none)
        if forced or (self.shift and sample.shift_type is None):
            episodes = draw_episodes(
                rng, T, H, self.shift_position_range, self.shift_recurring_prob, self.shift_duration_range,
                shift_point=shift_point, recurring=recurring, shift_episodes=shift_episodes,
                region=self.shift_region,
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

        effect = active.astype(np.float64)  # signed shift profile, step for source shifts
        if own_shift == "level":
            delta = scale * signed(rng, self.shift_level_range, shift_sign)
            effect = delta * active
            y += effect
            shift_info = {"delta": delta}
        elif own_shift == "trend":
            delta = scale * signed(rng, self.shift_slope_range, shift_sign)
            effect = delta * ramp
            y += effect
            shift_info = {"delta_slope": delta}
        elif own_shift == "amplitude":
            factor = scale_factor(rng, self.shift_scale_range, shift_sign)
            observed = ~np.isnan(y)
            outside = ~active & observed
            baseline = _linear_fit(y, outside if outside.sum() >= 2 else observed)
            y = np.where(active, baseline + factor * (y - baseline), y)
            effect = np.log(factor) * active
            shift_info = {"factor": factor}

        if self.noise or own_shift == "noise":
            noise_std = np.full(length, scale * rng.uniform(*self.noise_std_range))
            noisy = _region_mask(self.noise_region, T, H) if self.noise else np.zeros(length, dtype=bool)
            if own_shift == "noise":
                factor = scale_factor(rng, self.shift_scale_range, shift_sign)
                if self.noise:
                    noise_std *= np.where(active, factor, 1.0)
                noisy |= active
                effect = np.log(factor) * active
                shift_info = {"factor": factor}
            y += rng.normal(0.0, noise_std * noisy)

        spike_idx = sample.spike_idx
        spike_heights = np.zeros(0)  # (n_spikes,), signed heights added at spike_idx
        if self.spikes:
            allowed = _region_mask(self.spike_region, T, H) & ~np.isnan(y)  # no spikes on missing values
            if self.spike_period is None:
                spike_idx = np.flatnonzero((rng.random(length) < self.spike_prob) & allowed)
                spike_heights = np.array([scale * signed(rng, self.spike_magnitude_range) for _ in spike_idx])
            else:
                # phase over the whole window, so the pattern continues across the forecast start
                phase = int(rng.integers(0, self.spike_period))
                periodic = np.zeros(length, dtype=bool)
                periodic[phase :: self.spike_period] = True
                spike_idx = np.flatnonzero(periodic & allowed)
                spike_heights = np.full(len(spike_idx), scale * signed(rng, self.spike_magnitude_range))
            y[spike_idx] += spike_heights

        covariates = [] if sample.covariate is None else [sample.covariate]  # each (T + H, c)
        covariate_names = list(sample.covariate_names)
        shift_columns = {  # each (T + H,)
            "step": ("shift", active),
            "ramp": ("shift_ramp", ramp / length),
            "effect": ("shift_effect", effect),
        }
        for variant in self.shift_covariate:
            name, column = shift_columns[variant]
            covariates.append(column[:, None])
            covariate_names.append(name)
        if self.spike_covariate:
            spike_indicator = np.zeros((length, 1))
            spike_indicator[spike_idx, 0] = np.sign(spike_heights)
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
            info={**sample.info, "injection_scale": scale} if relative else sample.info,
        )


def _context_scale(context: np.ndarray) -> float:
    """Std of the observed context values, falling back to |mean| (or 1) for a constant context."""
    observed = context[~np.isnan(context)]
    if len(observed) == 0:
        return 1.0
    std, mean = float(observed.std()), float(observed.mean())
    if std > 1e-6 * (1.0 + abs(mean)):
        return std
    return abs(mean) or 1.0


def _region_mask(region: str, context_length: int, horizon: int) -> np.ndarray:
    """Boolean mask over T + H that is True in the given region ("both", "context", "horizon")."""
    mask = np.zeros(context_length + horizon, dtype=bool)
    if region in ("both", "context"):
        mask[:context_length] = True
    if region in ("both", "horizon"):
        mask[context_length:] = True
    return mask


def _linear_fit(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares line a + b * t through y[mask], evaluated at every t.

    With fewer than 2 points the line is flat at their mean (0 without any point).
    """
    t = np.arange(len(y), dtype=np.float64)
    if mask.sum() < 2:
        return np.full(len(y), float(y[mask].mean()) if mask.any() else 0.0)
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
    region: str = "any",
) -> list[tuple[int, int]]:
    """Shift episodes as sorted (start, end) indices into T + H, end exclusive.

    Explicit shift_episodes or shift_point (fractions of T + H) are used as given;
    otherwise one episode from a change point in position_range, or with
    recurring_prob (or recurring=True) a recurring pair.

    region places the drawn episode, with position_range as a fraction of that region:
    "any" (T + H, persistent), "horizon" (change point in the horizon, persistent),
    "both" (change point in the context, persistent) or "context" (change point in the
    context, episode ends at the forecast start).
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
        offset, span = {"any": (0, length), "horizon": (context_length, horizon)}.get(region, (0, context_length))
        lo, hi = (offset + round(f * span) for f in position_range)
        lo = max(lo, 1) if region in ("both", "context") else lo  # keep some normal context
        lo, hi = min(lo, offset + span - 1), min(hi, offset + span)  # at least one shifted step
        start = int(rng.integers(lo, max(hi, lo + 1)))
        return [(start, context_length if region == "context" else length)]
    if not 0 <= shift_point <= 1:
        raise ValueError(f"shift_point {shift_point} out of bounds. Valid range is [0, 1]")
    return [(round(shift_point * length), length)]


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


def signed(rng: np.random.Generator, value_range: tuple[float, float], sign: int | None = None) -> float:
    """Magnitude drawn from value_range with the given sign (+1 / -1), random if None."""
    if sign is None:
        sign = rng.choice((-1.0, 1.0))
    return float(sign * rng.uniform(*value_range))


def scale_factor(rng: np.random.Generator, value_range: tuple[float, float], sign: int | None = None) -> float:
    """Scale factor from value_range (>= 1), inverted for sign -1; with probability 0.5 if None."""
    factor = rng.uniform(*value_range)
    up = rng.random() < 0.5 if sign is None else sign > 0
    return float(factor if up else 1.0 / factor)
