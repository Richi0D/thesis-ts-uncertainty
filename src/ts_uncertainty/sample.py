"""Common sample type for all data sources, batching, and conversion to model inputs.

Every source (SyntheticGenerator, ChaoticGenerator, or a real series via
Sample.from_series) yields a Sample; an Injector maps Samples to Samples. So the steps
compose as

    gen = ChaoticGenerator(T, H, seed=0)
    inj = Injector(shift=True, spikes=True, shift_covariate=True, seed=1)
    s = inj(next(gen))                                    # one sample
    batch = stack_samples([inj(next(gen)) for _ in range(32)])
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import torch


@dataclass
class Sample:
    context: np.ndarray  # (T,)
    target: np.ndarray  # (H,)
    covariate: np.ndarray | None = None  # (T + H, C) features named in covariate_names, or None
    covariate_names: tuple[str, ...] = ()  # one name per covariate column
    shift_type: str | None = None  # injection.SHIFT_TYPES or chaotic.DYNAMICS_SHIFT_TYPES, or None
    shift_episodes: tuple[tuple[int, int], ...] = ()  # (start, end) of each shift episode, end exclusive
    shift_info: dict = field(default_factory=dict)  # magnitude, e.g. {"delta": 1.3}, {"factor": 2.3} or {"param": "rho", ...}
    spike_idx: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))  # indices into T + H
    info: dict = field(default_factory=dict)  # source metadata, e.g. {"periods": (24,)} or {"system": "Lorenz", ...}

    @classmethod
    def from_series(
        cls,
        series: np.ndarray,
        context_length: int,
        covariate: np.ndarray | None = None,
        covariate_names: Sequence[str] = (),
        **info,
    ) -> "Sample":
        """Wrap any (T + H,) series, e.g. a real one, so it can go through an Injector."""
        series = np.asarray(series, dtype=np.float32)
        if series.ndim != 1 or not 0 < context_length < len(series):
            raise ValueError("series must be 1-d and longer than context_length")
        if covariate is not None and (covariate.ndim != 2 or len(covariate) != len(series)):
            raise ValueError("covariate must have shape (T + H, C)")
        if covariate is not None and len(covariate_names) != covariate.shape[1]:
            raise ValueError("need one covariate name per covariate column")
        return cls(
            context=series[:context_length],
            target=series[context_length:],
            covariate=covariate,
            covariate_names=tuple(covariate_names),
            info=info,
        )

    @property
    def series(self) -> np.ndarray:
        """(T + H,) context followed by target."""
        return np.concatenate([self.context, self.target])

    @property
    def shift_point(self) -> int | None:
        """Start of the first shift episode as index into T + H, or None."""
        return self.shift_episodes[0][0] if self.shift_episodes else None

    @property
    def shift_mask(self) -> np.ndarray:
        """(T + H,) bool, True while a shift episode is active."""
        mask = np.zeros(len(self.context) + len(self.target), dtype=bool)
        for start, end in self.shift_episodes:
            mask[start:end] = True
        return mask


def stack_samples(samples: Sequence[Sample]) -> dict[str, np.ndarray | None]:
    """Stack samples of equal T and H.

    Returns context (B, T), target (B, H), covariate (B, T + H, C) or None,
    shift_point (B,) with -1 where no shift was applied (start of the first episode),
    and shift_mask (B, T + H), True while a shift episode is active.
    """
    has_cov = {s.covariate is not None for s in samples}
    if len(has_cov) != 1:
        raise ValueError("don't mix samples with and without covariates")
    return {
        "context": np.stack([s.context for s in samples]),
        "target": np.stack([s.target for s in samples]),
        "covariate": np.stack([s.covariate for s in samples]) if has_cov == {True} else None,
        "shift_point": np.array([-1 if s.shift_point is None else s.shift_point for s in samples]),
        "shift_mask": np.stack([s.shift_mask for s in samples]),
    }


"""
Samples to model input
"""

def sample_to_sf_input(s: Sample, uid: str = "0"):
    """
    returns:
      df:     long df (unique_id, ds, y, cov_*) over the history, length T
      X_df:   future covariates (unique_id, ds, cov_*), length H, or None
      target: (1, H) ground truth
    """
    x = np.asarray(s.context, dtype=np.float32)
    y = np.asarray(s.target, dtype=np.float32)
    T, H = len(x), len(y)

    df_cols = {"unique_id": [uid] * T, "ds": np.arange(T), "y": x}
    X_df = None

    if s.covariate is not None and np.size(s.covariate) > 0:
        c = np.asarray(s.covariate, dtype=np.float32)
        if c.ndim == 1:
            c = c[:, None]
        assert c.shape[0] >= T + H, "covariates must cover context + horizon"
        cov_names = [f"cov_{i}" for i in range(c.shape[1])]
        df_cols |= {n: c[:T, i] for i, n in enumerate(cov_names)}
        X_df = pl.DataFrame(
            {"unique_id": [uid] * H, "ds": np.arange(T, T + H)}
            | {n: c[T:T + H, i] for i, n in enumerate(cov_names)}
        )

    return pl.DataFrame(df_cols), X_df, torch.from_numpy(y).unsqueeze(0)

def samples_to_sf_inputs(samples):
    dfs, X_dfs, ys = zip(*(sample_to_sf_input(s, uid=str(i)) for i, s in enumerate(samples)))
    df = pl.concat(dfs)
    X_df = None if X_dfs[0] is None else pl.concat(X_dfs)
    return df, X_df, torch.stack(ys)    # (B, 1, H)


def sample_to_chronos2_input(s: Sample):
    """
    s.context:   (T,)               target history
    s.covariate: (T + H, V) or None covariates over history + forecast horizon
    s.target:    (H,)               ground truth over the forecast horizon
    returns:
      inp:    dict for Chronos2Pipeline.predict
      target: (1, H)                ground truth, shaped like one target variate
    """
    x = np.asarray(s.context, dtype=np.float32)
    y = np.asarray(s.target, dtype=np.float32)
    T, H = len(x), len(y)

    inp = {"target": x}

    if s.covariate is not None and np.size(s.covariate) > 0:
        c = np.asarray(s.covariate, dtype=np.float32)
        if c.ndim == 1:
            c = c[:, None]                      # single covariate given as (T+H,)
        assert c.shape[0] >= T + H, "covariates must cover context + horizon"
        V = c.shape[1]
        inp["past_covariates"] = {f"cov_{i}": c[:T, i] for i in range(V)}
        inp["future_covariates"] = {f"cov_{i}": c[T:T + H, i] for i in range(V)}

    return inp, torch.from_numpy(y).unsqueeze(0)

def samples_to_chronos2_inputs(samples):
    inputs, targets = zip(*(sample_to_chronos2_input(s) for s in samples))
    return list(inputs), torch.stack(targets)    # list of B dicts, (B, 1, H)


def sample_to_tirex2_input(s: Sample):
    """
    s.context:   (T,)               target history
    s.covariate: (T + H, V) or None covariates over history + forecast horizon
    s.target:    (H,)               ground truth over the forecast horizon
    returns:
      inp:    TimeseriesType
      target: (1, H)                ground truth, shaped like one target variate
    """
    from tirex2 import TimeseriesType

    x = torch.as_tensor(s.context, dtype=torch.float32)
    y = torch.as_tensor(s.target, dtype=torch.float32)
    T, H = len(x), len(y)

    future = None
    if s.covariate is not None and np.size(s.covariate) > 0:
        c = torch.as_tensor(s.covariate, dtype=torch.float32)
        if c.ndim == 1:
            c = c[:, None]                       # single covariate given as (T+H,)
        assert c.shape[0] >= T + H, "covariates must cover context + horizon"
        # past = c[:T].T                           # (V, T) # no past only covariates in generator
        future = c[:T + H].T                    # (V, T+H)

    inp = TimeseriesType(target=x[None], past_covariates=None, future_covariates=future)
    return inp, y[None]

def samples_to_tirex2_inputs(samples):
    inputs, targets = zip(*(sample_to_tirex2_input(s) for s in samples))
    return list(inputs), torch.stack(targets)    # list of B dicts, (B, 1, H)


def sample_to_timestral_input(s: Sample):
    """
    returns:
      context: (1, T)            target history
      known:   (V, T + H) or None  covariates over context + horizon
      target:  (1, H)            ground truth
    """
    x = torch.as_tensor(s.context, dtype=torch.float32)
    y = torch.as_tensor(s.target, dtype=torch.float32)
    T, H = len(x), len(y)

    known = None
    if s.covariate is not None and np.size(s.covariate) > 0:
        c = torch.as_tensor(s.covariate, dtype=torch.float32)
        if c.ndim == 1:
            c = c[:, None]
        assert c.shape[0] >= T + H, "covariates must cover context + horizon"
        known = c[:T + H].T                      # (V, T + H)

    return x[None], known, y[None]


def samples_to_timestral_inputs(samples):
    """
    returns:
      context: (B, 1, T)
      past:    None              no past-only covariates in the samples
      known:   (B, V, T + H) or None
      target:  (B, 1, H)
    """
    xs, knowns, ys = zip(*(sample_to_timestral_input(s) for s in samples))

    assert len({x.shape[-1] for x in xs}) == 1, "all samples need the same context length"
    assert len({y.shape[-1] for y in ys}) == 1, "all samples need the same horizon"
    has_cov = {k is not None for k in knowns}
    assert len(has_cov) == 1, "don't mix samples with and without covariates"

    context = torch.stack(xs)
    known = torch.stack(knowns) if has_cov == {True} else None
    return context, None, known, torch.stack(ys)
