# Uncertainty Quantification for Recurrent Time Series Foundation Models

This thesis investigates whether uncertainty estimates from recurrent time series foundation models, remain reliable under distribution shift.

# Notation

- (B) = batch size
- (V) = number of variates
- (T) = context length
- (H) = forecast horizon
- (P) = patch size
- (D) = model dimension
- (Q) = number of quantiles


## Starting Tasks

- Literature research: read and organize relevant papers.
- Study different time series models:
  - TiRex-2,
  - Chronos-2,
  - Toto 2.0,
  - compare them with their predecessors, and build the theoretical background needed to understand the models.
- Inspect the `fev-bench` implementations of these models.
- Study TCM, TSI, autoregressive generation, and univariate generators.
- Implement and investigate the Dyst benchmark.

## Synthetic Data Generator

`SyntheticGenerator` ([src/ts_uncertainty/synthetic.py](src/ts_uncertainty/synthetic.py)) produces an endless stream of univariate series with a known structure. This makes it possible to test forecasts and uncertainty estimates under controlled, labelled distribution shifts.

Each series has length T + H and is built as

```text
y_t = level + slope * t + seasonal_t + noise_t + spikes_t
```

It is then split into a context of length T and a forecast target of length H. All parameters (level, slope, seasonal amplitude and shape, noise level) are drawn at random for every series.

```python
from ts_uncertainty import SyntheticGenerator

gen = SyntheticGenerator(context_length=96, horizon=24, covariate=True, spikes=True, shift=True, seed=0)

s = next(gen)                 # one series
s.context                     # (T,)
s.target                      # (H,)
s.covariate                   # (T + H, 2 * n_periods) or None
s.shift_type, s.shift_point   # e.g. "trend", 114  (None if no shift)
s.spike_idx                   # indices of spikes in the T + H window

batch = gen.sample_batch(32)  # context (B, T), target (B, H), covariate (B, T + H, C) or None, shift_point (B,)
```

### Components

| Option | Default | Description |
| --- | --- | --- |
| `seasonal` | `True` | Adds `n_periods` seasonal patterns, with periods drawn from `periods` (default 7, 12, 24, 52). Each pattern is a random Fourier series with `n_harmonics` terms and a random start phase. |
| `covariate` | `False` | Returns sin/cos features of the seasonal phase over the full T + H window, so they are also known during the forecast horizon, like calendar features. They give the timing of the season but not its amplitude or shape. Requires `seasonal=True`. |
| `noise` | `True` | Gaussian observation noise, with std drawn from `noise_std_range`. |
| `spikes` | `False` | Each time step gets a spike with probability `spike_prob`. Spike heights are drawn from `spike_magnitude_range` and can be positive or negative. |
| `shift` | `False` | One abrupt distribution shift at a random change point τ (see below). |

### Distribution Shift

The change point τ is drawn from `shift_position_range`, given as a fraction of T + H (default 0.5–1.0). With the default range, a shift can therefore appear either in the context, where the model can react to it, or only in the forecast horizon, where the model cannot see it and its uncertainty estimates are put to the test. Use `shift_position_range=(T / (T + H), 1.0)` to place shifts only in the horizon.

For each series, one shift type is drawn from `shift_types`:

| Type | Effect for t ≥ τ | Magnitude |
| --- | --- | --- |
| `level` | level jumps up or down | `shift_level_range` |
| `trend` | slope changes | `shift_slope_range` |
| `noise` | noise std is multiplied or divided by a factor | `shift_scale_range` |
| `amplitude` | seasonal amplitude is multiplied or divided by a factor | `shift_scale_range` |

Shift types that need a disabled component (`noise` without noise, `amplitude` without seasonality) are skipped. The default magnitude ranges assume the series has a scale of about 1. If you change `level_range` or `amplitude_range` substantially, rescale the shift and spike ranges as well.

## Licensing

- Source code written for this project is licensed under the [Apache License 2.0](LICENSE).
- Thesis text, documentation, and figures are licensed under [CC BY 4.0](LICENSE-CC-BY-4.0).
- Third-party code and datasets remain under their respective original licenses.
