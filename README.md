# Uncertainty for Time Series Foundation Models

This thesis investigates whether uncertainty estimates from time series foundation models, remain reliable under distribution shift and check certain autoregressive methods in combination with distribution shift.

# Notation

- (B) = batch size
- (V) = number of variates
- (T) = context length
- (H) = forecast horizon
- (P) = patch size
- (D) = model dimension
- (K) = number of quantiles

## Models
These are the main models used for the thesis:

| Model | Type | Parameters | Context | Horizon | Patch size | Quantiles |
| --- | --- | --- | --- | --- | --- | --- |
| [AutoArima](https://www.researchgate.net/publication/222105759_Automatic_Time_Series_Forecasting_TheforecastPackage_forR) | local | fit per series | full history | unrestricted | - | any |
| [Chronos-2](https://github.com/amazon-science/chronos-forecasting) | zero-shot | 120M | 8192 | 1024 | 16 | 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99 |
| [TiRex-2](https://github.com/NX-AI/tirex-2) | zero-shot | 38.4M (univariate), 82.5M (multivariate) | 2048 | 320 | 32 | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |
| [Timestral](https://github.com/mistralai/timestral) | zero-shot | 5M, 28M, 123M, 209M | 32,768 | 1024 | 32 | 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99 |

The common setting for experiments that every model can cover is:
| Context | Horizon | Quantiles |
| --- | --- | --- |
| 2048 | 320 | 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 |


## Synthetic Data Generator

Data generation has two steps. First, a **generator** produces a clean series. Then an **`Injector`** perturbs it with noise, spikes and a distribution shift, and adds indicator covariates. The injector works on any series, so the same perturbations can be applied to synthetic, chaotic or real data. This makes it possible to test forecasts and uncertainty estimates under controlled, labelled distribution shifts.

| Step | Class | Module | Produces |
| --- | --- | --- | --- |
| generate | `SyntheticGenerator` | [synthetic.py](src/ts_uncertainty/synthetic.py) | `level + slope * t + seasonal_t`, optional seasonal-phase covariates |
| generate | `ChaoticGenerator` | [chaotic.py](src/ts_uncertainty/chaotic.py) | `level + slope * t + amplitude * x_t`, with `x_t` one coordinate of a dysts chaotic flow; optional dynamics shift |
| perturb | `Injector` | [injection.py](src/ts_uncertainty/injection.py) | shift, noise, spikes, shift/spike indicator covariates |

All of them work on `Sample` objects ([sample.py](src/ts_uncertainty/sample.py)). Each series has length T + H and is split into a context of length T and a forecast target of length H. All parameters are drawn at random for every series.

```python
from ts_uncertainty import Injector, Sample, SyntheticGenerator, stack_samples
from ts_uncertainty.chaotic import ChaoticGenerator

gen = SyntheticGenerator(context_length=96, horizon=24, covariate=True, seed=0)   # or ChaoticGenerator(96, 24, seed=0)
inj = Injector(spikes=True, shift=True, shift_covariate=True, seed=1)

s = inj(next(gen))            # one perturbed series
s.context                     # (T,)
s.target                      # (H,)
s.covariate                   # (T + H, C) or None, columns named in s.covariate_names
s.shift_type, s.shift_point   # e.g. "trend", 114  (None if no shift; start of the first episode)
s.shift_episodes              # e.g. ((114, 120),), (start, end) of each shift episode, end exclusive
s.shift_info                  # magnitude, e.g. {"delta_slope": 0.03}
s.spike_idx                   # indices of spikes in the T + H window
s.info                        # source metadata, e.g. {"periods": (24,)} or {"system": "Lorenz", ...}

stream = map(inj, gen)        # endless stream of perturbed series
batch = stack_samples([inj(next(gen)) for _ in range(32)])  # context (B, T), target (B, H), covariate (B, T + H, C) or None, shift_point (B,), shift_mask (B, T + H)

real = inj(Sample.from_series(y, context_length=96))        # any (T + H,) series
```

### SyntheticGenerator

| Option | Default | Description |
| --- | --- | --- |
| `seasonal` | `True` | Adds `n_periods` seasonal patterns, with periods drawn from `periods` (default 7, 12, 24, 52). Each pattern is a random Fourier series with `n_harmonics` terms and a random start phase. |
| `covariate` | `False` | Returns sin/cos features of the seasonal phase over the full T + H window, so they are also known during the forecast horizon, like calendar features. They give the timing of the season but not its amplitude or shape. Requires `seasonal=True`. |
| `level_range`, `slope_range`, `amplitude_range` | `(-1, 1)`, `(-0.01, 0.01)`, `(0.5, 2)` | Ranges of the constant level, the slope per time step and the seasonal amplitude per period. |

### ChaoticGenerator

Takes one coordinate of a chaotic ODE system from [dysts](https://github.com/williamgilpin/dysts) (Lorenz, Rössler, …, see `DEFAULT_SYSTEMS`), z-normalises it with the context statistics and scales it by `amplitude_range`. The system, the observed coordinate (`dim`), the sampling rate (`pts_per_period_range`) and the initial condition are drawn per series.

Two shifts change the dynamics themselves. They can't be applied to a finished series, so they stay in the generator (`shift=True`, `shift_types`):

| Type | Effect while the shift is active | Magnitude |
| --- | --- | --- |
| `timescale` | the system runs faster or slower | `shift_scale_range` |
| `parameter` | one system parameter is perturbed; the trajectory continues under the new equations | `shift_param_range` (relative) |

The episodes are drawn as for the injector (see below). A sample that already has a dynamics shift keeps it, and the injector adds no second shift.

### Injector

| Option | Default | Description |
| --- | --- | --- |
| `noise` | `True` | Gaussian observation noise, with std drawn from `noise_std_range`. |
| `spikes` | `False` | Each time step gets a spike with probability `spike_prob`. Spike heights are drawn from `spike_magnitude_range` and can be positive or negative. |
| `shift` | `False` | One abrupt distribution shift at a random change point τ (see below). |
| `shift_covariate` | `False` | Adds an indicator covariate over T + H that is 1 while a shift episode is active (all zeros if the series has no shift). It marks when the regime changes, but not the shift type or magnitude. It also marks dynamics shifts from `ChaoticGenerator`. |
| `spike_covariate` | `False` | Adds a pulse-indicator covariate that is 1 at spike positions over T + H, like a known event calendar. It does not give the spike sign or height. Requires `spikes=True`. |

Indicator covariates are appended after any covariates the source already returned.

### Distribution Shift

The change point τ is drawn from `shift_position_range`, given as a fraction of T + H (default 0.5–1.0). With the default range, a shift can therefore appear either in the context, where the model can react to it, or only in the forecast horizon, where the model cannot see it and its uncertainty estimates are put to the test. Use `shift_position_range=(T / (T + H), 1.0)` to place shifts only in the horizon.

For each series, the injector draws one shift type from `shift_types`:

| Type | Effect while the shift is active | Magnitude |
| --- | --- | --- |
| `level` | level jumps up or down | `shift_level_range` |
| `trend` | slope changes | `shift_slope_range` |
| `noise` | noise std is multiplied or divided by a factor | `shift_scale_range` |
| `amplitude` | deviation from the linear trend (fitted outside the shift) is multiplied or divided by a factor | `shift_scale_range` |

`noise` with `noise=False` adds noise only during the shift. `amplitude` on a series without a periodic or chaotic component has almost no effect. The default magnitude ranges assume the series has a scale of about 1. For other series, standardise first, or rescale the shift and spike ranges.

`inj(sample, shift_point=0.6, shift_type="level")` forces a specific shift on one sample, even with `shift=False`.

#### Recurring Shifts

With probability `shift_recurring_prob` (default 0), a shift is recurring instead of persistent. The shift then has two episodes with the same type and magnitude:

1. The first episode lies entirely inside the context. Its length is drawn from `shift_duration_range`, given as a fraction of T (default 0.1–0.25). The series then returns to normal for at least as long as the episode lasted.
2. The second episode starts at a random step in the horizon and lasts equally long, unless the window ends first.

This tests whether a model learns the shift from the context and expects it again in the horizon. During a `trend` episode, the ramp starts from 0 at each episode start, and the series drops back to its normal path when the episode ends.

For a single sample, `inj(s, recurring=True)` forces a recurring shift and `inj(s, recurring=False)` forces a persistent one. `inj(s, shift_episodes=[(0.1, 0.2), (0.85, 1.0)])` places episodes explicitly as fractions of T + H; they must be sorted and must not overlap. `ChaoticGenerator.sample` takes the same arguments for dynamics shifts.


## Experiments

Each experiment should be done with the synthetic, chaotic generator and on a real time series.
Gneral question: does more context help?

### Noise

#### Univariate case without covariates

Only a single univariate time series.

1. Inject noise only into the forecast horizon. Assumption: models can not detect noise. Never appear in context. Uncertainty should stay same but error increase.
2. Inject noise only into the context horizon. Assumption: models will halucinate and predict noise. Uncertainty should increase.
3. Inject noise into context and forecast horizon. Assumption: models should detect noise. Uncertainty should increase due to noise.

### Spikes

#### Univariate case without covariates

Only a single univariate time series.

1. Inject spikes only into the forecast horizon. Assumption: models can not detect spikes. Never appear in context. Uncertainty should stay same.
2. Inject spikes only into the context horizon. Assumption: models will halucinate and predict random spikes. Uncertainty should increase.
3. Inject spikes into context and forecast horizon. (at a defined period). Assumption: models should detect spikes. Uncertainty should decrease at spikes.
4. Inject spikes into context and forecast horizon. (at random timesteps). Assumption: models can not detect it. Uncertainty should increase.

#### Univariate case with covariates

A single univariate time series with a covariate (1/0) for spikes.

1. Inject spikes only into the forecast horizon. Assumption: models can detect spikes with the covariate. Uncertainty should decrease.
2. Inject spikes only into the context horizon. Assumption: models will know there are not spikes due to covariate. Uncertainty should decrease.
3. Inject spikes into context and forecast horizon. (at a defined period). Assumption: models should detect spikes. Uncertainty should decrease at spikes.
4. Inject spikes into context and forecast horizon. (at random timesteps). Assumption: models will know there are spikes due to covariate. Uncertainty should decrease.

### Continous Shift

#### Univariate case without covariates

Only a single univariate time series.

1. Inject shift only into the forecast horizon. Assumption: models can not detect shift. Never appear in context. Uncertainty should stay same.
2. Inject shift only into the context horizon. Assumption: models will halucinate and predict shift which then not appear. Uncertainty should increase.
3. Inject shift into context and forecast horizon. Assumption: models should detect shift. Uncertainty should decrease.

#### Univariate case with covariates

A single univariate time series with a covariate for shift.

1. Inject shift only into the forecast horizon. Assumption: models can detect shift with the covariate. Uncertainty should decrease.
2. Inject shift only into the context horizon. Assumption: models will know there is no shift due to covariate. Uncertainty should increase.
3. Inject shift into context and forecast horizon. Assumption: models should detect shift. Uncertainty should decrease.


### Autoregressive prediction

Check different autoregressive methods + injection of noise, spikes and shift.
How the models will react to this? Quantile collapse or uncertainty increase?

- Median
- Mean
- Autoregressive multi-quantile decoding (Morai 2.0)
- Only xlstm: keep state from previous generation. (or out of scope: fix state of real context and use it as streaming state)


## Licensing

- Source code written for this project is licensed under the [Apache License 2.0](LICENSE).
- Thesis text, documentation, and figures are licensed under [CC BY 4.0](LICENSE-CC-BY-4.0).
- Parts of the code are adapted from third-party projects. See [NOTICE](NOTICE) for the full
  list of attributions.
- Datasets and pretrained weights are not part of this repository and are subject
  to their own licenses.
