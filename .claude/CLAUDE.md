# CLAUDE.md

Master's thesis repo (JKU Linz): **uncertainty of time series models (TiRex / xLSTM) under distribution shift**. Research context and research questions are in [NOTES_PERSONAL.md](../NOTES_PERSONAL.md); the plan to reimplement Chronos-2 → Toto 2.0 → TiRex-2 from scratch is in [RoadMap_FoundationModels.md](../RoadMap_FoundationModels.md).

## Rules

- Avoid pandas library, instead use polars. (`pandas` is still listed in `pyproject.toml`, but don't import it in project code.)
- Plots use Altair (not matplotlib) and follow the style in `plotting.py` (`_style`, fixed colour constants).
- Keep the README's "Synthetic Data Generator" section in sync when changing the options or defaults of `SyntheticGenerator`, `ChaoticGenerator`, `RealSampler` or `Injector`.

## Environment

- Python 3.12 only (`>=3.12,<3.13`), managed with **uv**. `uv sync` installs deps plus the `ts_uncertainty` package (src layout, `uv_build` backend) into `.venv`.
- `torch` / `torchvision` come from the CUDA 13.0 index (`pytorch-cu130`; needed by `continuousintelligence`, which pins `cuda-toolkit==13.0.3`) on Linux/Windows. Importing `forecasting.py` points `CUDA_HOME` at the venv's `nvidia/cu13` toolkit (and adds unversioned `libcudart.so` / `libcublas.so` symlinks there), so TiRex-2's JIT-built flashrnn kernel links CUDA 13 instead of the system CUDA 12 — mixed with ci-proto's TileLang stub, a CUDA 12 build aborts the process.
- Run code: `uv run python ...`, or without syncing: `PYTHONPATH=src uv run --no-sync python -c '...'`.
- No test suite, linter config, or CI exists yet.

## Layout

- `src/ts_uncertainty/`
  - `sample.py` — `Sample`: common type of every data source (context (T,), target (H,), covariate (T+H, C) + names, shift type/episodes/info, spike indices, `season_length` (1 = none), source `info`); `Sample.from_series` wraps any series; `stack_samples` batches (`shift_point = -1` where no shift); `samples_to_*_inputs` convert to model inputs (statsforecast, Chronos-2, TiRex-2, Timestral, TimesFM-3, ci-proto).
  - `synthetic.py` — `SyntheticGenerator`: infinite iterator of clean series `level + slope*t + seasonal_t`, optional known-in-advance sin/cos seasonal-phase covariate over T+H.
  - `chaotic.py` — `ChaoticGenerator`: clean series from one coordinate of a dysts chaotic flow; optional dynamics shifts (`timescale`, `parameter`), which can't be injected afterwards.
  - `real.py` — `RealSampler`: random T + H windows of real univariate series from a long polars DataFrame (`timestamp`, `value`, `series_index`, `series_name`); skips windows with NaNs / timestamp gaps / constant context; z-normalises with context stats (`loc`, `scale` in `info`); `season_length` = `season` (default 1 day) / sampling step.
  - `injection.py` — `Injector`: callable `Sample -> Sample` that adds one shift (`level`, `trend`, `noise`, `amplitude`, single or recurring episodes), noise, spikes and shift/spike indicator covariates; works on any series. Also holds the shared episode helpers (`draw_episodes`, `signed`, `scale_factor`). Pipeline: `inj(next(gen))` or `map(inj, gen)`.
  - Each generator / injector draws all randomness from its own `self.rng` (seeded).
  - `plotting.py` — `plot_sample(sample)`: Altair chart with context/target, spikes, forecast-start and shift markers, hover crosshair, and optional covariate panels. Data wrangling in polars.
  - `forecasting.py` — `ForecastModel(name, size=..., quantile_levels=...)`: loads the statsforecast baselines `autoarima` / `autotheta` / `seasonalnaive` (`STAT_MODELS`) and `chronos2` / `tirex2` / `timestral` (sizes 5M/28M/123M/209M) / `timesfm3` (`TimesFM3Evaluator`, `make_positive=False` and `use_symmetric_averaging=False` by default) / `ciproto` (`continuousintelligence`, one call per sample to keep series independent; `clip_nonnegative` / `integer_round` TTA turned off) via their original libraries; the statistical models use each sample's `season_length` unless `season_length=` is given and fit series in parallel (`n_jobs`, default all cores); `forecast(samples)` returns a `QuantileForecast` with quantiles `(B, Q, H)` on a common level grid (default 0.1..0.9), `median`, `interval(c)`, and `metric_inputs()` in the time-first layout of `metrics.py` (incl. per-series `season_length`, which `sp` / `seasonal_period` of the scaled metrics accept as an array).
  - `fev_bench.py` — fev-bench (`fev` package) tasks as Samples: `load_tasks(names)`, `task_to_windows(task)` → `FevWindow` (one Sample per item × target column, full history as context unless `context_length` is set, raw values incl. NaN — all modifications only via `modify`, use `Injector(relative=True)`; full history kept for fev's metric scale; the Injector ignores NaNs; the model input is gap-filled only for models that can't take NaN, `NEEDS_FILLED_INPUT` = statsforecast models, `fill_missing=None` auto); `evaluate_task(models, task, modify=injector)` / `run_benchmark` (`modify` may return `{label: Sample}` variants, scored per `variant` column) forecast with ForecastModel and score with fev's own metric classes in raw units (identical to `task.evaluation_summary` when unmodified; MASE/SQL scale from the clean history by default) plus per-sample MASE / SQL / coverage_80.
  - `__init__.py` exports `ForecastModel`, `QuantileForecast`, `Injector`, `Sample`, `RealSampler`, `SyntheticGenerator`, `stack_samples` (`ChaoticGenerator` is imported from `ts_uncertainty.chaotic`, it pulls in dysts).
- `notebooks/` — exploration notebooks (e.g. `data_synthethic.ipynb`; uses `alt.renderers.enable("jupyter", offline=True)`).
- `thesis/tex/` — thesis in LaTeX, based on the JKU report template (`main-thesis.tex`, chapters `NN-*.tex`, `references.bib`). Compile with **XeLaTeX + biber** (see magic comments in `main-thesis.tex`). Template files (`jkureport.sty`, `ACM-Reference-Format.*`, fonts, logos) are third-party — don't edit.
- `data/`, `models/`, `presentations/` — currently empty placeholders.
- `NOTES_MEETING.md` — supervisor meeting notes.

## Conventions

- Array shapes use the README notation: B batch, V variates, T context length, H horizon, P patch size, D model dim, Q quantiles. Document shapes in comments, e.g. `# (B, T)`.
- Series values are `float32`; the default parameter ranges assume series scale ≈ 1.
- Validate configuration eagerly in `__init__` with `ValueError`s (see `SyntheticGenerator`).

## Licensing

Code: Apache 2.0 (`LICENSE`). Thesis text, docs, figures: CC BY 4.0 (`LICENSE-CC-BY-4.0`).
