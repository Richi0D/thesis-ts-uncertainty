# CLAUDE.md

Master's thesis repo (JKU Linz): **uncertainty quantification of recurrent time series foundation models (TiRex / xLSTM) under distribution shift**. Research context and research questions are in [NOTES_PERSONAL.md](../NOTES_PERSONAL.md); the plan to reimplement Chronos-2 → Toto 2.0 → TiRex-2 from scratch is in [RoadMap_FoundationModels.md](../RoadMap_FoundationModels.md).

## Rules

- Avoid pandas library, instead use polars. (`pandas` is still listed in `pyproject.toml`, but don't import it in project code.)
- Plots use Altair (not matplotlib) and follow the style in `plotting.py` (`_style`, fixed colour constants).
- Keep the README's "Synthetic Data Generator" section in sync when changing `SyntheticGenerator`'s options or defaults.

## Environment

- Python 3.12 only (`>=3.12,<3.13`), managed with **uv**. `uv sync` installs deps plus the `ts_uncertainty` package (src layout, `uv_build` backend) into `.venv`.
- `torch` / `torchvision` come from the CUDA 12.6 index (`pytorch-cu126`) on Linux/Windows.
- Run code: `uv run python ...`, or without syncing: `PYTHONPATH=src uv run --no-sync python -c '...'`.
- No test suite, linter config, or CI exists yet.

## Layout

- `src/ts_uncertainty/`
  - `synthetic.py` — `SyntheticGenerator` / `SyntheticSample`: infinite iterator of univariate series `y_t = level + slope*t + seasonal_t + noise_t + spikes_t`, split into context (T) and target (H). Optional known-in-advance sin/cos seasonal-phase covariate over T+H, spikes, and one abrupt distribution shift (`level`, `trend`, `noise`, `amplitude`) at change point τ. `sample(shift_point=<fraction of T+H>, shift_type=...)` forces a specific shift; `sample_batch(B)` returns a dict of stacked numpy arrays (`shift_point = -1` where no shift). All randomness goes through `self.rng` (seeded).
  - `plotting.py` — `plot_synthetic_sample(sample)`: Altair chart with context/target, spikes, forecast-start and shift markers, hover crosshair, and optional covariate panels. Data wrangling in polars.
  - `__init__.py` exports only `SyntheticGenerator`, `SyntheticSample`.
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
