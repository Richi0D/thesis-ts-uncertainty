"""fev-bench tasks as Samples, with optional modifications (Injector), evaluated with ForecastModel.

Every evaluation window of an fev Task becomes one list of Samples: one Sample per
(series, target column), so multivariate tasks are forecast univariately, one target at a time.
A Sample's context is the full history up to the cutoff (only the last `context_length` steps if set),
its target is the horizon, and `info` holds where it came from (task, window, item_id,
target_column, cutoff). The values are the raw fev values (no normalisation, missing values stay
NaN): every modification is left to `modify`, e.g. an Injector. fev series are not on a common
scale, so use `Injector(..., relative=True)` to make the injected magnitudes relative to the
context std of each series.

    tasks = load_tasks(["hospital", "ETT_1H"])
    windows = task_to_windows(tasks[0])                      # list[FevWindow]
    s = windows[0].samples[0]                                # a Sample, like from RealSampler

    models = {"tirex2": {"name": "tirex2"}, "seasonalnaive": {"name": "seasonalnaive"}}
    inj = Injector(shift=True, shift_types=("trend",), relative=True, seed=0)
    res = evaluate_task(models, tasks[0], modify=inj)        # TaskResult
    res.summary                                              # fev-bench metrics per model
    res.per_sample                                           # MASE / SQL / coverage per Sample

    # several modifications per sample: return {label: Sample}, scored per label ("variant")
    sweep = lambda s: {sp: inj(s, shift_point=sp, shift_type="trend") for sp in (0.5, 0.7, 0.9)}
    res = evaluate_task(models, tasks[0], modify=sweep, run_name="trend_sweep")

    bench = run_benchmark(models, load_tasks(), modify=inj)  # all 100 fev-bench tasks
    fev.leaderboard(bench.fev_summaries, baseline_model="seasonalnaive")

Metrics: the task's own fev metrics (eval_metric + extra_metrics, e.g. SQL, MASE, WAPE, WQL)
are computed with fev's metric classes, on the raw values and with fev's
aggregation (per window over all series, then averaged over windows). Without a modification
they equal `task.evaluation_summary` of the same forecasts, i.e. the official fev-bench numbers.
With a modification the ground truth is the modified target, and the scale of MASE / SQL comes
from the clean history (`scale_from="clean"`), so an injection in the context doesn't change the
denominator. `per_sample` holds the same scaled errors per Sample (from `metrics.py`, scale from
the Sample's context, not the full history).

What a Sample doesn't carry is left out: past-only dynamic covariates, static covariates and
non-numeric known covariates. Known numeric covariates are only used with `covariates=True`.
Missing values stay NaN in the Samples and are ignored everywhere: by the Injector (its scale
and fits use the observed values only, a missing value stays missing) and by the metrics. Only
the model input of models that can't handle NaN gets a filled copy (forward, then backward fill):
by default (`fill_missing=None`) only the statsforecast models (NEEDS_FILLED_INPUT); the neural
models get the NaNs as they are and treat them as missing.
"""

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import fev
import numpy as np
import polars as pl
from fev.metrics import get_metric

from ts_uncertainty.forecasting import STAT_MODELS, ForecastModel, QuantileForecast
from ts_uncertainty.metrics import seasonal_scale
from ts_uncertainty.sample import Sample

FEV_BENCH_URL = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks.yaml"
FEV_BENCH_MINI_URL = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/tasks_mini.yaml"
FEV_BENCH_EXAMPLE_URL = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/example/tasks.yaml"
SCALE_FROM = ("clean", "modified")
# Models whose input conversion stacks the contexts, so one forecast call needs equal context lengths.
EQUAL_CONTEXT_MODELS = ("timestral",)
# Models that don't support NaN in their input (statsforecast: AutoTheta fails to fit, AutoARIMA
# fails on NaN covariates, SeasonalNaive repeats a NaN). Chronos-2, TiRex-2, Timestral, TimesFM-3
# and ci-proto forecast fine with NaNs in context and covariates.
NEEDS_FILLED_INPUT = tuple(STAT_MODELS)

# One modified Sample, or several variants of it: a list (labels 0..K-1) or {label: Sample}.
Modification = Callable[[Sample], Sample | Sequence[Sample] | Mapping[object, Sample]]


def load_tasks(names: Sequence[str] | None = None, source: str = FEV_BENCH_URL) -> list[fev.Task]:
    """fev Tasks of a benchmark YAML (URL or path), by default fev-bench (100 tasks).

    names: keep only these task names (Task.task_name, = dataset_config unless set), in this order.
    """
    tasks = fev.Benchmark.from_yaml(source).tasks
    if names is None:
        return tasks
    by_name = {t.task_name: t for t in tasks}
    if missing := [n for n in names if n not in by_name]:
        raise ValueError(f"unknown task names {missing}, available: {sorted(by_name)}")
    return [by_name[n] for n in names]


@dataclass
class FevWindow:
    """One evaluation window of a task as Samples.

    samples: one per (item, target column), item-major (all targets of item 0, then item 1, ...),
        raw fev values (NaN where missing), ready for ForecastModel.forecast or an Injector.
    past:    per sample, the full history before the cutoff (L_i,), of which the context is the
        last T steps; fev scales MASE / SQL with it.
    item_ids, target_columns: the N items and D target columns, len(samples) == N * D.
    """

    task_name: str
    window: int
    cutoff: int | str
    samples: list[Sample]
    past: list[np.ndarray] = field(repr=False)
    item_ids: list[str] = field(repr=False)
    target_columns: list[str]


def task_to_windows(
    task: fev.Task,
    context_length: int | None = None,
    covariates: bool = False,
    windows: Sequence[int] | None = None,
) -> list[FevWindow]:
    """Convert the evaluation windows of `task` (all, or the given indices) to Samples.

    context_length: keep only the last T steps of the history as context; None (default) keeps the
        full history, each model then cuts it to its own maximum context. fev's MASE / SQL scale
        always uses the full history.
    covariates: add the task's numeric known dynamic columns as Sample.covariate (T + H, C),
        raw like the target (NaN where missing).
    """
    if context_length is not None and context_length < 1:
        raise ValueError(f"context_length must be >= 1 or None, got {context_length}")
    indices = range(task.num_windows) if windows is None else windows
    if any(not 0 <= i < task.num_windows for i in indices):
        raise ValueError(f"window indices must be in [0, {task.num_windows}), got {list(indices)}")
    return [
        window_to_samples(task, i, context_length=context_length, covariates=covariates)
        for i in indices
    ]


def window_to_samples(
    task: fev.Task,
    window: int,
    context_length: int | None = None,
    covariates: bool = False,
) -> FevWindow:
    """Convert one evaluation window of `task` to Samples, see task_to_windows."""
    fev_window = task.get_window(window)
    past_ds, future_ds = fev_window.get_input_data()
    truth_ds = fev_window.get_ground_truth()
    past = pl.from_arrow(past_ds.data.table)
    future = pl.from_arrow(future_ds.data.table)
    truth = pl.from_arrow(truth_ds.data.table)
    id_col, targets = task.id_column, task.target_columns
    if not (past[id_col].to_list() == future[id_col].to_list() == truth[id_col].to_list()):
        raise RuntimeError("fev returned past / future / ground truth in a different item order")

    cov_cols = _numeric_list_columns(future, task.known_dynamic_columns) if covariates else []
    samples, raw_past = [], []
    for i, item_id in enumerate(past[id_col].to_list()):
        cov = None
        if cov_cols:
            cov = np.stack(  # (L + H, C)
                [np.concatenate([_row(past, c, i), _row(future, c, i)]) for c in cov_cols], axis=1
            )
        for col in targets:
            history = _row(past, col, i)  # (L,)
            y_true = _row(truth, col, i)  # (H,)
            T = len(history) if context_length is None else min(context_length, len(history))
            samples.append(
                Sample.from_series(
                    np.concatenate([history[-T:], y_true]),
                    context_length=T,
                    covariate=None if cov is None else cov[len(history) - T :].astype(np.float32),
                    covariate_names=cov_cols,
                    season_length=task.seasonality,
                    task=task.task_name,
                    window=window,
                    cutoff=fev_window.cutoff,
                    item_id=str(item_id),
                    target_column=col,
                    n_missing_context=int(np.isnan(history[-T:]).sum()),
                )
            )
            raw_past.append(history)
    return FevWindow(
        task_name=task.task_name,
        window=window,
        cutoff=fev_window.cutoff,
        samples=samples,
        past=raw_past,
        item_ids=[str(x) for x in past[id_col].to_list()],
        target_columns=list(targets),
    )


@dataclass
class TaskResult:
    """Evaluation of several models on one task.

    summary:      one row per (variant, model): task, run, variant, model, test_error
                  (= task.eval_metric), every fev metric of the task averaged over windows,
                  inference_time_s (all variants together), n_samples (per variant).
    per_window:   the fev metrics per (variant, model, window).
    per_sample:   one row per (variant, model, window, sample): item_id, target_column, mase, sql,
                  coverage_80 (share of the horizon inside q0.1–q0.9, if both levels exist),
                  shift_type, shift_point, context_length.
    fev_summaries: dicts in the format of task.evaluation_summary, for fev.leaderboard.
    variant is the label of the modification (key / index of what modify returned), None when
    modify returns a single Sample or is None.
    windows:      the clean FevWindows.
    modified:     per window, {variant: Samples the models saw}.
    forecasts:    model -> per window {variant: QuantileForecast}, if keep_forecasts.
    """

    summary: pl.DataFrame
    per_window: pl.DataFrame
    per_sample: pl.DataFrame
    fev_summaries: list[dict]
    windows: list[FevWindow] = field(repr=False)
    modified: list[dict[object, list[Sample]]] = field(repr=False)
    forecasts: dict[str, list[dict[object, QuantileForecast]]] | None = field(default=None, repr=False)


def evaluate_task(
    models: Mapping[str, ForecastModel | Mapping],
    task: fev.Task,
    modify: Modification | None = None,
    context_length: int | None = None,
    covariates: bool = False,
    windows: Sequence[int] | None = None,
    scale_from: str = "clean",
    fill_missing: bool | None = None,
    keep_forecasts: bool = False,
    run_name: str | None = None,
    **forecast_kwargs,
) -> TaskResult:
    """Forecast every window of `task` with every model and score it with the task's fev metrics.

    models: name -> ForecastModel, or name -> ForecastModel kwargs (loaded here, once).
    modify: applied to every clean sample before forecasting, e.g. an Injector (must keep NaNs
        for missing values, as the Injector does; applied in order, so a seeded Injector gives the
        same modifications on every run). It returns one Sample, or several variants of it as a
        list or {label: Sample} (same labels for every sample), e.g.
            lambda s: {sp: inj(s, shift_point=sp, shift_type="trend") for sp in shift_points}
        Every variant is scored on its own (column "variant"); the variants of one window are
        forecast in one call per model.
    scale_from: history that scales MASE / SQL: "clean" (before modify) or "modified".
    fill_missing: fill NaNs of the context / covariates in the model input, see forecast_samples;
        None (default) only for the models in NEEDS_FILLED_INPUT, True / False for every model.
    run_name: label of the modification in the outputs (column "run" / extra_info), default
        "clean" without modify, else the class name of modify.
    windows, context_length, covariates: see task_to_windows.
    forecast_kwargs: passed to every ForecastModel.forecast call.
    """
    if scale_from not in SCALE_FROM:
        raise ValueError(f"scale_from must be one of {SCALE_FROM}, got {scale_from!r}")
    models = _load_models(models)
    for name, model in models.items():
        levels = model.quantile_levels or model.native_quantile_levels
        if missing := [q for q in task.quantile_levels if not np.isclose(levels, q).any()]:
            raise ValueError(f"model {name} has no quantile levels {missing} that task {task.task_name} needs")
    run = run_name or ("clean" if modify is None else type(modify).__name__)
    metrics = [get_metric(m) for m in [task.eval_metric, *task.extra_metrics]]

    fev_windows = task_to_windows(task, context_length, covariates, windows)
    modified = [_apply_modify(modify, w.samples) for w in fev_windows]  # per window {variant: samples}

    window_rows, sample_rows = [], []
    forecasts = {name: [] for name in models} if keep_forecasts else None
    inference_time = dict.fromkeys(models, 0.0)
    for w, variants in zip(fev_windows, modified):
        batch = [s for samples in variants.values() for s in samples]  # all variants, variant-major
        B = len(w.samples)
        for name, model in models.items():
            start = time.perf_counter()
            fc_all = forecast_samples(model, batch, fill_missing=fill_missing, **forecast_kwargs)
            inference_time[name] += time.perf_counter() - start
            fcs = {label: _take(fc_all, slice(k * B, (k + 1) * B)) for k, label in enumerate(variants)}
            if keep_forecasts:
                forecasts[name].append(fcs)
            for label, samples in variants.items():
                fc = fcs[label]
                scores = _fev_scores(task, w, samples, fc, metrics, scale_from)
                window_rows.append(
                    {"task": task.task_name, "run": run, "variant": label, "model": name, "window": w.window, **scores}
                )
                sample_rows.append(_sample_scores(task, w, samples, fc, scale_from, run, name, label))

    keys = ("task", "run", "variant", "model")
    per_window = pl.DataFrame(window_rows, strict=False)
    metric_cols = [c for c in per_window.columns if c not in (*keys, "window")]
    n_samples = sum(len(w.samples) for w in fev_windows)  # per variant
    summary = (
        per_window.group_by(*keys, maintain_order=True)
        .agg(pl.col(metric_cols).mean())
        .with_columns(
            test_error=pl.col(metrics[0].name),
            inference_time_s=pl.col("model").replace_strict(inference_time),
            n_samples=pl.lit(n_samples),
        )
        .select(*keys, "test_error", *metric_cols, "inference_time_s", "n_samples")
    )
    fev_summaries = [
        {
            "model_name": row["model"],
            **task.to_dict(),
            "test_error": row["test_error"],
            "training_time_s": None,
            "inference_time_s": row["inference_time_s"],
            "num_forecasts": n_samples,
            "dataset_fingerprint": getattr(task, "_dataset_fingerprint", None),
            "trained_on_this_dataset": False,
            "fev_version": fev.__version__,
            **{c: row[c] for c in metric_cols},
            "run": run,
            "variant": row["variant"],
            "context_length": context_length,
            "covariates": covariates,
        }
        for row in summary.iter_rows(named=True)
    ]
    return TaskResult(
        summary=summary,
        per_window=per_window,
        per_sample=pl.concat(sample_rows, how="diagonal_relaxed"),
        fev_summaries=fev_summaries,
        windows=fev_windows,
        modified=modified,
        forecasts=forecasts,
    )


@dataclass
class BenchmarkResult:
    """TaskResults of several tasks; summary / per_window / per_sample are concatenated."""

    summary: pl.DataFrame
    per_window: pl.DataFrame
    per_sample: pl.DataFrame
    fev_summaries: list[dict]
    tasks: dict[str, TaskResult] = field(repr=False)
    failed: dict[str, str] = field(default_factory=dict)  # task name -> error message


def run_benchmark(
    models: Mapping[str, ForecastModel | Mapping],
    tasks: Iterable[fev.Task] | None = None,
    modify: Modification | None = None,
    skip_errors: bool = False,
    keep_task_results: bool = False,
    **kwargs,
) -> BenchmarkResult:
    """evaluate_task for every task (default: all fev-bench tasks), loading each model once.

    skip_errors: record a failing task in `failed` and go on instead of raising.
    keep_task_results: keep each TaskResult (with all Samples; memory-heavy on the large tasks).
    kwargs: passed to evaluate_task (context_length, covariates, scale_from, ...).
    """
    models = _load_models(models)
    tasks = load_tasks() if tasks is None else list(tasks)
    results, failed = {}, {}
    for i, task in enumerate(tasks):
        print(f"[{i + 1}/{len(tasks)}] {task.task_name}")
        try:
            results[task.task_name] = evaluate_task(models, task, modify=modify, **kwargs)
        except Exception as e:
            if not skip_errors:
                raise
            failed[task.task_name] = f"{type(e).__name__}: {e}"
            print(f"  failed: {failed[task.task_name]}")
    done = list(results.values())

    def cat(attr):
        frames = [getattr(r, attr) for r in done]
        return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()

    return BenchmarkResult(
        summary=cat("summary"),
        per_window=cat("per_window"),
        per_sample=cat("per_sample"),
        fev_summaries=[s for r in done for s in r.fev_summaries],
        tasks=results if keep_task_results else {},
        failed=failed,
    )


def forecast_samples(
    model: ForecastModel, samples: Sequence[Sample], fill_missing: bool | None = None, **kwargs
) -> QuantileForecast:
    """model.forecast(samples); models that need equal context lengths get one call per length.

    fill_missing: give the model a copy of each sample with NaNs in the context and covariates
        filled (forward, then backward; all-NaN -> 0). The samples themselves are not changed.
        None fills only for models in NEEDS_FILLED_INPUT (the statsforecast models).
    The returned QuantileForecast keeps the sample order; its context and target are those of
    the samples (NaN where missing), the context cut to the common tail (shortest T).
    """
    originals = samples
    if fill_missing is None:
        fill_missing = model.name in NEEDS_FILLED_INPUT
    if fill_missing:
        samples = [_fill_sample(s) for s in samples]
    fc = _forecast_grouped(model, samples, **kwargs)
    T = fc.context.shape[1]
    return replace(
        fc,
        context=np.stack([s.context[-T:] for s in originals]),
        target=np.stack([s.target for s in originals]),
    )


def _fill_sample(s: Sample) -> Sample:
    """Copy of s with NaNs in context and covariates filled, or s itself without NaNs."""
    cov_missing = s.covariate is not None and np.isnan(s.covariate).any()
    if not (np.isnan(s.context).any() or cov_missing):
        return s
    covariate = s.covariate
    if cov_missing:
        covariate = np.stack([_fill(c) for c in s.covariate.T], axis=1).astype(np.float32)
    return replace(s, context=_fill(s.context).astype(np.float32), covariate=covariate)


def _forecast_grouped(model: ForecastModel, samples: Sequence[Sample], **kwargs) -> QuantileForecast:
    lengths = np.array([len(s.context) for s in samples])
    if model.name not in EQUAL_CONTEXT_MODELS or len(np.unique(lengths)) == 1:
        return model.forecast(samples, **kwargs)
    parts = {
        T: (idx := np.flatnonzero(lengths == T), model.forecast([samples[i] for i in idx], **kwargs))
        for T in np.unique(lengths)
    }
    first = next(iter(parts.values()))[1]
    quantiles = np.empty((len(samples), *first.quantiles.shape[1:]), dtype=np.float32)  # (B, Q, H)
    for idx, fc in parts.values():
        quantiles[idx] = fc.quantiles
    T = int(lengths.min())
    return QuantileForecast(
        model=first.model,
        quantiles=quantiles,
        quantile_levels=first.quantile_levels,
        context=np.stack([s.context[-T:] for s in samples]),
        target=np.stack([s.target for s in samples]),
        season_length=np.array([s.season_length for s in samples]),
    )


def _apply_modify(modify: Modification | None, samples: list[Sample]) -> dict[object, list[Sample]]:
    """{variant label: modified samples}; {None: samples} without modify or for one Sample each."""
    if modify is None:
        return {None: samples}
    outs = [modify(s) for s in samples]
    if all(isinstance(o, Sample) for o in outs):
        return {None: outs}
    as_dicts = [dict(o) if isinstance(o, Mapping) else dict(enumerate(o)) for o in outs]
    labels = list(as_dicts[0])
    if not labels:
        raise ValueError("modify returned no variants")
    if any(list(d) != labels for d in as_dicts):
        raise ValueError("modify must return the same variant labels (in the same order) for every sample")
    if any(not isinstance(x, Sample) for d in as_dicts for x in d.values()):
        raise ValueError("modify must return a Sample, a list of Samples or {label: Sample}")
    return {label: [d[label] for d in as_dicts] for label in labels}


def _take(fc: QuantileForecast, idx: slice) -> QuantileForecast:
    """The forecasts of a subset of the series."""
    return replace(
        fc,
        quantiles=fc.quantiles[idx],
        context=fc.context[idx],
        target=fc.target[idx],
        season_length=None if fc.season_length is None else fc.season_length[idx],
    )


def _load_models(models: Mapping[str, ForecastModel | Mapping]) -> dict[str, ForecastModel]:
    loaded = {}
    for name, m in models.items():
        if not isinstance(m, ForecastModel):
            print(f"Loading model: {name}")
            m = ForecastModel(**m)
        loaded[name] = m
    return loaded


def _fev_scores(task, w: FevWindow, samples, fc: QuantileForecast, metrics, scale_from) -> dict[str, float]:
    """fev metrics of one window: y_true (N, H, D), q_pred (N, H, D, Q), ragged y_past."""
    N, D = len(w.item_ids), len(w.target_columns)
    y_true = np.stack([s.target for s in samples])  # (B, H)
    q = np.stack([fc.quantile(level) for level in task.quantile_levels], axis=-1)  # (B, H, Q)
    y_pred = fc.median  # (B, H)

    past = w.past
    if scale_from == "modified":  # history with its last T steps replaced by the modified context
        past = [
            np.concatenate([p[: len(p) - len(s.context)], s.context])
            for p, s in zip(w.past, samples)
        ]
    # samples are item-major: sample b = item b // D, target column b % D
    lengths = np.array([len(p) for p in past]).reshape(N, D)
    if (lengths != lengths[:, :1]).any():
        raise RuntimeError("target columns of one item have histories of different length")
    y_past = np.concatenate([np.stack(past[i * D : (i + 1) * D], axis=1) for i in range(N)])  # (sum L, D)

    def items_first(x):  # (B, H, ...) -> (N, H, D, ...)
        return np.moveaxis(x.reshape(N, D, *x.shape[1:]), 1, 2).astype(np.float64)

    scores = {}
    for metric in metrics:
        scores |= metric.compute_scores(
            y_true=items_first(y_true),
            y_pred=items_first(y_pred),
            y_past=y_past.astype(np.float64),
            y_past_lengths=lengths[:, 0],
            q_pred=items_first(q),
            seasonality=task.seasonality,
            quantile_levels=list(task.quantile_levels),
        )
    return scores


def _sample_scores(task, w: FevWindow, samples, fc: QuantileForecast, scale_from, run, model, variant) -> pl.DataFrame:
    """Per-sample MASE / SQL (metrics.py conventions, scale from the sample's context) and coverage."""
    contexts = [c.context for c in (w.samples if scale_from == "clean" else samples)]
    scale = np.array([seasonal_scale(c[:, None], s.season_length)[0] for c, s in zip(contexts, samples)])  # (B,)
    y = np.stack([s.target for s in samples])  # (B, H)
    levels = fc.quantile_levels
    diff = y[:, None, :] - fc.quantiles  # (B, Q, H)
    ql = 2.0 * np.maximum(levels[None, :, None] * diff, (levels[None, :, None] - 1) * diff).mean(axis=1)  # (B, H)
    cols = {
        "task": task.task_name,
        "run": run,
        "variant": [variant] * len(samples),
        "model": model,
        "window": w.window,
        "item_id": [s.info["item_id"] for s in samples],
        "target_column": [s.info["target_column"] for s in samples],
        "mase": np.nanmean(np.abs(y - fc.median), axis=1) / scale,
        "sql": np.nanmean(ql, axis=1) / scale,
    }
    if np.isclose(levels, 0.1).any() and np.isclose(levels, 0.9).any():
        lo, hi = fc.interval(0.8)
        inside = np.where(np.isnan(y), np.nan, (y >= lo) & (y <= hi))
        cols["coverage_80"] = np.nanmean(inside, axis=1)
    cols |= {
        "shift_type": [s.shift_type for s in samples],
        "shift_point": [-1 if s.shift_point is None else s.shift_point for s in samples],
        "context_length": [len(s.context) for s in samples],
    }
    return pl.DataFrame(cols, strict=False)


def _row(df: pl.DataFrame, col: str, i: int) -> np.ndarray:
    """Row i of a list column as float64 (nulls -> NaN)."""
    return df[col][i].cast(pl.Float64).fill_null(np.nan).to_numpy()


def _numeric_list_columns(df: pl.DataFrame, columns: Sequence[str]) -> list[str]:
    """The given list columns with a numeric or boolean inner type (others can't be covariates)."""
    keep = []
    for c in columns:
        inner = df.schema[c].inner
        if inner.is_numeric() or inner == pl.Boolean:
            keep.append(c)
    return keep


def _fill(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs, then backward-fill the leading ones; all-NaN -> zeros."""
    x = np.asarray(x, dtype=np.float64)
    missing = np.isnan(x)
    if not missing.any():
        return x
    if missing.all():
        return np.zeros_like(x)
    idx = np.where(missing, 0, np.arange(len(x)))
    np.maximum.accumulate(idx, out=idx)
    out = x[idx]
    first = int(np.argmax(~missing))
    out[:first] = x[first]
    return out
