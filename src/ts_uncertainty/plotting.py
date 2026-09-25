"""Plotting utilities (Altair)."""

import altair as alt
import numpy as np
import polars as pl

from ts_uncertainty.synthetic import SyntheticSample

# categorical slots in fixed order, colour follows the entity
SERIES_COLORS = {"context": "#2a78d6", "target": "#eb6834", "spike": "#1baf7a"}
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
MUTED = "#8a8984"


def plot_synthetic_sample(
    sample: SyntheticSample,
    title: str | None = None,
    width: int = 640,
    height: int = 260,
    show_covariate: bool = True,
) -> alt.VConcatChart | alt.LayerChart:
    """Plot context, forecast target, spikes, forecast start and shift point of one sample.

    If the sample has a covariate and show_covariate is True, each covariate feature is
    drawn in its own small panel below the series, sharing the time axis.
    """
    context_length = len(sample.context)
    y = np.concatenate([sample.context, sample.target])
    df = pl.DataFrame({"t": np.arange(len(y)), "value": y}).with_columns(
        segment=pl.when(pl.col("t") < context_length).then(pl.lit("context")).otherwise(pl.lit("target")),
        spike=pl.col("t").is_in(sample.spike_idx.tolist()),
    )

    x = alt.X("t:Q", title="time step", scale=alt.Scale(domain=[0, len(y) - 1], nice=False))
    series = ["context", "target"] + (["spike"] if len(sample.spike_idx) else [])
    color = alt.Color(
        "segment:N",
        title=None,
        scale=alt.Scale(domain=series, range=[SERIES_COLORS[s] for s in series]),
        legend=alt.Legend(orient="top", direction="horizontal", symbolType="stroke"),
    )
    base = alt.Chart(df).encode(x=x)

    lines = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(
        y=alt.Y("value:Q", title="value"), color=color
    )
    spikes = (
        base.transform_filter(alt.datum.spike)
        .transform_calculate(segment="'spike'")
        .mark_point(filled=True, size=80, stroke=SURFACE, strokeWidth=2, opacity=1)
        .encode(y="value:Q", color=color)
    )

    # crosshair: nearest time step under the pointer
    hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", clear="pointerout", empty=False)
    tooltip = [
        alt.Tooltip("t:Q", title="t"),
        alt.Tooltip("segment:N", title="segment"),
        alt.Tooltip("value:Q", title="value", format=".3f"),
        alt.Tooltip("spike:N", title="spike"),
    ]
    hit_targets = base.mark_point(size=400, opacity=0).encode(y="value:Q", tooltip=tooltip).add_params(hover)
    crosshair = base.mark_rule(color=MUTED, strokeWidth=1).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)))
    hover_dot = (
        base.mark_point(filled=True, size=80, stroke=SURFACE, strokeWidth=2, opacity=1)
        .encode(y="value:Q", color=color)
        .transform_filter(hover)
    )

    events = [{"t": context_length - 0.5, "label": "forecast start", "color": TEXT_SECONDARY}]
    if sample.shift_point is not None:
        events.append({"t": sample.shift_point - 0.5, "label": f"shift: {sample.shift_type}", "color": TEXT_PRIMARY})
    # stack labels so neighbouring events don't collide
    events_df = pl.DataFrame(events).with_columns(y=4 + 14 * pl.int_range(pl.len()))
    event_rules = (
        alt.Chart(events_df)
        .mark_rule(strokeWidth=1.5)
        .encode(x="t:Q", color=alt.Color("color:N", scale=None), tooltip=[alt.Tooltip("label:N", title="event")])
    )
    event_labels = (
        alt.Chart(events_df)
        .mark_text(align="left", baseline="top", dx=4, fontSize=11)
        .encode(x="t:Q", y=alt.Y("y:Q", scale=None), text="label:N", color=alt.Color("color:N", scale=None))
    )

    subtitle = []
    if sample.periods:
        subtitle.append("periods: " + ", ".join(str(p) for p in sample.periods))
    if sample.shift_point is not None:
        subtitle.append(f"{sample.shift_type} shift at t = {sample.shift_point}")
    if len(sample.spike_idx):
        subtitle.append(f"{len(sample.spike_idx)} spikes")

    # spikes below the line so the line runs into the dot instead of stopping at its ring
    main = alt.layer(event_rules, spikes, lines, crosshair, hover_dot, hit_targets, event_labels).properties(
        width=width, height=height
    )

    chart = main
    if show_covariate and sample.covariate is not None:
        chart = alt.vconcat(main, _covariate_panels(sample, x, width), spacing=12)

    return _style(
        chart.properties(
            title=alt.Title(
                title or "Synthetic sample",
                subtitle=" · ".join(subtitle) if subtitle else alt.Undefined,
                anchor="start",
            )
        )
    )


def _covariate_panels(sample: SyntheticSample, x: alt.X, width: int) -> alt.FacetChart:
    """One small panel per covariate feature (sin/cos per period), single neutral hue."""
    t = np.arange(sample.covariate.shape[0])
    names = [f"sin (p={p})" for p in sample.periods] + [f"cos (p={p})" for p in sample.periods]
    df = pl.concat(
        [pl.DataFrame({"t": t, "value": sample.covariate[:, i], "feature": name}) for i, name in enumerate(names)]
    )
    return (
        alt.Chart(df)
        .mark_line(strokeWidth=2, color=TEXT_SECONDARY)
        .encode(
            x=x,
            y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(values=[-1, 0, 1])),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("t:Q"),
                alt.Tooltip("value:Q", format=".3f"),
            ],
        )
        .properties(width=width, height=40)
        .facet(row=alt.Row("feature:N", title="covariate", sort=names, header=alt.Header(labelAngle=0, labelAlign="left")))
        .resolve_scale(x="shared")
    )


def _style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    """Recessive axes and grid, text in text colours."""
    return (
        chart.configure(background=SURFACE, font="system-ui, sans-serif")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=GRIDLINE,
            gridWidth=1,
            domainColor=GRIDLINE,
            tickColor=GRIDLINE,
            labelColor=TEXT_SECONDARY,
            titleColor=TEXT_SECONDARY,
            titleFontWeight="normal",
        )
        .configure_legend(labelColor=TEXT_PRIMARY)
        .configure_header(labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY, titleFontWeight="normal")
        .configure_title(color=TEXT_PRIMARY, subtitleColor=TEXT_SECONDARY, fontSize=14)
    )
