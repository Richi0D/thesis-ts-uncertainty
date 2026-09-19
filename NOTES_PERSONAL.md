# Personal Notes

## Starting Tasks

- Literature research: read and organize relevant papers.
- Study different time series models:
  - TiRex-2,
  - Chronos-2,
  - Toto 2.0,
  -  compare them with their predecessors, and build the theoretical background needed to understand the models.
- Inspect the `fev-bench` implementations of these models.
- Study TCM, TSI, autoregressive generation, and univariate generators.
- Implement and investigate the Dyst benchmark.



## Starting Notes 

### Background
In my practical work, I fine-tuned and modified the TiRex time series foundation model (built on xLSTM) for energy forecasting. One clear finding was that switching from the model's original multi-patch generation to autoregressive generation improves forecasts for long horizons, but causes quantile collapse: the model's uncertainty estimates degrade sharply after the first few autoregressive steps. A closely related paper (Moirai 2.0, late 2025) addresses this same problem, but only for parallel, transformer-based architectures — the fix does not transfer to recurrent, stateful architectures like xLSTM.
Separately, the energy system itself is changing (more PV, EVs, heat pumps, shifting consumption behavior). Recent 2026 papers benchmark how well foundation models handle distribution shift in energy data (e.g. training on pre-COVID years and testing on later years), but these papers evaluate existing models as-is and do not test or propose calibration fixes, nor do they isolate specific causes of shift through controlled experiments.

### Research Gap
No existing work combines these two lines: nobody has tested whether a foundation model's uncertainty calibration (and any proposed fix for quantile collapse) remains reliable once the underlying energy data distribution shifts. This is an important gap: a forecast that looks confident but is silently miscalibrated is arguably more dangerous for grid operations than one that is simply less accurate.

### Proposed Thesis

**Title (working):** Robust Uncertainty Quantification for Recurrent Time Series Foundation Models under Distribution Shift: A Study on Energy Forecasting with TiRex

**Research questions:**
1. How does the calibration quality (coverage, sharpness, quantile loss) of TiRex's autoregressive generation degrade as distribution shift increases, compared to in-distribution performance?
2. Does the previously observed quantile collapse problem worsen disproportionately under shift, and does this interact with forecast horizon?
3. Can a distribution-preserving decoding method — designed to propagate more than just the median through the model's recurrent state during autoregressive generation — remain robust under shift, compared to a conformal-prediction baseline?

**Method:**
- Reuse the practical work's TiRex pipeline and energy datasets (household, OPSD, Open Energy Hub, FETS).
- Evaluate shift along two tracks: (a) realistic — train on earlier years, test on later years; (b) controlled/synthetic — systematically inject isolated changes (e.g. simulated PV increase, simulated EV charging load) at increasing severity, to obtain a degradation curve rather than a single before/after number.
- Compare decoding strategies: original NaN-fill, naive median-feedback (both from the practical work), a new distribution-preserving decoding method, and conformal prediction as a cheap baseline.
- Metrics: SMAPE for accuracy; coverage, CRPS, and quantile loss for calibration; degradation slope across shift severity as the main robustness metric.

### Why This Is a Suitable Master's Thesis
- Builds directly on the practical work's codebase, model, and datasets — no new compute infrastructure required.
- Clear, citable research gap: the closest related papers (Moirai 2.0 on quantile collapse; 2026 TSFM energy benchmarks on distribution shift) each cover only one half of the problem.
- Scoped novelty: one new decoding method, compared against three well-established baselines, on a well-defined evaluation protocol.
- Produces a useful result either way: if the new decoding method wins, it is a genuine methodological contribution; if it does not outperform conformal prediction, that is still a valid and informative finding for the field.