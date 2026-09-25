# Thesis Roadmap: Rebuilding Chronos-2, Toto 2.0, and TiRex-2

## Goal

The goal of this project is **not merely to run three time-series
foundation models**. The goal is to understand them deeply enough to:

1.  explain the mathematical and architectural choices,
2.  implement simplified versions from scratch in PyTorch,
3.  reproduce the important architectural details,
4.  train small versions yourself,
5.  compare your implementation with the official implementations,
6.  understand what changed from version 1 to version 2 for each model,
    and
7.  use the resulting knowledge in a thesis rather than treating the
    models as black boxes.

## Recommended order

> **1. Chronos-2 → 2. Toto 2.0 → 3. TiRex-2**

For a learning/reimplementation project, this order gives a useful
progression:

  -----------------------------------------------------------------------------------
  Model            Main architecture                    Relative Main reason to study
                                                  implementation 
                                                      difficulty 
  ---------------- ----------------------- --------------------- --------------------
  **Chronos-2**    T5-style encoder-only                   Lower Best entry point
                   Transformer                                   into modern TS
                                                                 foundation-model
                                                                 architecture

  **Toto 2.0**     Decoder-only                      Medium/High Introduces a
                   Transformer +                                 different
                   time/variate                                  Transformer
                   attention + u-μP                              organization and
                                                                 scaling methodology

  **TiRex-2**      Recurrent/xLSTM-based                    High Moves beyond
                   architecture                                  standard Transformer
                                                                 forecasting and
                                                                 introduces
                                                                 streaming-oriented
                                                                 recurrent
                                                                 computation
  -----------------------------------------------------------------------------------

This is a **learning-order recommendation, not a model-quality
ranking**.

------------------------------------------------------------------------

# 1. The overall learning strategy

Do not start by cloning an official repository and trying to understand
thousands of lines of code.

Use this loop for every model:

``` text
Paper
  ↓
Architecture diagram
  ↓
Mathematical formulation
  ↓
Minimal implementation from scratch
  ↓
Synthetic experiments
  ↓
Faithful implementation
  ↓
Official implementation
  ↓
Implementation comparison
  ↓
Reproduction experiment
```

The most important principle is:

> **If you cannot explain the shape and meaning of every tensor, you do
> not yet understand the model.**

For every major module, document:

-   input shape,
-   output shape,
-   trainable parameters,
-   mathematical operation,
-   computational complexity,
-   reason the authors introduced it,
-   what would happen if it were removed.

------------------------------------------------------------------------

# 2. Prerequisites

Before implementing the three models, make sure you are comfortable
with:

## Python

-   NumPy
-   pandas
-   matplotlib
-   dataclasses
-   typing
-   basic object-oriented Python
-   PyTorch

## PyTorch

You should be able to write without copying a tutorial:

-   `nn.Module`
-   `nn.Parameter`
-   `forward()`
-   `Dataset`
-   `DataLoader`
-   batching
-   masking
-   broadcasting
-   `einsum`
-   attention
-   residual connections
-   normalization
-   mixed precision
-   checkpointing
-   GPU training

## Mathematics

You should understand:

### Linear algebra

-   vectors
-   matrices
-   tensors
-   matrix multiplication
-   dot products
-   projections
-   eigenvalues/eigenvectors at a basic level

### Probability

-   probability distributions
-   expectation
-   variance
-   covariance
-   Gaussian distribution
-   Student-t distribution
-   mixture distributions
-   quantiles
-   likelihood
-   negative log likelihood

### Optimization

-   gradient descent
-   Adam/AdamW
-   learning rate schedules
-   weight decay
-   gradient clipping
-   initialization
-   overfitting

### Time-series concepts

-   trend
-   seasonality
-   autocorrelation
-   stationarity
-   non-stationarity
-   multivariate time series
-   covariates
-   forecasting horizon
-   probabilistic forecasting
-   point vs probabilistic forecasts

------------------------------------------------------------------------

# 3. Transformer and LSTM foundations to implement first

## LSTM

Create a simple LSTM Model as basis model.


## Transformer
Before Chronos-2, build a tiny Transformer yourself.

Do not use `nn.Transformer` for the first version.

Implement:

``` text
Linear projections
      ↓
Q, K, V
      ↓
Scaled dot-product attention
      ↓
Multi-head attention
      ↓
Residual connection
      ↓
Normalization
      ↓
Feed-forward network
      ↓
Residual connection
      ↓
Normalization
```

You should be able to derive:

\[ Q=XW_Q,`\qquad `{=tex}K=XW_K,`\qquad `{=tex}V=XW_V \]

and

\[ Attention(Q,K,V) =
softmax`\left`{=tex}(`\frac{QK^T}{\sqrt{d_k}}`{=tex}`\right`{=tex})V. \]

Then implement:

-   causal attention,
-   bidirectional/self attention,
-   masking,
-   RoPE,
-   a small Transformer encoder,
-   a small Transformer decoder.

This work will pay off twice: Chronos-2 uses an encoder-only T5-style
Transformer, while Toto 2.0 uses a decoder-only Transformer.

------------------------------------------------------------------------

# 4. Chronos-1 → Chronos-2

## Why start here?

Chronos-2 is the cleanest starting point for this thesis because it
combines familiar Transformer machinery with time-series-specific ideas.

The original Chronos converts scaled numerical values into discrete
tokens and trains a language-model-style architecture with
cross-entropy. During inference, it autoregressively samples tokens and
converts them back into numerical forecasts.

Chronos-2 changes the formulation substantially: it is an encoder-only
Transformer, works directly with continuous numerical patches, supports
univariate and multivariate forecasting plus covariates in one
architecture, and predicts multiple forecast steps directly using
quantile regression.

## Chronos-1 conceptual pipeline

``` text
time series
    ↓
scaling
    ↓
quantization / binning
    ↓
discrete tokens
    ↓
T5 / decoder Transformer
    ↓
autoregressive token prediction
    ↓
sample many trajectories
    ↓
map tokens back to numbers
    ↓
forecast distribution
```

The original Chronos models are based on T5-style architectures with a
reduced vocabulary of 4096 tokens. Training uses cross-entropy on token
sequences, and inference is autoregressive.

## Chronos-2 conceptual pipeline

``` text
time series
    ↓
robust normalization
    ↓
patching
    ↓
real-valued patch embeddings
    ↓
time + group attention
    ↓
T5-style encoder
    ↓
quantile head
    ↓
direct multi-step forecast
```

The published Chronos-2 model is a 120M-parameter encoder-only
Transformer. Its configuration uses a context length of 8192,
input/output patch size 16, 12 Transformer layers, 12 attention heads,
and up to 1024 forecast steps through output patches.

## Key Chronos-1 vs Chronos-2 differences

  ------------------------------------------------------------------------
  Component               Chronos-1               Chronos-2
  ----------------------- ----------------------- ------------------------
  Main Transformer        T5-style language model T5-style encoder-only
                                                  Transformer

  Representation          Discrete numerical      Continuous/real-valued
                          tokens                  patches

  Forecasting             Autoregressive          Direct multi-step

  Output                  Token probabilities     Quantiles

  Main loss               Cross-entropy           Quantile-based
                                                  forecasting objective

  Data support            Primarily univariate    Univariate +
                                                  multivariate +
                                                  covariates

  Cross-series learning   Not native              Group attention

  Position handling       T5-style positional     RoPE
                          mechanism               

  Context                 Shorter in original     Up to 8192 in published
                          model                   Chronos-2

  Forecast horizon        Up to 64 in original    Up to 1024 in published
                          Chronos                 Chronos-2
  ------------------------------------------------------------------------

Sources: [Chronos model
card](https://huggingface.co/amazon/chronos-t5-base), [Chronos-2 model
card](https://huggingface.co/amazon/chronos-2), [Chronos-2 technical
report](https://arxiv.org/abs/2510.15821).

## What you should implement

### Stage C1 --- toy forecasting Transformer

Input:

``` text
[B, T]
```

Output:

``` text
[B, H]
```

Implement:

-   normalization,
-   simple patching,
-   patch embedding,
-   Transformer encoder,
-   linear prediction head.

### Stage C2 --- probabilistic output

Change:

``` text
[B, H]
```

to:

``` text
[B, Q, H]
```

where `Q` is the number of quantiles.

Implement quantile loss:

\[ L_q(y,`\hat `{=tex}y) =
`\max`{=tex}(q(y-`\hat `{=tex}y),(q-1)(y-`\hat `{=tex}y)). \]

### Stage C3 --- multivariate data

Move from:

``` text
[B, T]
```

to something like:

``` text
[B, V, T]
```

where:

-   `B` = batch
-   `V` = number of variates
-   `T` = time

Then understand exactly where time attention and group attention
operate.

### Stage C4 --- faithful Chronos-2

Implement:

-   robust scaling,
-   arcsinh transformation,
-   time encoding,
-   mask/meta features,
-   patch embedding,
-   REG token,
-   RoPE,
-   time attention,
-   group attention,
-   quantile decoder,
-   direct multi-horizon forecasting.

Chronos-2's published configuration uses 16-step patches and a
64-output-patch limit. Its input representation also includes time/mask
information and an optional REG token.

### Stage C5 --- reproduction

Compare your implementation against:

``` text
amazon/chronos-2
```

Compare:

-   tensor shapes,
-   parameter count,
-   output distributions,
-   deterministic inference,
-   forecast accuracy,
-   runtime,
-   memory usage.

------------------------------------------------------------------------

# 5. Toto 1.0 → Toto 2.0

Toto is especially useful after Chronos because it remains
Transformer-based but makes very different design decisions.

## Toto 1.0

Toto 1.0 is a decoder-only Transformer designed around observability
time series.

Its important components include:

-   proportional factorized space-time attention,
-   causal patch scaling,
-   patch-based input representation,
-   Student-t mixture output distribution,
-   autoregressive forecasting,
-   heavy use of observability data during pretraining.

The original Toto was trained on roughly one trillion time-series data
points, with a large fraction coming from anonymized Datadog
observability metrics.

The open Toto 1.0 checkpoint was about 151M parameters.

## Toto 1.0 conceptual pipeline

``` text
multivariate time series
        ↓
causal patch scaling
        ↓
patch embedding
        ↓
decoder-only Transformer
        ↓
factorized space/time attention
        ↓
Student-t mixture head
        ↓
sample forecast patch
        ↓
feed sample back
        ↓
repeat autoregressively
```

The Student-t mixture head models multiple Student-t components and
samples from the resulting distribution during forecasting.

------------------------------------------------------------------------

# 6. Toto 2.0

Toto 2.0 changes the emphasis toward scalable Transformer training.

The current Toto 2.0 family contains models from approximately 4M to
2.5B parameters:

``` text
4M
22M
313M
1B
2.5B
```

It uses:

-   decoder-only Transformer,
-   alternating time/variate attention,
-   quantile forecasting,
-   u-μP scaling,
-   high-dimensional multivariate forecasting.

Unlike Toto 1.0, Toto 2.0's current release does not yet provide the
same fine-tuning/exogenous-variable workflow; the official repository
says those capabilities are planned for a future 2.0 release.

## Toto 1.0 vs Toto 2.0

  -------------------------------------------------------------------------
  Component               Toto 1.0                  Toto 2.0
  ----------------------- ------------------------- -----------------------
  Transformer             Decoder-only              Decoder-only

  Main attention idea     Proportional factorized   Alternating
                          space-time attention      time/variate attention

  Output                  Student-t mixture         Quantile head

  Forecasting             Sampling/autoregressive   Quantile-based
                                                    forecasting

  Scaling                 Standard training         u-μP scaling
                          approach                  

  Model sizes             \~151M open base          4M → 2.5B family

  Main emphasis           Observability +           Scaling +
                          probabilistic modeling    high-dimensional
                                                    multivariate
                                                    forecasting

  Fine-tuning             Supported                 Not in current 2.0
                                                    release

  Exogenous variables     Supported in 1.0 workflow Not in current 2.0
                                                    release
  -------------------------------------------------------------------------

The official Toto repository should be treated as the primary source for
current feature availability.

Sources: [Toto repository](https://github.com/DataDog/toto), [Toto 1.0
paper](https://arxiv.org/abs/2407.07874), [Toto 1.0 model
card](https://huggingface.co/Datadog/Toto-Open-Base-1.0), [Toto 2.0
report](https://arxiv.org/abs/2605.20119).

## What you should implement

### Stage T1 --- decoder Transformer

Start with:

``` text
patches
  ↓
decoder blocks
  ↓
causal attention
  ↓
forecast head
```

Understand why decoder-only architecture can forecast arbitrary
horizons.

### Stage T2 --- time/variate structure

Represent the data explicitly:

``` text
[B, V, T]
```

Then study:

``` text
time attention:
within each variate

variate attention:
across variables
```

Draw the tensor transformations by hand.

### Stage T3 --- probabilistic forecasting

First implement a simple Gaussian output.

Then implement the Student-t distribution.

Finally implement the Student-t mixture.

You should understand:

\[ p(y) = `\sum`{=tex}\_{k=1}\^{K} `\pi`{=tex}\_k
,StudentT(y;`\mu`{=tex}\_k,`\sigma`{=tex}\_k,`\nu`{=tex}\_k). \]

### Stage T4 --- Toto 2.0 quantile head

Replace the mixture distribution with the 2.0 quantile output mechanism.

Study why the model can use quantiles instead of explicitly
parameterizing a mixture distribution.

### Stage T5 --- u-μP

Only after the model works should you study:

-   μP,
-   unit scaling,
-   width scaling,
-   parameterization,
-   initialization,
-   learning-rate scaling,
-   transfer across model widths.

This should be a separate thesis sub-project rather than something you
mix into your first implementation.

### Stage T6 --- scale experiments

Train:

``` text
Toto-mini
   ↓
larger Toto
   ↓
same training recipe
   ↓
compare scaling behavior
```

This is where Toto 2.0 becomes particularly interesting scientifically.

------------------------------------------------------------------------

# 7. TiRex 1 → TiRex-2

TiRex is the biggest architectural jump.

## TiRex 1

TiRex is a roughly 35M-parameter univariate zero-shot forecasting model
based on xLSTM. It provides point and quantile forecasts.

Conceptually:

``` text
univariate history
      ↓
patch representation
      ↓
xLSTM-based recurrent model
      ↓
forecast representation
      ↓
quantile forecasts
```

Source: [TiRex repository](https://github.com/NX-AI/tirex).

------------------------------------------------------------------------

# 8. TiRex-2

TiRex-2 generalizes TiRex in two major directions:

### 1. Multivariate forecasting

One checkpoint can forecast multiple target variates jointly.

### 2. Streaming-oriented recurrent architecture

The architecture extends the recurrent/xLSTM design so that it is
suitable for streaming scenarios.

It also supports:

-   past covariates,
-   future-known covariates,
-   univariate forecasting,
-   multivariate forecasting,
-   zero-shot inference.

The official documentation explicitly describes TiRex-2 as a
generalization of the original univariate TiRex along these axes.

## TiRex 1 vs TiRex-2

  -----------------------------------------------------------------------
  Component               TiRex 1                 TiRex-2
  ----------------------- ----------------------- -----------------------
  Core family             xLSTM-based recurrent   Extended
                          model                   recurrent/xLSTM
                                                  architecture

  Forecasting             Univariate              Univariate +
                                                  multivariate

  Covariates              More limited            Past + future-known
                                                  covariates

  Zero-shot               Yes                     Yes

  Quantile forecasts      Yes                     Yes

  Streaming focus         Original model          Explicit
                                                  streaming-oriented
                                                  design

  Main research direction Long/short-horizon      Multivariate +
                          forecasting             streaming
  -----------------------------------------------------------------------

The official TiRex-2 repository currently describes the open model as
the multivariate model, while its Pro version adds further
streaming/hardware-optimized inference and other capabilities. Do not
accidentally assume every Pro feature is part of the open model.

Sources: [TiRex repository](https://github.com/NX-AI/tirex), [TiRex-2
documentation](https://nx-ai.github.io/tirex-2/), [TiRex-2
repository](https://github.com/NX-AI/tirex-2), [TiRex-2
paper](https://arxiv.org/abs/2607.01204).

------------------------------------------------------------------------

# 9. What you need to learn for TiRex-2

Before touching the full model, implement an xLSTM yourself.

Study:

``` text
LSTM
 ↓
modern LSTM variants
 ↓
sLSTM
 ↓
mLSTM
 ↓
xLSTM
 ↓
TiRex
 ↓
TiRex-2
```

You should understand the recurrent state mathematically.

Then investigate:

-   how state is updated,
-   what information persists,
-   what is computed at every timestep,
-   what can be cached,
-   why recurrent state can help streaming,
-   how the model handles multiple variates,
-   how time and variate processing interact.

This is the point where the project becomes substantially different from
ordinary Transformer implementation.

------------------------------------------------------------------------

# 10. The three architectures side-by-side

A useful thesis diagram is:

``` text
                     TIME SERIES FOUNDATION MODELS

                               Input
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
         Chronos-2           Toto 2.0          TiRex-2
              │                  │                  │
              ▼                  ▼                  ▼
       Continuous patches   Continuous patches   Recurrent/
              │                  │               xLSTM state
              ▼                  ▼                  │
       Encoder-only          Decoder-only           ▼
       Transformer           Transformer       recurrent blocks
              │                  │                  │
              ▼                  ▼                  ▼
       Time attention       Time attention       temporal
              │              + variate           processing
              ▼               attention              │
       Group attention            │                 + variate
              │                  │                  processing
              ▼                  ▼                  │
        Quantile head       Quantile head           ▼
              │                  │             Quantile head
              ▼                  ▼                  │
          Forecast            Forecast          Forecast
```

The important thesis question is not:

> "Which model is best?"

Instead ask:

> **"What architectural assumptions does each model make about temporal
> structure, cross-series dependencies, probabilistic forecasting, and
> computational efficiency?"**

That question gives you a much stronger research framework.

------------------------------------------------------------------------

# 11. A unified tensor notation

Use consistent notation throughout your thesis.

Let:

-   \(B\) = batch size
-   \(V\) = number of variates
-   \(T\) = context length
-   \(H\) = forecast horizon
-   \(P\) = patch size
-   \(D\) = model dimension
-   \(Q\) = number of quantiles

Input:

\[ X `\in `{=tex}`\mathbb{R}`{=tex}\^{B`\times `{=tex}V`\times `{=tex}T}
\]

After patching:

\[ X_p `\in`{=tex}
`\mathbb{R}`{=tex}\^{B`\times `{=tex}V`\times `{=tex}N_p`\times `{=tex}P}
\]

where approximately:

\[ N_p = `\frac{T}{P}`{=tex}. \]

After embedding:

\[ E `\in`{=tex}
`\mathbb{R}`{=tex}\^{B`\times `{=tex}V`\times `{=tex}N_p`\times `{=tex}D}.
\]

Then the architectures differ mainly in **how they process these
dimensions**.

This notation will make your comparison much easier.

------------------------------------------------------------------------

# 12. Experiments to run for every model

For every model, run the same conceptual experiments.

## Experiment A --- sine wave

``` text
y(t) = sin(t)
```

Purpose:

-   verify basic forecasting,
-   understand horizon behavior,
-   inspect learned representations.

## Experiment B --- trend + seasonality

``` text
y(t) = trend(t) + seasonality(t) + noise
```

Purpose:

-   distinguish trend from periodic structure.

## Experiment C --- noisy series

Increase:

\[ `\sigma`{=tex}\_{`\epsilon`{=tex}} \]

and study prediction intervals.

## Experiment D --- correlated multivariate series

Generate:

``` text
series A
series B = 0.7 A + noise
series C = lag(A) + noise
```

Purpose:

-   investigate cross-series learning.

## Experiment E --- missing observations

Randomly remove values.

Purpose:

-   understand masks and missing-value handling.

## Experiment F --- covariates

Add:

``` text
day_of_week
holiday
promotion
temperature
```

Purpose:

-   study conditioning.

## Experiment G --- long context

Compare:

``` text
context = 128
context = 512
context = 2048
context = 8192
```

Purpose:

-   understand context scaling.

## Experiment H --- forecast horizon

Compare:

``` text
H = 16
H = 32
H = 64
H = 128
H = 512
```

Purpose:

-   investigate short vs long horizon behavior.

------------------------------------------------------------------------

# 13. Reproduction checklist

For each model, maintain a document like this:

``` text
[ ] Read original paper
[ ] Read version-2 paper
[ ] Read official repository
[ ] Draw architecture
[ ] Write tensor shapes
[ ] Implement preprocessing
[ ] Implement patching
[ ] Implement positional encoding
[ ] Implement attention/recurrent mechanism
[ ] Implement forecasting head
[ ] Implement loss
[ ] Train toy model
[ ] Overfit tiny dataset
[ ] Train synthetic dataset
[ ] Compare against official implementation
[ ] Reproduce selected benchmark
[ ] Document discrepancies
```

The **"overfit tiny dataset"** step is particularly important.

If your model cannot overfit a tiny dataset, do not proceed to
large-scale experiments.

------------------------------------------------------------------------

# 14. Suggested repository structure

``` text
thesis-timeseries-foundation-models/
│
├── README.md
│
├── docs/
│   ├── roadmap.md
│   ├── chronos2_notes.md
│   ├── toto2_notes.md
│   ├── tirex2_notes.md
│   └── experiments.md
│
├── foundations/
│   ├── attention.py
│   ├── rope.py
│   ├── transformer_encoder.py
│   ├── transformer_decoder.py
│   ├── lstm.py
│   └── xlstm.py
│
├── data/
│   ├── synthetic.py
│   ├── preprocessing.py
│   └── datasets.py
│
├── chronos2/
│   ├── normalization.py
│   ├── patching.py
│   ├── embeddings.py
│   ├── attention.py
│   ├── model.py
│   ├── quantile_head.py
│   ├── loss.py
│   └── train.py
│
├── toto2/
│   ├── scaling.py
│   ├── patching.py
│   ├── attention.py
│   ├── model.py
│   ├── quantile_head.py
│   ├── loss.py
│   └── train.py
│
├── tirex2/
│   ├── xlstm.py
│   ├── recurrent_blocks.py
│   ├── variate_mixer.py
│   ├── model.py
│   ├── quantile_head.py
│   └── train.py
│
├── experiments/
│   ├── sine/
│   ├── multivariate/
│   ├── missing_values/
│   ├── covariates/
│   └── long_context/
│
└── notebooks/
    ├── 01_attention.ipynb
    ├── 02_patching.ipynb
    ├── 03_chronos2.ipynb
    ├── 04_toto2.ipynb
    └── 05_tirex2.ipynb
```

------------------------------------------------------------------------

# 15. A staged thesis timeline

A possible 16-week learning plan:

## Weeks 1--2 --- Foundations

Study:

-   Transformer attention
-   positional encoding
-   RoPE
-   normalization
-   patching
-   probabilistic forecasting
-   quantile regression

Implement:

-   attention from scratch,
-   Transformer encoder,
-   Transformer decoder,
-   quantile loss.

## Weeks 3--5 --- Chronos-2

Week 3:

-   paper,
-   preprocessing,
-   scaling,
-   patching.

Week 4:

-   time attention,
-   group attention,
-   embeddings,
-   quantile head.

Week 5:

-   training,
-   evaluation,
-   official implementation comparison.

Deliverable:

> A small but faithful Chronos-2 implementation.

## Weeks 6--9 --- Toto 2.0

Week 6:

-   understand Toto 1.0,
-   Student-t distribution,
-   mixture distributions,
-   factorized attention.

Week 7:

-   Toto 2.0 architecture,
-   time/variate attention.

Week 8:

-   quantile forecasting,
-   decoder-only design.

Week 9:

-   u-μP,
-   scaling experiment.

Deliverable:

> A small Toto 2.0 implementation plus a documented 1.0 → 2.0
> comparison.

## Weeks 10--13 --- TiRex-2

Week 10:

-   LSTM,
-   sLSTM,
-   mLSTM,
-   xLSTM.

Week 11:

-   TiRex architecture.

Week 12:

-   TiRex-2 multivariate architecture,
-   covariates.

Week 13:

-   recurrent state,
-   streaming behavior,
-   reproduction experiments.

Deliverable:

> A small TiRex-2 implementation plus a documented TiRex → TiRex-2
> comparison.

## Weeks 14--16 --- Comparative study

Create one common evaluation framework.

Compare:

-   architecture,
-   parameter count,
-   context length,
-   horizon,
-   inference speed,
-   memory,
-   univariate forecasting,
-   multivariate forecasting,
-   covariate handling,
-   uncertainty,
-   long-context behavior,
-   missing values,
-   streaming suitability.

Do **not** reduce the thesis to a single "winner" score.

Instead, report the empirical measurements and discuss the architectural
trade-offs.

------------------------------------------------------------------------

# 16. What to read first

## First: Chronos

1.  **Chronos: Learning the Language of Time Series**
2.  Chronos GitHub/model card
3.  T5 architecture
4.  **Chronos-2: From Univariate to Universal Forecasting**
5.  Chronos-2 official implementation

The conceptual progression is:

``` text
language modeling
      ↓
Chronos-1
      ↓
continuous time-series representation
      ↓
multivariate modeling
      ↓
Chronos-2
```

## Second: Toto

1.  Toto 1.0 paper
2.  Toto 1.0 implementation
3.  Student-t mixtures
4.  factorized space-time attention
5.  Toto 2.0 technical report
6.  u-μP / μP literature
7.  Toto 2.0 implementation

The conceptual progression is:

``` text
decoder Transformer
      ↓
multivariate attention
      ↓
probabilistic mixture forecasting
      ↓
large-scale training
      ↓
u-μP scaling
      ↓
Toto 2.0
```

## Third: TiRex

1.  LSTM
2.  xLSTM paper
3.  TiRex paper
4.  TiRex implementation
5.  TiRex-2 paper
6.  TiRex-2 implementation

The conceptual progression is:

``` text
LSTM
 ↓
xLSTM
 ↓
TiRex
 ↓
multivariate recurrent forecasting
 ↓
TiRex-2
 ↓
streaming-oriented forecasting
```

------------------------------------------------------------------------

# 17. The most important comparison questions for your thesis

For each model, answer these questions.

## Representation

> How is a numerical time series represented internally?

Chronos-2, Toto 2.0, and TiRex-2 make substantially different choices
here.

## Temporal modeling

> How does the model represent temporal dependencies?

Compare:

-   Transformer attention,
-   causal attention,
-   time attention,
-   recurrent state.

## Cross-series modeling

> How does information move between different variates?

This is especially important for Chronos-2 and Toto 2.0, and becomes a
major extension in TiRex-2.

## Probabilistic forecasting

> How does the model represent uncertainty?

Compare:

``` text
Chronos-1
categorical token distribution

Chronos-2
quantile forecasts

Toto-1
Student-t mixture

Toto-2
quantile forecasts

TiRex/TiRex-2
quantile forecasts
```

## Computational complexity

Ask:

> What happens when context length increases?

and:

> What happens when the number of variates increases?

This can become a very useful thesis analysis.

## Streaming

Especially for TiRex-2:

> Can the model update its internal state when a new observation arrives
> without recomputing the entire history?

That is fundamentally different from ordinary full-context Transformer
inference.

------------------------------------------------------------------------

# 18. A good final thesis experiment

A strong final experiment would use **one common synthetic testbed**
plus several real datasets.

For example:

``` text
                  ┌──────────────────────┐
                  │ Common test datasets │
                  └──────────┬───────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
        Chronos-2         Toto 2.0         TiRex-2
           │                 │                 │
           └─────────────────┼─────────────────┘
                             ▼
                    Common evaluation
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
       Accuracy          Uncertainty        Efficiency
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                    Architectural analysis
```

Measure at least:

-   MAE
-   MSE
-   MASE
-   quantile loss / WQL
-   calibration/coverage where appropriate
-   inference time
-   peak GPU memory
-   parameter count
-   context length
-   forecast horizon

Then explain the measurements through the architecture.

That last part is important.

Instead of:

> "Model A got 0.23 and Model B got 0.25."

Aim for:

> "Under long-context conditions, the models behaved differently. This
> can be investigated in relation to their attention/recurrent
> mechanisms and the way they represent temporal information."

That turns the work from a benchmark exercise into a
**model-understanding thesis**.

------------------------------------------------------------------------

# 19. Final learning path

The entire project can be summarized as:

``` text
                    TIME-SERIES FOUNDATION MODELS
                              │
                              ▼
                       Transformer basics
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
        Encoder Transformer             Decoder Transformer
              │                               │
              ▼                               ▼
          Chronos-1                         Toto-1
              │                               │
              ▼                               ▼
          Chronos-2                         Toto-2
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
                    xLSTM / recurrent models
                              │
                              ▼
                            TiRex
                              │
                              ▼
                           TiRex-2
```

### Recommended order

**Start with Chronos-2.**

Then:

**Toto 2.0.**

Then:

**TiRex-2.**

But while studying each version 2 model, keep version 1 beside it. The
v1 → v2 comparison is valuable because it lets you ask:

> **What limitation of the previous architecture was the new
> architecture trying to address?**

That is a much more interesting thesis question than simply documenting
three implementations.

------------------------------------------------------------------------

# 20. Primary sources

### Chronos

-   Chronos paper: https://arxiv.org/abs/2403.07815
-   Chronos model: https://huggingface.co/amazon/chronos-t5-base
-   Chronos-2 paper: https://arxiv.org/abs/2510.15821
-   Chronos-2 model: https://huggingface.co/amazon/chronos-2

### Toto

-   Toto 1.0 paper: https://arxiv.org/abs/2407.07874
-   Toto repository: https://github.com/DataDog/toto
-   Toto 1.0 model: https://huggingface.co/Datadog/Toto-Open-Base-1.0
-   Toto 2.0 paper: https://arxiv.org/abs/2605.20119

### TiRex

-   TiRex repository: https://github.com/NX-AI/tirex
-   TiRex-2 paper: https://arxiv.org/abs/2607.01204
-   TiRex-2 repository: https://github.com/NX-AI/tirex-2
-   TiRex-2 documentation: https://nx-ai.github.io/tirex-2/

------------------------------------------------------------------------

# One final recommendation

Do **not** make your first goal:

> "I need to reproduce the published benchmark."

Make your first goal:

> **"I can build a tiny version of the model, train it, inspect every
> tensor, deliberately break individual components, and explain why the
> forecast changes."**

Once you can do that, reproducing the larger models becomes a much more
manageable engineering problem.

For this particular thesis, the most valuable progression is:

**Chronos-2 → Toto 2.0 → TiRex-2**, while explicitly studying
**Chronos-1 → Chronos-2, Toto-1 → Toto-2, and TiRex → TiRex-2** as
architectural evolution rather than treating the version-2 models in
isolation.
