# Method: Temporal R*(t) Estimation

## Pipeline

```
Patient records (irregularly sampled)
            │
            ▼
Truncate at horizon t ∈ {6, 12, 24, 48, 72}h
            │
            ▼
Train encoder f_θ (GRU-D / BRITS / SAITS / iTransformer / TimesNet / SeFT)
            │
            ▼
Raw softmax probs p(y|x)
            │
            ▼
Isotonic calibration (5-fold CV) → η̂(x)
            │
            ▼
R*(t) = 1/2 − 1/2 · E[|2η̂(x) − 1|]   (Ishida et al., ICLR 2023)
```

## Why isotonic calibration?

Raw softmax outputs from deep networks are usually overconfident (Guo et al., 2017).
Three calibration methods were compared:

| Method | Assumption | Parametric? |
|--------|-----------|-------------|
| Platt scaling | Sigmoid shape | Yes (2 params) |
| Temperature scaling | Softmax shape | Yes (1 param) |
| **Isotonic regression** | **Monotonicity only** | **No** |

Ushio et al. (ICLR 2026) prove that isotonic regression gives a statistically
consistent R* estimator as long as the calibration function is monotone,
which is a much weaker assumption than Platt or temperature scaling.

## Why instance-free?

The estimator

    R*(t) = 1/2 − 1/2 · E_{x~p(x)}[|2η(x) − 1|]

only requires the soft labels η̂(x_i), not the input features x_i themselves.
This is useful for privacy-sensitive settings (e.g. hospitals cannot share
raw patient time series but can share model outputs).

## Why truncation?

In static classification, R* is a single number. For time series, the
available information at time t is a strict subset of the information at
time t' > t. Therefore R*(t) is monotonically non-increasing.

Truncation simulates the clinical scenario where a decision must be made
at time t without waiting for future observations.

## References

See [README.md](../README.md#citation) for full BibTeX entries.
