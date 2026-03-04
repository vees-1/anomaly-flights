# `Flight Anomaly Detection`

Domain: Aviation Safety · Unsupervised Anomaly Detection  
Dataset: Aircraft telemetry — November through December 2022 (Kaggle)  
Framework: TensorFlow · TensorFlow Probability · Pandas · Matplotlib

---

## `Overview`

An unsupervised anomaly detection system for aircraft telemetry using `Multivariate Gaussian Density Estimation`. The model learns the joint distribution of normal flight behavior and flags observations with low probability as anomalous — requiring no labeled data. This mirrors the anomaly detection framework from Andrew Ng's Machine Learning course and is well-suited to aviation telemetry where labeled anomaly data is scarce.

---

## `Methodology`

1. Clean: Drop rows missing `mph`, `alt`, `lat`, `long`
2. Engineer Features: Derive `vertical_rate` (altitude diff) and `trajectory_change` (abs lat/long diff)
3. Fit Model: Estimate `μ` and full covariance matrix `Σ` via TensorFlow Probability
4. Score & Threshold: Compute `p(x)` per observation; flag bottom 1% as anomalies (`p(x) < ε`)
5. Visualize: Probability histogram, geographic anomaly map, temporal anomaly plot

### `Pipeline`

```
Raw CSV (aircraft telemetry)
        │
        ▼
┌──────────────────────────┐
│  Data Cleaning           │  Drop rows missing mph, alt, lat, long
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Feature Engineering     │  vertical_rate = diff(alt)
│                          │  trajectory_change = |diff(lat)| + |diff(long)|
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Exploratory Analysis    │  Distribution plots for all 4 features
│                          │  Geographic scatter plot of flight paths
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Gaussian Model Fitting  │  μ = feature means, Σ = full covariance matrix
│  (TensorFlow Probability)│  via tfp.distributions.MultivariateNormalFullCovariance
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Anomaly Scoring         │  p(x) computed for all observations
│  & Thresholding          │  ε = 1st percentile of p(x)
└──────────────────────────┘
        │
        ▼
┌──────────────────────────┐
│  Visualization           │  Probability distribution histogram
│                          │  Geographic anomaly map (lat/long scatter)
│                          │  Temporal anomaly plot (p(x) over time)
└──────────────────────────┘
```

---

## `Model`

Fits a Multivariate Normal `N(x | μ, Σ)` over the 4-dimensional feature space. A full covariance matrix captures inter-feature correlations (e.g., speed–altitude relationship during climb/descent). The anomaly threshold `ε` is the 1st percentile of `p(x)`, flagging ~1% of observations with no need for ground truth labels.

---

## `Approach: Parametric Density Estimation`

*Rather than using a discriminative classifier (which would require labeled anomaly data), this project takes a generative, unsupervised approach:*

1. `Model` the joint distribution of flight features as a *Multivariate Normal distribution* `p(x) = N(μ, Σ)`
2. Compute the likelihood `p(x)` for every observation
3. Flag observations where `p(x) < ε` as anomalies, where `ε` is the 1st percentile of the probability distribution



