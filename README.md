---
author:
- Amin Fazl Kazemi
date: 2026-07-27
title: |
  ClimateProcessingEngine\
  A Comprehensive Framework for Climate Data Processing
---

# Introduction

## Overview

**ClimateProcessingEngine** is a high-performance, modular framework for
fitting probability distributions to climatological time series data. It
supports station-based and gridded datasets, handles large-scale
processing with Zarr storage, and provides advanced features for
uncertainty quantification, quality control, and reproducibility.

## Core Philosophy

The engine is built on three core principles:

1.  **Reproducibility** -- Every result can be reproduced exactly with
    the same inputs and configuration.

2.  **Scalability** -- Process millions of data points efficiently with
    minimal memory footprint.

3.  **Extensibility** -- Add new distributions, quality controls, or
    statistical methods without modifying core code.

## Scientific Background

Climate data analysis often requires fitting probability distributions
to temperature or precipitation time series. This allows researchers to:

-   Estimate return periods for extreme events (e.g., 100-year floods)

-   Assess climate variability and change

-   Detect shifts in distribution parameters

-   Compare different statistical models

-   Generate synthetic climate data for impact studies

The **ClimateProcessingEngine** automates this workflow, providing a
robust, validated, and efficient implementation of the state-of-the-art
methods.

# Design Philosophy

## Design Principles

The framework is built on four core principles:

  **Principle**                           **Description**
  --------------------------------------- -------------------------------------------------------------------------------------------------------------------------------------
  **Principle**                           **Description**
  **Plugin-first architecture**           All statistical distributions and quality controls are implemented as plugins, enabling easy extension without modifying core code.
  **Reproducible scientific workflows**   Every run records configuration, dependencies, and input hashes to ensure results can be exactly reproduced.
  **Separation of concerns**              I/O, numerical computation, orchestration, and monitoring are clearly separated into independent modules.
  **Configuration over hard-coding**      All runtime parameters are defined in a single `config.yaml` file, making experimentation and deployment straightforward.

## Design Goals

The project is designed with the following primary goals:

  **Goal**                         **Description**
  -------------------------------- -----------------------------------------------------------------------------------------------------------------------
  **Goal**                         **Description**
  **Scientific correctness**       All statistical methods are validated against reference implementations and literature.
  **Reproducibility**              Every run captures configuration, dependencies, and input metadata for exact replication.
  **Extensibility**                Plugin architecture allows adding new distributions, quality controls, and data adapters without modifying core code.
  **High performance**             Block-based processing, vectorized operations, and optional parallel execution handle large datasets efficiently.
  **Modular architecture**         Clear separation of concerns (I/O, computation, orchestration, monitoring) simplifies testing and maintenance.
  **Production-ready workflows**   Designed for batch processing of real-world climate data with checkpoint recovery and comprehensive logging.

## Non-goals

This project explicitly does **not** aim to:

-   **Be a full GIS platform** -- while it handles geospatial data, it
    does not provide map visualization, spatial analysis, or advanced
    geospatial operations.

-   **Provide weather forecasting** -- it performs statistical fitting
    and climatology generation, not real-time or short-term weather
    prediction.

-   **Be a machine learning framework** -- it does not include deep
    learning models, neural networks, or general-purpose ML pipelines.

-   **Serve as a visualization library** -- while it includes basic
    plotting utilities, it is not a replacement for tools like
    Matplotlib, Cartopy, or Seaborn.

-   **Replace specialized extreme value packages** -- for advanced
    non-stationary extreme value analysis, refer to dedicated tools like
    `extRemes` or `climextRemes`.

These boundaries ensure the project remains focused, maintainable, and
effective for its core mission.

## Design Trade-offs

The engine makes several deliberate trade-offs to achieve its goals:

  **Trade-off**                       **Decision**                                           **Rationale**
  ----------------------------------- ------------------------------------------------------ -----------------------------------------------------------------------------------------------
  **Trade-off**                       **Decision**                                           **Rationale**
  **Accuracy vs. Speed**              Prioritizes scientific accuracy over raw speed         Climate research requires reliable statistical estimates; moderate performance is acceptable.
  **Memory vs. I/O**                  Uses block processing with moderate memory footprint   Ensures datasets larger than RAM can be processed; I/O overhead is managed with caching.
  **Extensibility vs. Simplicity**    Adopts plugin architecture                             Adds complexity but enables easy extension without changing core code.
  **Python vs. Compiled Languages**   Written in Python with Numba acceleration              Balances development velocity with performance for numerical workloads.
  **Zarr vs. NetCDF**                 Zarr as primary storage; NetCDF export supported       Zarr provides better scalability and cloud compatibility; NetCDF ensures interoperability.

# Key Features

  **Feature**                       **Description**
  --------------------------------- -------------------------------------------------------------------------------------------------------------
  **Feature**                       **Description**
  **Plugin Architecture**           Add new distributions without modifying core code -- just drop a `.py` file in `plugins/distributions/`.
  **Multi-Variable Support**        Process `tmin`, `tmean`, and `tmax` simultaneously or individually.
  **Normal & Extreme Modes**        Switch between standard 5-day windows and extreme-value (max/min) extraction with automatic GEV activation.
  **Quality Flag System**           Automatically flags low-sample, non-convergence, high-AICC, outlier, and NaN/Inf fits.
  **Bootstrap Uncertainty**         Estimate parameter confidence intervals (95% CI) for every fit.
  **Parallel Processing**           Block-based processing with optional multiprocessing, Dask, or Ray backends.
  **Scalable Storage**              Outputs to Zarr (v3) with Blosc/Zstd compression; NetCDF export also supported.
  **Checkpoint & Recovery**         Resistant to power failures -- resumes exactly where it stopped.
  **Sample Data Included**          10 synthetic stations for quick testing and tutorials.
  **CI/CD Ready**                   GitHub Actions workflows for testing, coverage, and notebook execution.
  **Comprehensive Documentation**   Sphinx-based docs + Jupyter Notebook tutorials.
  **Full Provenance Tracking**      Tracks git commit, Python version, config, and execution time.
  **Out-of-Core Processing**        Handles datasets larger than available RAM.
  **Spatial Consistency Checks**    Compares neighboring stations to identify anomalous fits.
  **Temporal Consistency Checks**   Detects unrealistic day-to-day changes in best distribution.
  **Intelligent Disk Caching**     Multi-block-size detection (1000, 2000, 5000) to reuse existing cache files and avoid redundant I/O.
  **Auto-Resume Checkpoints**      Automatically detects the last valid processing point from the output Zarr and resumes after interruptions.
  **Selective Cache Builder**      Use `build_cache_only.py` to pre-build cache files for unprocessed blocks without running full analysis.

# Target Audience

  **Audience**                    **Application**
  ------------------------------- -------------------------------------------------------------
  **Audience**                    **Application**
  **Climate Scientists**          Research on climate variability, trends, and extremes
  **Meteorologists**              Operational weather and climate monitoring
  **Hydrologists**                Water resource assessment and flood risk analysis
  **Environmental Researchers**   Ecosystem and environmental impact studies
  **Agricultural Scientists**     Crop yield modeling and climate adaptation
  **Geospatial Data Engineers**   Large-scale climate data processing pipelines
  **Graduate Students**           Academic research and thesis projects
  **Data Scientists**             Machine learning and statistical modeling with climate data

# Installation

## System Requirements

  **Component**            **Minimum**                       **Recommended**
  ------------------------ --------------------------------- -----------------
  **Component**            **Minimum**                       **Recommended**
  **Operating System**     Windows 10, Linux, macOS 10.15+   Linux/macOS
  **Python Version**       3.8                               3.11+
  **RAM**                  16 GB                             32 GB+
  **Storage**              50 GB free                        100 GB+ SSD
  **CPU**                  4 cores                           8+ cores
  **Python Environment**   Conda or venv                     Conda

## Installation From Source (recommended)

``` {.bash caption="Installation Steps" language="bash"}
# Clone The Repository
git clone Source: https://github.com/AminFazlKazemi/ClimateProcessingEngine.git
cd ClimateProcessingEngine

# Create Virtual Environment (optional But Recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
## Or Venv\scripts\activate # Windows

# Install The Package
pip install -e .
```

## Installation With Conda (alternative)

``` {.bash caption="Conda Installation" language="bash"}
conda create -n climatology python=3.11
conda activate climatology
conda install numpy scipy xarray zarr numba pandas matplotlib seaborn pyyaml
pip install -e .
```

# Quick Start

## Using The Sample Dataset

``` {.python caption="Loading Sample Data" language="python"}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load Sample Station Data
sample_dir = "sample_data"
station_data = pd.read_csv(f"{sample_dir}/station_001.csv").values

# Display Data Shape
print(f"Data shape: {station_data.shape}")  # (10950, 3) = 30 years * 365 days, 3 variables

# Plot Temperature For The First 365 Days
plt.figure(figsize=(12, 6))
plt.plot(station_data\left[:365, 1\right], label='tmean', linewidth=2)
plt.plot(station_data\left[:365, 0\right], label='tmin', alpha=0.7)
plt.plot(station_data\left[:365, 2\right], label='tmax', alpha=0.7)
plt.xlabel('Day of Year')
plt.ylabel('Temperature (°C)')
plt.title('Station 001 - Daily Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('station_001_temperature.png', dpi=300)
plt.show()
```

## Fitting A Normal Distribution

``` {.python caption="Normal Distribution Fitting" language="python"}
from plugins.distributions.normal import NormalDistribution

# Extract Tmean Data For The First 30 Years
tmean_data = station_data\left[:365, 1\right]  # First 365 days (one year)

# Fit Normal Distribution
dist = NormalDistribution()
result = dist.fit(tmean_data)

print("Normal Distribution Fit Results:")
print(f" Mean (μ): {result\left['p1'\right]:.3f}")
print(f" Std Dev (σ): {result\left['p2'\right]:.3f}")
print(f" Log-likelihood: {result\left['loglik'\right]:.3f}")
print(f" AICc: {result\left['aicc'\right]:.3f}")
print(f" BIC: {result\left['bic'\right]:.3f}")
```

## Fitting All Distributions

``` {.python caption="All Distributions Fitting" language="python"}
from core.engine.plugin_loader import load_plugins
from core.engine.distribution_plugin import DistributionPlugin

# Load All Plugin Distributions
plugins = load_plugins()
print(f"Loaded {len(plugins)} distributions: {\left[p.name for p in plugins.values()\right]}")

# Fit All Distributions To The Data
results = {}
for code, dist in plugins.items():
    try:
        res = dist.fit(tmean_data)
        results\left[dist.name\right] = res
        print(f"{dist.name}: AICc = {res.get('aicc', np.nan):.3f}")
    except Exception as e:
        print(f"{dist.name}: Failed - {str(e)}")

# Find The Best Distribution (minimum Aicc)
best_dist = min(results.items(), key=lambda x: x\left[1\right].get('aicc', np.inf))
print(f"\n✅ Best distribution: {best_dist\left[0\right]} (AICc = {best_dist\left[1\right]\left['aicc'\right]:.3f})")
```

# Project Structure

``` {.text caption="Project Directory Structure" language="text"}
ClimateProcessingEngine/
├── core/                          # Core framework (abstract layers)
│   ├── engine/                    # Engine components
│   │   ├── distribution_plugin.py # Base class for all distributions
│   │   └── plugin_loader.py       # Auto-discover and load plugins
│   ├── interfaces/                # Data interface adapters
│   │   └── data_adapter.py        # Station & gridded data adapters
│   ├── storage/                   # Storage backends
│   │   └── zarr_schema.py         # Zarr output schema definition
│   ├── quality/                   # Quality control system
│   │   └── quality_flag.py        # Quality flag definitions and evaluator
│   └── uncertainty/               # Uncertainty quantification
│       └── bootstrap.py           # Bootstrap confidence intervals
├── plugins/                       # Plugin architecture
│   ├── distributions/             # Distribution plugins
│   │   ├── normal.py              # Normal distribution
│   │   ├── skewnormal.py          # Skew-Normal distribution
│   │   ├── gev.py                 # Generalized Extreme Value
│   │   ├── bimodal.py             # Bimodal mixture distribution
│   │   └── pearson.py             # Pearson Type III
│   ├── qc/                        # Quality control plugins (future)
│   │   └── (empty)
│   └── statistics/                # Statistical metric plugins (future)
│       └── (empty)
├── notebooks/                     # Jupyter Notebook tutorials
│   └── 01_Quick_Start.ipynb
├── tests/                         # Test suite
│   ├── test_distributions.py      # Distribution unit tests
│   ├── test_adapters.py           # Data adapter tests
│   ├── test_quality.py            # Quality flag tests
│   └── expected_results/          # Golden results for CI
│       └── station_001_expected.csv
├── benchmark/                     # Performance benchmarks
│   └── benchmark.py               # Benchmark runner
├── docs/                          # Sphinx documentation
│   └── source/
│       ├── conf.py                # Sphinx configuration
│       ├── index.rst              # Documentation home page
│       ├── installation.rst
│       ├── usage.rst
│       ├── api.rst
│       └── contributing.rst
├── sample_data/                   # Sample data (10 synthetic stations)
│   ├── station_001.csv
│   ├── station_002.csv
│   ├── ...
│   └── station_010.csv
├── .github/                       # GitHub-specific files
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD
├── config.yaml                    # Main configuration file
├── main.py                        # Entry point
├── pyproject.toml                 # Package metadata and build configuration
├── setup.cfg                      # Additional configuration
├── LICENSE                        # MIT License
├── CITATION.cff                   # Citation information
├── .pre-commit-config.yaml        # Pre-commit hooks
├── .gitignore                     # Git ignore patterns
├── .readthedocs.yaml              # ReadTheDocs configuration
├── MANIFEST.in                    # Package manifest
├── README.md                      # This file
├── CONTRIBUTING.md                # Contributing guidelines
├── CODE_OF_CONDUCT.md             # Code of conduct
└── SECURITY.md                    # Security policy
```

# Configuration

The main configuration file is `config.yaml`. This file controls all
aspects of the engine.

## Complete Configuration Example

``` {.yaml caption="Full config.yaml" language="yaml"}
# General Project Settings
project:
  name: "Climatology Engine"
  version: "4.0.0"
  description: "Climate data distribution fitting and analysis"

# Data Paths
paths:
  input_zarr_base: "M:/temp/zarr_input"
  output_dir: "./nature_output"
  output_zarr_name: "climatology_stationwise_final.zarr"
  checkpoint_dir: "./nature_output"
  log_dir: "./logs"
  cache_dir: "./cache"
  sample_data_dir: "./sample_data"

# Time Range
time:
  start_year: 1370
  end_year: 1399
  n_days: 366

# Processing Parameters
processing:
  block_size: 1000
  max_blocks_in_memory: 5
  chunk_size: \left[100, 100\right]
  compression: "zstd"
  compression_level: 3
  n_points_max: 40000
  output_precision: "float32"

# Window Extraction Parameters
window:
  days: 2
  use_extreme_values: false
  min_valid_years: 10
  window_type: "centered"

# Active Distributions (per Processing Mode)
distributions:
  normal_mode:
    - normal
    - skew
    - bimodal
    - pearson
  extreme_mode:
    - normal
    - skew
    - gev
    - bimodal
    - pearson

# Data Format And Spatial Extent
data_format: "auto"
lat_min: 25.0
lat_max: 40.0
lon_min: 44.0
lon_max: 64.0
point_sampling: "all"
n_sample_points: 40000

# Cache System
cache:
  enabled: true
  max_size_gb: 10
  ttl_hours: 24
  cache_path: "./cache"

# Quality Control
quality:
  min_sample_size: 3
  threshold_aicc: 1000
  threshold_skew: 5.0
  detect_outliers: true
  outlier_sigma: 4.0

# Bootstrap Uncertainty
bootstrap:
  enabled: true
  n_iterations: 100
  confidence_level: 0.95
  random_seed: 42

# Parallel Processing
parallel:
  enabled: true
  backend: "multiprocessing"
  max_workers: 6
  chunk_size: 100
  use_gpu: false

# Logging
logging:
  level: "INFO"
  console_output: true
  file_output: true
  file_rotation: "10 MB"
  log_format: "detailed"
  log_dir: "./logs"

# Model Selection
selection:
  criterion: "aicc"
  min_delta: 2.0
  report_all: false
  save_weights: true

# Output Settings
output:
  format: "zarr"
  overwrite: true
  include_metadata: true
  include_provenance: true
  compress: true
  compression_algorithm: "zstd"
  compression_level: 3

# Visualization
visualization:
  enabled: true
  output_dir: "./visualizations"
  format: \left["png", "pdf"\right]
  interactive: true
  map_projection: "PlateCarree"
  dpi: 300
```

# Output Schema

## Reading The Output

``` {.python caption="Reading Output Data" language="python"}
import xarray as xr

# Open The Zarr Output
ds = xr.open_zarr("climatology_stationwise_final.zarr", consolidated=False)

# List All Variables
print(list(ds.data_vars))

# Access The Best Distribution
best_dist = ds\left['best_dist'\right].values  # Shape: (n_points, 366)

# Access The Mean Temperature
mean_temp = ds\left['mean'\right].values  # Shape: (n_points, 366)

# Access Station Metadata
station_ids = ds\left['stationid'\right].values
lats = ds\left['lat'\right].values
lons = ds\left['lon'\right].values
elevs = ds\left['elev'\right].values

# Close The Dataset
ds.close()
```

# Statistical Distribution Fitting

## Normal Distribution

The Normal (Gaussian) distribution is the most commonly used
distribution in climatology. Its probability density function (PDF) is:

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

**Parameters:**

-   $\mu$: Mean (location parameter)

-   $\sigma$: Standard deviation (scale parameter, $\sigma > 0$)

**Properties:**

-   Symmetric about the mean

-   Support: $(-\infty, \infty)$

-   Skewness = 0, Excess kurtosis = 0

-   Maximum entropy distribution for given mean and variance

**Application:** Temperature data, precipitation anomalies, standardized
indices.

**Estimation:** Maximum Likelihood Estimation (MLE):
$$
\hat{\mu} = \bar{x} \quad \text{(sample mean)}
$$
$$
\hat{\sigma} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \bar{x})^2} \quad \text{(sample standard deviation)}
$$

**Goodness of fit:** The Normal distribution is recommended when the
data are symmetric and the sample size is large.

## Skew-normal Distribution

The Skew-Normal distribution extends the Normal distribution to allow
for asymmetry. Its PDF is:

$$
f(x) = \frac{2}{\omega} \phi\left(\frac{x-\xi}{\omega}\right) \Phi\left(\alpha\frac{x-\xi}{\omega}\right)
$$

where $\phi$ is the standard Normal PDF and $\Phi$ is the standard
Normal CDF.

**Parameters:**

-   $\xi$: Location parameter (not the mean unless $\alpha = 0$)

-   $\omega$: Scale parameter ($\omega > 0$)

-   $\alpha$: Shape parameter (controls skewness)

**Properties:**

-   Support: $(-\infty, \infty)$

-   Skewness varies with $\alpha$

-   When $\alpha = 0$, it reduces to Normal($\xi, \omega^2$)

-   Mean: $\mu = \xi + \omega \delta \sqrt{2/\pi}$ where
    $\delta = \alpha / \sqrt{1 + \alpha^2}$

-   Variance: $\sigma^2 = \omega^2 (1 - 2\delta^2/\pi)$

**Application:** Temperature data with asymmetry, precipitation with
skewness.

**Estimation:** MLE with iterative optimization.

## Gev Distribution (extreme Value Mode)

The Generalized Extreme Value (GEV) distribution is the limit
distribution of block maxima (or minima). It is only active in extreme
value mode. The PDF is:

\[
f(x) = \frac{1}{\sigma}
\left(1 + \xi \frac{x - \mu}{\sigma}\right)^{-\left(1/\xi + 1\right)}
\exp\left(
-\left(1 + \xi \frac{x - \mu}{\sigma}\right)^{-1/\xi}
\right)
\]

**Parameters:**

-   $\mu$: Location parameter

-   $\sigma$: Scale parameter ($\sigma > 0$)

-   $\xi$: Shape parameter

**Constraints:**
\[
1 + \xi \frac{x - \mu}{\sigma} > 0
\]

**Three types:**

-   $\xi = 0$: Gumbel distribution (light tails)

-   $\xi > 0$: Fréchet distribution (heavy tails)

-   $\xi < 0$: Weibull distribution (bounded upper tail)

**Properties:**

-   Support:

    -   $\xi > 0$: $x \in [\mu - \sigma/\xi, \infty)$

    -   $\xi < 0$: $x \in (-\infty, \mu - \sigma/\xi]$

    -   $\xi = 0$: $x \in (-\infty, \infty)$

-   Models block maxima (e.g., annual maximum temperature)

**Application:** Extreme temperature analysis, return period estimation
(100-year events).

**Estimation:** MLE with constraints on $\xi$.

## Bimodal Normal Distribution

The Bimodal Normal distribution is a mixture of two Normal
distributions. It is suitable for data with two distinct peaks (e.g.,
seasonal temperature patterns, precipitation with two regimes). The PDF
is:

$$
f(x) = w_1 \mathcal{N}(x; \mu_1, \sigma_1^2) + w_2 \mathcal{N}(x; \mu_2, \sigma_2^2)
$$

**Parameters:**

-   $w_1, w_2$: Mixture weights ($w_1 + w_2 = 1$, $0 \leq w_i \leq 1$)

-   $\mu_1, \mu_2$: Means of the two components

-   $\sigma_1, \sigma_2$: Standard deviations of the two components

**Properties:**

-   Support: $(-\infty, \infty)$

-   Can model two distinct climate regimes

-   Bimodality if $|\mu_1 - \mu_2|$ is large relative to
    $\sigma_1, \sigma_2$

**Metrics:**

-   **Ashman's D:** Measures separation between modes
    $$
D = \frac{|\mu_2 - \mu_1|}{\sqrt{(\sigma_1^2 + \sigma_2^2)/2}}
$$
    $D > 2$ indicates clear bimodality.

-   **Overlap Coefficient (OVL):**
    $$
\text{OVL} = \int_{-\infty}^{\infty} \min\{f_1(x), f_2(x)\} \, dx
$$
    OVL ranges from 0 (no overlap) to 1 (complete overlap).

**Application:** Temperature with warm and cold seasons, precipitation
with wet and dry regimes.

**Estimation:** Expectation-Maximization (EM) algorithm or MLE.

## Pearson Type Iii Distribution

The Pearson Type III distribution is a shifted Gamma distribution,
widely used in hydrology and climatology for positively skewed data
(e.g., precipitation, flood flows). The PDF is:

$$
f(x) = \frac{1}{\Gamma(\alpha)\beta^\alpha} (x-\gamma)^{\alpha-1} \exp\left(-\frac{x-\gamma}{\beta}\right)
$$

**Parameters:**

-   $\alpha$: Shape parameter ($\alpha > 0$)

-   $\beta$: Scale parameter ($\beta > 0$)

-   $\gamma$: Location parameter (lower bound)

**Properties:**

-   Support: $x \in (\gamma, \infty)$

-   Skewness: positive if $\alpha > 0$

-   Gamma distribution is a special case ($\gamma = 0$)

-   Mean: $\mu = \gamma + \alpha\beta$

-   Variance: $\sigma^2 = \alpha\beta^2$

**Application:** Precipitation analysis, streamflow, flood frequency
analysis.

**Estimation:** Method of moments or MLE.

# Model Selection Criteria

The engine selects the best distribution using information criteria. All
criteria aim to balance goodness-of-fit with model complexity.

## Akaike Information Criterion (aic)

AIC estimates the relative quality of statistical models for a given
dataset.

$$
\text{AIC} = 2k - 2\ln(\hat{L})
$$

where:

-   $k$: Number of parameters

-   $\hat{L}$: Maximized likelihood value

**Interpretation:**

-   Lower AIC indicates a better model

-   AIC rewards goodness of fit (higher likelihood) and penalizes
    complexity (more parameters)

-   Not an absolute test; only relative differences matter

**Thresholds:**

-   $\Delta\text{AIC} > 10$: Essentially no support for the worse model

-   $4 < \Delta\text{AIC} \leq 7$: Weak support

-   $2 < \Delta\text{AIC} \leq 4$: Moderate support

-   $\Delta\text{AIC} \leq 2$: Substantial support

## Corrected Aic (aicc)

AICc is AIC with a correction for small sample sizes.

$$
\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n - k - 1}
$$

where $n$ is the sample size.

**When to use:**

-   Recommended when $n/k < 40$

-   As $n \to \infty$, AICc $\to$ AIC

-   More stringent penalization for complex models with small samples

## Bayesian Information Criterion (bic)

BIC is derived from a Bayesian perspective and penalizes complexity more
strongly than AIC.

$$
\text{BIC} = k\ln(n) - 2\ln(\hat{L})
$$

**Characteristics:**

-   Penalizes parameters more heavily ($\ln(n)$ vs 2)

-   Asymptotically consistent (selects the true model if it is among
    candidates)

-   Tends to favor simpler models compared to AIC

## Likelihood Ratio Test

For nested models, the Likelihood Ratio Test can determine if the more
complex model significantly improves the fit.

$$
\Lambda = 2(\ln(\hat{L}_1) - \ln(\hat{L}_0))
$$

where:

-   $\hat{L}_1$: Likelihood of the more complex model

-   $\hat{L}_0$: Likelihood of the simpler model

**Interpretation:**

-   $\Lambda$ follows a $\chi^2$ distribution with degrees of freedom
    equal to the difference in the number of parameters

-   Significant $\Lambda$ indicates the complex model is better

# Quality Control System

## Quality Flags

The engine assigns quality flags to each fit to indicate potential
issues.

## Automatic Flagging

Quality flags are automatically assigned based on:

**Data validation:**

-   Missing values check

-   Infinite value check

-   Sample size check

**Fit validation:**

-   Convergence status

-   Parameter bounds (e.g., $\sigma > 0$, $\alpha > 0$)

-   Likelihood check ($\ln L > -\infty$)

**Statistical validation:**

-   AICc threshold

-   Skewness threshold

-   Variance check

**Diagnostic validation:**

-   Hessian matrix positive definite

-   Gradient norm

## Threshold Configuration

All quality thresholds are configurable in `config.yaml`:

``` {.yaml caption="Quality Control Configuration" language="yaml"}
quality:
  min_sample_size: 3
  threshold_aicc: 1000
  threshold_skew: 5.0
  detect_outliers: true
  outlier_sigma: 4.0
  max_iterations: 1000
  tolerance: 1e-6
  check_hessian: true
```

# Uncertainty Quantification

## Bootstrap Method

The engine uses the parametric bootstrap to estimate the uncertainty of
fitted parameters.

**Method:**

1.  Fit the distribution to the original data to obtain parameter
    estimates $\hat{\theta}$

2.  Generate $B$ bootstrap samples from the fitted distribution

3.  Fit the distribution to each bootstrap sample to obtain
    $\hat{\theta}^{(b)}$ for $b = 1, \ldots, B$

4.  Compute statistics from the bootstrap distribution of $\hat{\theta}$

**Advantages:**

-   Does not rely on asymptotic assumptions

-   Works well for small samples

-   Provides confidence intervals for any statistic

**Bootstrap iterations:**

-   Recommended $B = 100$ for quick assessment

-   $B = 1000$ for publication-quality results

## Confidence Intervals

The engine computes percentile-based confidence intervals:

$$
\mathrm{CI}_{95\\%}(\theta) = \left[\theta_{(0.025)}, \theta_{(0.975)}\right]
$$

where $\theta_{(p)}$ is the $p$-th percentile of the bootstrap
distribution.

**Other CI methods (available):**

-   Normal interval: $\bar{\theta} \pm 1.96 \cdot \text{SE}(\theta)$

-   BCa (Bias-Corrected and Accelerated): More robust for skewed
    distributions

-   Studentized interval: Uses bootstrap estimate of standard error

## Parameter Uncertainty

For each fitted distribution, the engine outputs:

``` {.python caption="Parameter Uncertainty Output" language="python"}
{
    'mean': {
        'value': 15.23,
        'lower_ci': 14.87,
        'upper_ci': 15.59,
        'std_err': 0.18
    },
    'scale': {
        'value': 4.12,
        'lower_ci': 3.89,
        'upper_ci': 4.35,
        'std_err': 0.12
    },
    'shape': {
        'value': 0.34,
        'lower_ci': 0.28,
        'upper_ci': 0.40,
        'std_err': 0.03
    }
}
```

# Plugin Architecture

## Adding New Distributions

To add a new distribution:

1.  Create a file in `plugins/distributions/` (e.g.,
    `my_distribution.py`)

2.  Define a class inheriting from `DistributionPlugin`

3.  Implement the `fit()` method

``` {.python caption="New Distribution Example" language="python"}
from core.engine.distribution_plugin import DistributionPlugin

class MyDistribution(DistributionPlugin):
    name = "MyDistribution"
    code = 5
    params = \left["p1", "p2", "p3"\right]
    n_params = 3
    supports_negative = True
    supports_zero = True
    supports_positive = True
    extreme_only = False

    def fit(self, data):
        # Your fitting algorithm here
        # Must return a dict with parameter values and metrics
        return {
            "p1": value1,
            "p2": value2,
            "p3": value3,
            "loglik": loglik,
            "aicc": aicc,
            "bic": bic,
        }
```

The engine will automatically discover and load your plugin.

## Plugin Structure

Each distribution plugin must implement:

``` {.python caption="Plugin Base Class" language="python"}
class DistributionPlugin:
    # Required class attributes
    name = None  # Display name (string)
    code = None  # Unique integer code
    params = \left[\right]  # List of parameter names (strings)
    n_params = 0  # Number of parameters (integer)

    # Optional class attributes
    supports_negative = False
    supports_zero = False
    supports_positive = False
    extreme_only = False
    requires_bootstrap = False
    require_initial_guess = False

    # Required method
    def fit(self, data):
        """Fit the distribution to data.
        
        Args:
            data: 1D numpy array of observations
        
        Returns:
            dict with keys: param names, 'loglik', 'aicc', 'bic'
        """
        raise NotImplementedError

    # Optional methods
    def initial_guess(self, data):
        """Provide initial parameter guesses for optimization."""
        return \left[np.mean(data), np.std(data)\right]

    def pdf(self, x, params):
        """Probability density function."""
        pass

    def cdf(self, x, params):
        """Cumulative distribution function."""
        pass

    def ppf(self, p, params):
        """Percent point function (inverse CDF)."""
        pass

    def rvs(self, size, params):
        """Generate random samples from the distribution."""
        pass
```

## Loading Plugins

The engine loads all plugin distributions automatically:

``` {.python caption="Loading Plugins" language="python"}
from core.engine.plugin_loader import load_plugins

plugins = load_plugins()  # plugins is a dict: {code: DistributionPlugin}
```

To load only specific distributions:

``` {.python caption="Loading Specific Plugins" language="python"}
from core.engine.plugin_loader import load_plugins
from plugins.distributions.normal import NormalDistribution
from plugins.distributions.skewnormal import SkewNormalDistribution

plugins = {
    0: NormalDistribution(),
    1: SkewNormalDistribution()
}
```

# Data Adapters

## Station Data Adapter

Handles station-based data where each station has a unique identifier.

``` {.python caption="Station Data Adapter" language="python"}
from core.interfaces.data_adapter import StationDataAdapter

adapter = StationDataAdapter(
    zarr_base="M:/temp/zarr_input",
    year_list=\left[1370, 1371, ..., 1399\right],
    cache_enabled=True,
    max_points=40000
)

# Get Coordinates
coords = adapter.get_coords()
station_ids = coords\left['stationid'\right]
lats = coords\left['lat'\right]
lons = coords\left['lon'\right]
elevs = coords\left['elev'\right]

# Load A Block Of Data
data_block = adapter.load_block(
    block_start=0,
    block_size=1000,
    year_idx=0,
    month=1,
    var_idx=1  # tmean
)
```

## Gridded Data Adapter

Handles gridded data with dimensions (time, latitude, longitude).

``` {.python caption="Gridded Data Adapter" language="python"}
from core.interfaces.data_adapter import GriddedDataAdapter

adapter = GriddedDataAdapter(
    zarr_base="M:/temp/zarr_input",
    year_list=\left[1370, ..., 1399\right],
    cache_enabled=True,
    max_points=40000,
    lat_min=25.0,
    lat_max=40.0,
    lon_min=44.0,
    lon_max=64.0
)

# Data Is Automatically Flattened To (time, Point) Format
data_block = adapter.load_block(
    block_start=0,
    block_size=1000,
    year_idx=0,
    month=1,
    var_idx=1  # tmean
)
```

## Auto-detection

The engine can automatically detect the data format:

``` {.python caption="Auto-Detection" language="python"}
from core.interfaces.data_adapter import create_adapter

adapter = create_adapter(
    zarr_base="M:/temp/zarr_input",
    year_list=year_list,
    data_format="auto",  # auto, station, gridded
    cache_enabled=True,
    max_points=40000
)

# The Adapter Detects Station Or Gridded Format From The Zarr Files
# And Returns The Appropriate Adapter Instance
```

# Processing Modes

## Normal Mode

Default mode for standard climatological analysis.

**Window extraction:** For each day, extract values from the target day
and 2 days before/after for each year.

-   Total values per day: $5 \times 30 = 150$ observations

-   Distributions used: Normal, Skew-Normal, Bimodal, Pearson III

-   GEV is disabled

-   Best distribution selected by AICc

**Application:** General climatology studies, seasonal cycle analysis.

## Extreme Value Mode

Specialized mode for extreme events analysis.

**Window extraction:** For each day, extract the maximum and minimum
from the 5-day window for each year.

-   Maxima: 30 values (one per year)

-   Minima: 30 values (one per year)

-   Distributions used: Normal, Skew-Normal, GEV, Bimodal, Pearson III

-   GEV is enabled and often outperforms other distributions

-   Best distribution selected by AICc

**Application:** Return period estimation, heatwave analysis, cold
spells, flood frequency.

**Activation:**

``` {.yaml caption="Activating Extreme Value Mode" language="yaml"}
# Config.yaml
window:
  use_extreme_values: true  # Enable extreme value mode
```

# Performance Optimization

## Block-based Processing

The engine processes data in blocks to manage memory efficiently.

**How it works:**

-   Data is divided into blocks of `block_size` stations

-   Each block is loaded, processed, and written separately

-   Memory usage is proportional to `block_size`, not `n_stations`

-   Checkpoint is saved after each block

**Configuration:**

``` {.yaml caption="Block Size Configuration" language="yaml"}
processing:
  block_size: 1000  # Smaller block size = lower memory usage, more I/O
                    # Larger block size = higher memory usage, less I/O
```

## Parallel Processing

The engine supports multiple parallel backends.

**Backend options:**

**Configuration:**

``` {.yaml caption="Parallel Processing Configuration" language="yaml"}
parallel:
  enabled: true
  backend: "multiprocessing"
  max_workers: 6
  chunk_size: 100
```

## Memory Management

**Memory optimization strategies:**

### Data Types

Using `float32` instead of `float64` reduces memory by 50%:

``` {.python caption="Using float32" language="python"}
arr = np.full((n_days, n_stations), np.nan, dtype=np.float32)
```

### Block Processing

Never load all data at once:

``` {.python caption="Block Processing Loop" language="python"}
for block_start in range(0, n_stations, block_size):
    block_data = load_block(block_start, block_size)
    process_block(block_data)
```

### Garbage Collection

Explicitly free memory:

``` {.python caption="Garbage Collection" language="python"}
import gc
del block_data
gc.collect()
```

### Memory Monitoring

Track memory usage:

``` {.python caption="Memory Monitoring" language="python"}
import psutil
mem = psutil.Process().memory_info()
print(f"Memory usage: {mem.rss / 1024**3:.2f} GB")
```

**Configuration:**

``` {.yaml caption="Memory Configuration" language="yaml"}
processing:
  max_blocks_in_memory: 5
  output_precision: "float32"
```

## Caching System

The disk cache reduces I/O by storing loaded data.

**How it works:**

**Advanced Features:**

-   **Multi-block-size detection**: The cache system automatically searches for existing cache files with block sizes 1000, 2000, and 5000, ensuring maximum reuse regardless of previous processing settings.
-   **Intelligent cache lookup**: When loading data, the system first checks all possible block size and sample hash combinations before falling back to Zarr I/O.
-   **Selective cache building**: Use `build_cache_only.py` to pre-build cache files for specific blocks without running the full analysis pipeline.


-   First time data is loaded, it's cached

-   Subsequent requests load from cache (faster)

-   Old cache entries are automatically purged

**Configuration:**

``` {.yaml caption="Cache Configuration" language="yaml"}
cache:
  enabled: true
  max_size_gb: 10
  ttl_hours: 24
  cache_path: "./cache"
```

## Compression

Zarr uses compression to reduce file size.

**Compression options:**

**Configuration:**

``` {.yaml caption="Compression Configuration" language="yaml"}
processing:
  compression: "zstd"
  compression_level: 3  # 1 = fast, 9 = high compression
```

# Checkpoint & Recovery

The engine automatically saves checkpoints to enable recovery from
interruptions.

**Checkpoint Format:** Checkpoints are stored in key=value format:

``` {.text caption="Checkpoint Format" language="text"}
block=86 station=86999 timestamp=1785060753 version=1
```

**Recovery Process:**

1.  Engine starts and checks for existing checkpoint

2.  If found, loads block and station

3.  Resumes processing from that exact point

4.  If not found, starts from the beginning

**Save frequency:** Checkpoints are saved after every 100 stations or at
block boundaries.


**Auto-Detection (New in v4.1):**

The engine now includes an **auto-detection** feature that examines the output Zarr file to find the last valid processing point. If a checkpoint file is missing or corrupted, the system will automatically determine the correct starting block and station based on the data already written to Zarr.

This eliminates the need for manual checkpoint management and ensures seamless recovery even if the checkpoint file is accidentally deleted.

**Manual Checkpoint Management (Legacy):**

For backward compatibility, manual checkpoint management is still supported:

**Manual checkpoint management:**

``` {.python caption="Manual Checkpoint Management" language="python"}
from monitoring.checkpoint import save_checkpoint, load_checkpoint

# Save Checkpoint
save_checkpoint("nature_output", block=86, station=86999)

# Load Checkpoint
cp = load_checkpoint("nature_output")
print(f"Last block: {cp.get('block')}")
```

# Testing

## Unit Tests

``` {.bash caption="Running Unit Tests" language="bash"}
pytest tests/test_distributions.py -v
```

## Integration Tests

``` {.bash caption="Running Integration Tests" language="bash"}
pytest tests/ --run-integration
```

Integration tests include:

-   Full pipeline on sample data

-   Comparison with expected results

-   Performance regression tests

## Coverage Reports

``` {.bash caption="Coverage Reports" language="bash"}
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

**Coverage targets:**

-   Core modules: $\geq 95\%$

-   Plugin modules: $\geq 90\%$

-   Overall: $\geq 90\%$

**High-priority areas:**

-   `distributions.py` (core fitting)

-   `plugin_loader.py` (discovery)

-   `quality_flag.py` (quality control)

-   `bootstrap.py` (uncertainty)

# Benchmarking

## Performance Metrics

The benchmark measures:

-   Load time

-   Window extraction time

-   Distribution fitting time

-   Quality control time

-   Bootstrap time

-   Write time

-   Memory usage

## Benchmark Results

Run the benchmark:

``` {.bash caption="Running Benchmark" language="bash"}
python benchmark/benchmark.py
```

Example output:

``` {.text caption="Benchmark Output" language="text"}
============================================================
Performance Benchmark
============================================================
Sample size: 1000 stations
Mode: Normal
Workers: 6

Operation           Time (s)   Memory (GB)
------------------  ---------- ------------
Load data           12.34      1.23
Window extraction   8.90       2.45
Distribution fit    45.67      3.56
Quality control     5.67       0.78
Bootstrap           18.90      1.89
Write output        7.89       1.11
Total               99.37      3.56
============================================================
```

# Documentation

## Sphinx Documentation

Build the full documentation:

``` {.bash caption="Building Documentation" language="bash"}
cd docs
make html
open _build/html/index.html
```

Documentation structure:

``` {.text caption="Documentation Structure" language="text"}
docs/
├── source/
│   ├── index.rst
│   ├── installation.rst
│   ├── usage/
│   │   ├── quickstart.rst
│   │   ├── configuration.rst
│   │   ├── running.rst
│   │   └── output.rst
│   ├── api/
│   │   ├── core.rst
│   │   ├── plugins.rst
│   │   └── utilities.rst
│   ├── theory/
│   │   ├── distributions.rst
│   │   ├── selection.rst
│   │   └── uncertainty.rst
│   ├── contributing.rst
│   └── changelog.rst
└── build/
```

## Jupyter Notebooks

Interactive tutorials are available in the `notebooks/` directory:

``` {.bash caption="Running Jupyter Notebooks" language="bash"}
jupyter notebook notebooks/01_Quick_Start.ipynb
```

# Troubleshooting

## Common Errors

### Filenotfounderror

``` {.text caption="FileNotFoundError" language="text"}
FileNotFoundError: \left[WinError 3\right] The system cannot find the path specified
```

**Causes:**

-   Incorrect paths in `config.yaml`

-   Missing input data

-   Permission issues

**Solutions:**

-   Check paths in `config.yaml` (use absolute paths)

-   Verify input files exist

-   Check read permissions

-   Create necessary directories

### Memoryerror

``` {.text caption="MemoryError" language="text"}
MemoryError: Unable to allocate array with shape (10000, 366, 30)
```

**Causes:**

-   Block size too large

-   Too many parallel workers

-   Data type too large (float64 vs float32)

**Solutions:**

-   Reduce `block_size` in config

-   Reduce `max_workers`

-   Use float32 output precision

-   Enable disk cache

### Keyerror

``` {.text caption="KeyError" language="text"}
KeyError: 'variable_name'
```

**Causes:**

-   Incorrect variable name in config

-   Missing variable in input data

-   Incorrect variable index

**Solutions:**

-   Check variable names in `config.yaml`

-   Verify input dataset schema

-   Update VARS in `constants.py`

### Convergencewarning

``` {.text caption="ConvergenceWarning" language="text"}
ConvergenceWarning: Optimization did not converge
```

**Causes:**

-   Poor initial guesses

-   Insufficient data

-   Difficult parameter space

**Solutions:**

-   Increase `max_iterations`

-   Provide better initial guesses

-   Add more data if possible

-   Consider different distribution

### Zarrstoreerror

``` {.text caption="ZarrStoreError" language="text"}
ZarrStoreError: Failed to write to Zarr store
```

**Causes:**

-   Disk full

-   Permission issues

-   Corrupted store

**Solutions:**

-   Check disk space

-   Verify write permissions

-   Delete corrupted store and restart

## General Debugging

**Enable DEBUG logging:**

``` {.yaml caption="Debug Logging" language="yaml"}
logging:
  level: "DEBUG"
```

**Run on a smaller sample:**

``` {.python caption="Smaller Sample" language="python"}
# In Main.py, Reduce N_points_max
n_points_max = 100  # Instead of 40000
```

**Check log files:**

``` {.bash caption="Check Logs" language="bash"}
tail -f logs/climatology.log
```

**Use interactive debugger:**

``` {.python caption="Interactive Debugger" language="python"}
import ipdb; ipdb.set_trace()
```

# Contributing

## Development Setup

``` {.bash caption="Development Setup" language="bash"}
# Clone The Repository
git clone Source: https://github.com/AminFazlKazemi/ClimateProcessingEngine.git
cd ClimateProcessingEngine

# Create And Activate Virtual Environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install Development Dependencies
pip install -e .\left[dev\right]

# Install Pre-commit Hooks
pre-commit install

# Run Tests
pytest tests/
```

## Code Style

The project follows the Black code style.

``` {.bash caption="Code Style Checks" language="bash"}
# Format Code
black .

# Check Code Style
flake8 .

# Type Checking
mypy .

# Sort Imports
isort .
```

**Style rules:**

-   Max line length: 88 (Black default)

-   Use type hints

-   Write docstrings for all public functions

-   Include unit tests for new features

-   Follow PEP 8 conventions

## Pull Request Process

1.  Fork the repository

2.  Create a branch: `git checkout -b feature/your-feature`

3.  Make changes with tests

4.  Run tests: `pytest tests/ -v`

5.  Format code: `black .`

6.  Commit: `git commit -m "Add your feature"`

7.  Push: `git push origin feature/your-feature`

8.  Open a Pull Request

**PR requirements:**

-   Tests added/updated

-   Code formatted with Black

-   Documentation updated

-   All tests passing

-   No style violations

-   Changelog updated

# License

This project is licensed under the MIT License.

**MIT License Summary:**

-   Commercial use

-   Modification

-   Distribution

-   Private use

-   Liability

-   Warranty

See the `LICENSE` file for the complete license text.

# Citation

If you use this software in your research, please cite:

**BibTeX:**

``` {.bibtex caption="BibTeX Citation" language="bibtex"}
@software{FazlKazemi_ClimateProcessingEngine_2025,
  author = {Fazl Kazemi, Amin},
  title = {ClimateProcessingEngine},
  year = {2025},
  publisher = {GitHub},
  url = {Source: https://github.com/AminFazlKazemi/ClimateProcessingEngine},
  version = {4.0}
}
```

**APA:**

``` {.text caption="APA Citation" language="text"}
Fazl Kazemi, A. (2025). ClimateProcessingEngine (Version 4.0) \left[Computer software\right].
GitHub. Source: https://github.com/AminFazlKazemi/ClimateProcessingEngine
```

# Contact

-   **Author:** Amin Fazl Kazemi

-   **GitHub:** <Source: https://github.com/AminFazlKazemi>

-   **LinkedIn:** <https://linkedin.com/in/aminfazlkazemi>

-   **Email:** <aminfazlkazemi@gmail.com>

-   **Twitter/X:** <https://twitter.com/AminFazlKazemi>

# Acknowledgments

This project builds upon the work of the open-source scientific Python
community:

-   **NumPy & SciPy:** For fundamental numerical computing tools

-   **Xarray & Zarr:** For modern, scalable data structures

-   **Numba:** For high-performance JIT compilation

-   **Matplotlib & Seaborn:** For visualization capabilities

-   **Pandas:** For data manipulation and analysis

-   **Scikit-learn:** For machine learning components

Special thanks to the climate science community for their support and
feedback, and to all contributors who have helped improve this
framework.

# Frequently Asked Questions

## What Is The Recommended Block Size?

The optimal block size depends on your available RAM and the number of
stations.

``` {.python caption="Block Size Formula" language="python"}
block_size = min(1000, int(ram_gb * 100))  # approximate
```

## When Should I Use Extreme Value Mode?

Use extreme value mode when analyzing:

-   Annual maxima (e.g., maximum temperature)

-   Return periods (e.g., 100-year events)

-   Extreme events (heatwaves, cold spells)

-   Climate extremes indices

Set `use_extreme_values: true` in `config.yaml`.

## How Do I Add A New Distribution?

1.  Create a new file in `plugins/distributions/`

2.  Define a class inheriting from `DistributionPlugin`

3.  Implement the `fit()` method

4.  The engine will automatically discover it

**Note:** Make sure to assign a unique code integer (5, 6, 7, ...).

## What Is The Quality Flag System?

The quality flag system automatically evaluates each fit and assigns
flags based on:

-   Sample size

-   Convergence

-   AICc value

-   Skewness

-   Data validity (NaN, Inf)

Flags are stored in the output and can be used for filtering.

## How Is Gev Different From Other Distributions?

GEV is specifically designed for block maxima (e.g., annual maximum
temperature). It is only active in extreme value mode and is not used in
normal mode.

## Can I Process Gridded Data?

Yes! Set `data_format: "gridded"` in `config.yaml` and specify the
spatial extent using `lat_min`, `lat_max`, `lon_min`, `lon_max`.

# Changelog


## Version 4.1 (2026-07-31)

**New Features:**

-   **Intelligent Disk Caching**: Multi-block-size detection (1000, 2000, 5000) to reuse existing cache files and avoid redundant I/O. Significantly speeds up repeated processing.
-   **Auto-Resume Checkpoints**: Automatically detects the last valid processing point from the output Zarr and resumes seamlessly after interruptions. No manual checkpoint management needed.
-   **Selective Cache Builder**: `build_cache_only.py` script to pre-build cache files for unprocessed blocks without running the full statistical analysis.
-   **Multi-block-size Cache Detection**: DiskCache now searches for existing cache files with different block sizes (1000, 2000, 5000) to maximize reuse.

**Bug Fixes:**

-   Fixed `FutureWarning` in `checkpoint_manager.py` by replacing deprecated `ds.dims` with `ds.sizes`.
-   Improved `cache_exists` logic to check all possible block sizes and sample hashes.

**Performance Improvements:**

-   60% faster I/O for repeated processing due to intelligent cache reuse.
-   Reduced memory footprint during cache loading.

## Version 4.0 (2025-07-26)

**Breaking Changes:**

-   Complete migration to plugin architecture

-   New core directory structure (`core/`, `plugins/`)

-   `zarr_schema.py` moved to `core/storage/`

**New Features:**

-   Plugin architecture for distributions

-   Extreme value mode with GEV support

-   Quality flag system

-   Bootstrap uncertainty estimation

-   Disk cache for I/O optimization

-   Gridded data support (time, lat, lon)

-   Sample data and Jupyter notebooks

-   CI/CD with GitHub Actions

**Performance Improvements:**

-   40% faster distribution fitting with Numba

-   30% reduced memory usage with float32

-   Disk cache reducing I/O by 60%

**Bug Fixes:**

-   Fixed checkpoint format incompatibility

-   Fixed memory leak in parallel processing

-   Fixed Zarr store permission issues

-   Fixed NaN handling in bootstrap

## Version 3.0 (2025-06-15)

-   Added Bimodal distribution

-   Added Pearson Type III distribution

-   Parallel processing with multiprocessing

-   Zarr v3 storage format

-   Checkpoint resume functionality

## Version 2.0 (2025-05-01)

-   Added Skew-Normal distribution

-   Zarr output support

-   Window extraction

-   Basic quality control

-   Logging system

## Version 1.0 (2025-04-01)

-   Initial release

-   Normal distribution only

-   CSV output

-   Single-threaded processing

# Roadmap

## Version 4.1 (q3 2025)

-   Bayesian fitting (PyMC integration)

-   Additional distributions: Johnson SU, Wakeby

-   GPU acceleration with CuPy

-   Return period calculation

-   Spatial consistency checks

-   Temporal consistency checks

## Version 4.2 (q4 2025)

-   Dask backend for distributed processing

-   Ray backend integration

-   NetCDF output with CF conventions

-   S3 / Azure / GCS cloud storage

-   REST API

-   Digital twin framework

## Version 5.0 (q1 2026)

-   Full plugin ecosystem

-   Online fitting (streaming data)

-   Ensemble modeling

-   Cross-validation

-   Interactive dashboard

-   Full provenance tracking

# Final Note

If you find this project useful for your research or applications,
please star the repository on GitHub and cite it in your work. Your
support helps us continue developing and maintaining this tool for the
climate science community.

*Built with ❤️ for the climate science community.*

Last updated: July 27, 2026