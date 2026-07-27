::: center
[ [ github.com/AminFazlKazemi/ClimateProcessingEngine
]{style="color: blue"}
](https://github.com/AminFazlKazemi/ClimateProcessingEngine)
:::

# Overview {#overview .unnumbered}

**ClimateProcessingEngine** is an open-source Python framework designed
for large-scale climate data processing, statistical distribution
fitting, geospatial analysis, and scientific visualization.

The framework provides an integrated workflow for:

- Climate time-series processing

- Temperature and precipitation analysis

- Missing data handling

- Statistical distribution fitting

- Climate variability assessment

- Spatial analysis

- Scientific visualization

- Large-scale Zarr data processing

The project is developed using the scientific Python ecosystem:

- NumPy

- Pandas

- Xarray

- Dask

- SciPy

- Scikit-learn

- Rasterio

- GeoPandas

- Cartopy

- Matplotlib

- Numba

# Target Audience

ClimateProcessingEngine is designed for:

- Climate scientists

- Meteorologists

- Hydrologists

- Environmental researchers

- Agricultural scientists

- Geospatial data engineers

- Graduate students

# Key Features

## Climate Data Processing

The engine supports:

- Minimum temperature analysis ($t_{min}$)

- Mean temperature analysis ($t_{mean}$)

- Maximum temperature analysis ($t_{max}$)

- Precipitation processing

- Climate index calculation

- Long-term variability analysis

## Quality Control

Implemented capabilities:

- Missing value detection

- Outlier identification

- Data validation

- Statistical preprocessing

## Missing Data Handling

Advanced approaches include:

- Window-based estimation

- Interpolation techniques

- Robust statistical methods

# Geospatial Analysis

Supported operations:

- Raster processing

- Vector processing

- Coordinate transformation

- Spatial interpolation

- Regional climate mapping

- Scientific cartography

Core libraries:

- Rasterio

- GeoPandas

- Xarray

- Cartopy

- Shapely

# Architecture

ClimateProcessingEngine follows a modular pipeline-based architecture.

    ClimateProcessingEngine

            |
            |
    Configuration Layer
            |
            |
    Input / Data Pipeline
            |
            |
    Numerical Processing Engine
            |
            |
    Statistical Distribution Engine
            |
            |
    Result Pipeline
            |
            |
    Monitoring System

## Core Components

### Configuration Layer

Responsible for centralized management of:

- File paths

- Climate variables

- Processing parameters

- Validation rules

- Logging configuration

### Input Pipeline

The input layer handles:

- Reading climate archives

- Loading station blocks

- Data assembly

- Input validation

### Numerical Engine

The numerical core performs:

- Window extraction

- Statistical calculation

- Distribution fitting

- Parameter estimation

- Model comparison

### Result Pipeline

Responsible for:

- Output validation

- Result generation

- Zarr dataset writing

### Monitoring System

Provides:

- Logging

- Benchmarking

- Checkpoint recovery

- Performance monitoring

# Processing Workflow

The complete workflow is:

1.  Load raw climate datasets

2.  Assemble station-based data blocks

3.  Validate input data

4.  Extract temporal windows

5.  Fit statistical distributions

6.  Select the best probability model

7.  Validate results

8.  Write output datasets

9.  Store checkpoint information

# Statistical Distribution Fitting

ClimateProcessingEngine fits multiple probability distributions and
selects the optimal model using information criteria.

## Normal Distribution

The probability density function is:

$$
f(x)=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

Parameters:

- $\mu$: Mean value

- $\sigma$: Standard deviation

## Skew-Normal Distribution

The skew-normal density is:

$$
f(x)=
\frac{2}{\omega}
\phi
\left(
\frac{x-\xi}{\omega}
\right)
\Phi
\left(
\alpha
\frac{x-\xi}{\omega}
\right)
$$

Parameters:

- $\xi$: Location parameter

- $\omega$: Scale parameter

- $\alpha$: Shape parameter

## Bimodal Normal Distribution

The bimodal model is defined as a mixture distribution:

$$
f(x)=
w_1
N(x;\mu_1,\sigma_1^2)
+
w_2
N(x;\mu_2,\sigma_2^2)
$$

where:

- $w_1,w_2$: Mixture weights

- $\mu_1,\mu_2$: Distribution means

- $\sigma_1,\sigma_2$: Standard deviations

## Pearson Type III Distribution

The Pearson Type III probability density function:

$$
f(x)=
\frac{
1
}{
\Gamma(\alpha)\beta^\alpha
}
(x-\gamma)^{\alpha-1}
\exp
\left(
-\frac{x-\gamma}{\beta}
\right)
$$

Parameters:

- $\alpha$: Shape parameter

- $\beta$: Scale parameter

- $\gamma$: Location parameter

# Model Selection

The engine evaluates candidate distributions using:

## AIC

$$
AIC=2k-2\ln(\hat{L})
$$

## Corrected AIC

$$
AIC_c
=
AIC+
\frac{2k(k+1)}
{n-k-1}
$$

## Bayesian Information Criterion

$$
BIC=
k\ln(n)-2\ln(\hat{L})
$$

where:

- $k$: Number of parameters

- $n$: Number of observations

- $\hat{L}$: Maximum likelihood estimate

# Bimodality Metrics

## Ashman's D

The separation between two Gaussian modes:

$$
D=
\frac{
|\mu_2-\mu_1|
}
{
\sqrt{
\frac{\sigma_1^2+\sigma_2^2}{2}
}
}
$$

## Overlap Coefficient

$$
OVL=
\int_{-\infty}^{\infty}
\min
\{f_1(x),f_2(x)\}
dx
$$

# Installation

## System Requirements

Recommended environment:

- Python 3.10 or higher

- Minimum RAM: 16 GB

- Recommended RAM: 32 GB

- Storage: 100 GB free space

- CPU: 4 cores minimum

- Recommended CPU: 8+ cores

## Clone Repository

    git clone https://github.com/AminFazlKazemi/ClimateProcessingEngine.git

    cd ClimateProcessingEngine

## Create Virtual Environment

Windows:

    python -m venv venv

    venv\Scripts\activate

Linux/macOS:

    python3 -m venv venv

    source venv/bin/activate

## Install Dependencies

    pip install -r requirements.txt

# Configuration

The main configuration file is:

    climatology_engine/config.yaml

Example configuration:

    paths:

      calendar_file:
          K:/Temp/calendar.txt

      checkpoint_file:
          checkpoint.csv

      output_dir:
          I:/climatology_output

      output_zarr_name:
          climatology_stationwise_final.zarr


    years:

      start: 1369

      end: 1399


    days: 366


    window_days: 2



    processing:

      block_size: 1000

      cores: 6

      use_parallel: false



    variables:

      - tmin

      - tmean

      - tmax



    logging:

      level: INFO

      log_file:
          climatology.log

# Important Constants

  Constant             Value          Description
  -------------------- -------------- --------------------------------------
  WINDOW_SIZE          5              Two days before and after target day
  MAX_VALUES_PER_FIT   155            Maximum fitting samples
  MIN_VALID_VALUES     5              Minimum valid observations
  N_OUTPUTS            87             Number of output variables
  VALID_BEST_DIST      {-1,0,1,2,3}   Distribution codes

# Usage Guide

## Running the Engine

    cd climatology_engine

    python main.py

The engine automatically detects existing checkpoints and resumes
unfinished processing.

## Starting a New Processing Run

    rm checkpoint.csv

    python main.py

## Analyzing Results

After completion:

    python analyze_distributions1.py

# Python Example

Load generated Zarr output:

    import xarray as xr


    ds = xr.open_zarr(
    "climatology_stationwise_final.zarr"
    )


    print(list(ds.data_vars))


    best_distribution = (
    ds.tmin_best_dist
    .isel(point=0)
    .values
    )


    print(best_distribution.shape)

# Output Data

The engine produces:

- Best fitted distribution type

- Distribution parameters

- Statistical scores

- Quality indicators

- Station-wise climatological summaries

- Zarr optimized datasets

# Performance Optimization

ClimateProcessingEngine is designed for efficient processing of
large-scale climate datasets.

## Block-Based Processing

Large datasets are divided into smaller blocks:

- Reduces memory consumption

- Improves cache efficiency

- Enables checkpoint recovery

- Supports parallel execution

## Parallel Processing

The engine supports:

- Multiprocessing

- Concurrent execution

- CPU-based parallel computation

The number of workers can be configured:

    cores: 6

    use_parallel: true

## Memory Management

Optimization strategies:

- Block-wise data loading

- Garbage collection

- Efficient NumPy arrays

- Float32 memory optimization

- Chunked Zarr storage

## Performance Recommendations

Recommended practices:

1.  Use SSD storage

2.  Increase CPU cores when available

3.  Adjust block size according to RAM

4.  Disable unnecessary validation during production runs

5.  Close memory-intensive applications

# Project Structure

The project organization:

    ClimateProcessingEngine/

    |

    |-- climatology_engine/

    |   |

    |   |-- main.py

    |   |-- constants.py

    |   |-- config.yaml

    |   |-- zarr_schema.py

    |   |-- bimodal_normal.py

    |   |-- calendar_tables.py

    |   |-- runtime_tables.py

    |

    |   |-- io_pipeline/

    |   |     |

    |   |     |-- read_month_files.py

    |   |     |-- assemble_block.py

    |   |     |-- validate_block.py

    |

    |   |-- numerical_engine/

    |   |     |

    |   |     |-- window_engine.py

    |   |     |-- distributions.py

    |   |     |-- analyze_station.py

    |   |     |-- merge_results.py

    |

    |   |-- orchestrator/

    |   |     |

    |   |     |-- process_block.py

    |

    |   |-- result_pipeline/

    |   |     |

    |   |     |-- validate_result.py

    |   |     |-- write_block.py

    |

    |   |-- monitoring/

    |         |

    |         |-- benchmark.py

    |         |-- checkpoint.py

    |         |-- logger.py


    |

    |-- README.md

    |-- README.tex

    |-- LICENSE

    |-- requirements.txt

    |-- CITATION.cff

# Dependencies

Required Python packages:

    numpy>=1.24.0

    pandas>=2.0.0

    xarray>=2023.0.0

    dask>=2023.0.0

    rasterio>=1.3.0

    geopandas>=0.14.0

    cartopy>=0.22.0

    matplotlib>=3.7.0

    scipy>=1.10.0

    scikit-learn>=1.3.0

    shapely>=2.0.0

    netcdf4>=1.6.0

    zarr>=2.15.0

    numba>=0.57.0

    pyyaml>=6.0

    psutil>=5.9.0

# Scientific Computing Stack

The framework integrates:

  Category              Libraries
  --------------------- ---------------------------------------
  Numerical Computing   NumPy, SciPy
  Data Processing       Pandas, Xarray, Dask
  Geospatial            Rasterio, GeoPandas, Cartopy, Shapely
  Machine Learning      Scikit-learn
  Performance           Numba
  Visualization         Matplotlib
  Storage               Zarr, NetCDF

# Troubleshooting

## File Not Found Error

Common error:

    FileNotFoundError:
    No such file or directory

Possible solutions:

- Check paths in config.yaml

- Use absolute paths

- Verify input files exist

- Check read permissions

## Memory Error

Example:

    MemoryError:
    Unable to allocate array

Solutions:

- Reduce block size

- Disable parallel processing

- Reduce validation frequency

- Increase available RAM

## Key Error

Example:

    KeyError:
    variable_name

Solutions:

- Check variable names

- Verify input dataset schema

- Update configuration variables

# Logging and Debugging

The engine provides detailed logging:

- Console progress information

- Log file records

- Processing checkpoints

- Error tracking

Available log levels:

    DEBUG

    INFO

    WARNING

    ERROR

## Debugging Recommendations

1.  Enable DEBUG logging

2.  Check climatology.log

3.  Run a small test block

4.  Verify input data quality

5.  Use Python debugger if necessary

# Contributing

Contributions are welcome.

Contribution workflow:

1.  Fork the repository

2.  Create a new branch:

        git checkout -b feature-name

3.  Commit changes:

        git commit -m "Add new feature"

4.  Push changes:

        git push origin feature-name

5.  Submit a Pull Request

# Citation

If you use ClimateProcessingEngine in scientific research, please cite
this project:

> Amin Fazl Kazemi.
>
> **ClimateProcessingEngine: A Python Framework for Large-Scale Climate
> Data Processing, Statistical Distribution Fitting and Scientific
> Analysis.**
>
> GitHub Repository:
>
> <https://github.com/AminFazlKazemi/ClimateProcessingEngine>

# License

This project is released under the MIT License.

See:

    LICENSE

for complete license information.

# Contact

**Amin Fazl Kazemi**

GitHub:

<https://github.com/AminFazlKazemi>

LinkedIn:

<https://linkedin.com/in/aminfazlkazemi>

Email:

[aminfazlkazemi@gmail.com](aminfazlkazemi@gmail.com){.uri}

# Acknowledgements {#acknowledgements .unnumbered}

ClimateProcessingEngine is built upon the open-source scientific Python
ecosystem and benefits from contributions by the global climate and data
science community.

# Final Note {#final-note .unnumbered}

If this project is useful for your climate research, please consider
giving it a star on GitHub.

::: center
**ClimateProcessingEngine**
:::
