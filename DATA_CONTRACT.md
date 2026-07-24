# Data Contract - climatology_engine v2.0 (Final)

## Load Phase
**ورودی:** block_start, block_size, file_map, year_list, doy_table
**خروجی:** block_data: (block_size, N_YEARS, N_DAYS, N_VARS), float32, C_CONTIGUOUS, NaN for missing

## Window Phase
**ورودی:** station_data: (N_YEARS, N_DAYS, N_VARS), window_table
**خروجی:** List of 366 entries, each ndarray or None

## Analyze Phase
**ورودی:** station_data: (N_YEARS, N_DAYS, N_VARS)
**خروجی:** dict with 33 keys, each (N_DAYS,)

## Write Phase
**ورودی:** block_result: dict with 33 keys, each (N_DAYS, block_size)
**خروجی:** Zarr store with 33 variables, shape (N_DAYS, n_stations)

## Constants
WINDOW_SIZE = 5
MAX_VALUES_PER_FIT = 155
MIN_VALID_VALUES = 5
N_OUTPUTS = 33
VALID_BEST_DIST = {-1, 0, 1, 2, 3}

## Error Types
- IOError: Read/Write errors
- DataError: Invalid data
- FitError: Distribution fitting failed
