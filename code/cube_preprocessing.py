#cube_preprocessing.py
#Python 3.10.2


#package imports
#----------
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
#----------

#Environment Setup


root = Path(os.getcwd()).resolve()
data_dir = root / "data"
data_path = root / "data" / "snowmask_dem_utm31n.nc"
outdir = root / "output" / "figs" / "snow_days_cal"
outdir.mkdir(parents=True, exist_ok=True)


"""
Since we only have very few actual acquisitions 
(due to the nature of cloudiness in the alps + 
the poor performance of Sen2Cor Cloud Masking (reason for a conservative cloud cover threshold))
We have enormous data gaps which makes assessing the duration of snow covered days difficult.

---------------------

At first we will resample the Snowmask to have daily data.
To keep the computational effort as low as possible and avoid Dask usage (for simplicity),
we will work with the data in uint format.
This involves multiple steps: 
1. Ensure valid pixels exist only inside the basin -> mask with dem (incorporated in datacube)
2. Resample the Snowmask to have daily data
   a. Fill gaps with NaN dummies
   b. Use interpolation to create a continuous time series
        I. Rules
           1. Only interpolate over valid (non-NODATA) pixels
           2. If observatioon value is snow before and after -> Pixel becomes snow in between
           3. If observation value is snowfree before and after -> Pixel becomes snowfree in between
           4. If obseration value is snow before and snowfree after -> Pixel becomes snow for the first half and snowfree for the second half
           5. If observation value is snowfree before and snow after -> Pixel becomes snowfree for the first half and snowfree for the second half
3. Calculate the amount of days a pixel is snow covered per calendar year. Then calc the average out of the 5 years available


"""



# Load
ds = xr.open_dataset(data_path, engine="netcdf4")
sm = ds["snowmask"].astype("float32")

# Basin mask
in_bounds = ~np.isnan(ds["dem"])

# Daily target index
timerange = pd.date_range("2017-06-19", "2025-05-30", freq="D")

# --- Normalize snowmask timestamps to daily (00:00) and drop duplicates ---
# (pick 'first' / 'mean' / 'max' depending on what you want if multiple acquisitions per day)
sm_norm = sm.assign_coords(time=pd.to_datetime(sm.time.values).normalize()) \
            .groupby("time").first()

# Mask outside basin
sm_norm = xr.where(in_bounds, sm_norm, np.nan).astype("float32")

# Reindex to daily grid
sm_daily_sparse = sm_norm.reindex(time=timerange)

# Build daily dummy (3 inside basin, NaN outside)
dummy_2d = xr.where(in_bounds, 3.0, np.nan).astype("float32")
dummy_daily = dummy_2d.expand_dims(time=timerange)

# Overlay: observed values overwrite dummies
sm_with_dummies = sm_daily_sparse.combine_first(dummy_daily)

#---------------------------------
#Interpolation

def fill_gap_between_two_observations(
    out: np.ndarray, 
    left_idx: int, 
    right_idx: int
) -> None:
    """
    Mutates `out` in-place between (left_idx, right_idx) according to your rules.

    Rules recap:
      values are {0: NoSnow, 1: Snow, 3: Dummy}
      - If left == right (both 0 or both 1): fill the gap with that value.
      - If left != right: split the gap in half.
          * first half gets the left value
          * second half gets the right value
        For odd gaps, the extra day goes to the FIRST half.
    """
    a = out[left_idx]
    b = out[right_idx]
    gap_len = right_idx - left_idx - 1  # number of positions strictly between the two obs

    if gap_len <= 0:
        return  # adjacent or overlapping, nothing to fill

    # same on both sides -> constant fill
    if a == b:
        out[left_idx + 1 : right_idx] = a
        return

    # different values -> split
    # Example: gap_len=5 -> first_half=3, second_half=2
    first_half = (gap_len + 1) // 2  # ceil(gap_len/2)
    left_start = left_idx + 1
    left_end   = left_idx + 1 + first_half
    right_start = left_end
    right_end   = right_idx

    out[left_start:left_end]   = a
    out[right_start:right_end] = b


def interpolate_snow_dummies_01_3(arr_1d: np.ndarray) -> np.ndarray:
    """
    Interpolates a 1D time-series with values in {0,1,3} (and possibly NaN).
    - 0/1 are real observations
    - 3 is a dummy needing interpolation
    - NaN stays NaN (outside basin)
    - Leading/trailing dummies beyond the first/last observation remain 3
    """
    out = arr_1d.copy()

    # Indices where we actually have observations (0 or 1). Dummies (3) are not observations.
    is_observation = np.isin(arr_1d, (0.0, 1.0))
    obs_indices = np.flatnonzero(is_observation)

    # Need at least two observations to fill anything in-between
    if obs_indices.size < 2:
        return out

    # Walk pairwise through consecutive observations and fill the gap between them
    for left_idx, right_idx in zip(obs_indices[:-1], obs_indices[1:]):
        # ignore if any endpoint is NaN (shouldn't happen, but safe)
        if np.isnan(out[left_idx]) or np.isnan(out[right_idx]):
            continue
        fill_gap_between_two_observations(out, left_idx, right_idx)

    return out


# --- Apply across the full cube (time, y, x) ---
# sm_with_dummies has daily time, 0/1 where observed, 3 where missing, NaN outside basin
sm_interpolated = xr.apply_ufunc(
    interpolate_snow_dummies_01_3,
    sm_with_dummies,                     # (time, y, x)
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    vectorize=True,                      # apply per (y,x) pixel
    dask="parallelized"   
)

#-----------------------------
#3: Calc the amount of days a pixel is snowcovered per calendaric year for the years where full data is available
valid_obs = sm_interpolated.where(sm_interpolated.isin([0, 1]))

# Binary snow (1) vs not-snow (0), NaN elsewhere
snow_binary = xr.where(valid_obs == 1, 1, 0).where(~xr.ufuncs.isnan(valid_obs))

# Focus on full calendar years 2018–2024
subset = snow_binary.sel(time=slice("2018-01-01", "2024-12-31"))
valid_mask = valid_obs.sel(time=slice("2018-01-01", "2024-12-31")).notnull()

#Snow-day count per pixel per year
snow_days_per_year = subset.groupby("time.year").sum("time")  # dims: (year, y, x)


# mask out pixels outside of aoi
snow_days_per_year = snow_days_per_year.where(in_bounds)

#to_netcdf
snow_days_per_year.to_netcdf(data_dir / "snow_days_per_year.nc")
