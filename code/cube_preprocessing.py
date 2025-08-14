# cube_preprocessing.py
# Python 3.10

from pathlib import Path
import os
import numpy as np
import xarray as xr
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IN_CUBE  = DATA_DIR / "elevation_snowmask_cube.nc"
OUT_NC   = DATA_DIR / "snow_days_per_year.nc"

print("Input cube :", IN_CUBE)
print("Output file:", OUT_NC)
if not IN_CUBE.exists():
    raise FileNotFoundError(f"Missing input: {IN_CUBE}")

# --- load ---
ds = xr.open_dataset(IN_CUBE, engine="netcdf4")
if "snowmask" not in ds or "dem" not in ds:
    raise KeyError("Need variables 'snowmask'(time,y,x) and 'dem'(y,x).")

# tri-state: 1.0 snow, 0.0 no-snow, NaN missing/outside
sm  = ds["snowmask"].astype("float32")
dem = ds["dem"].astype("float32")
in_bounds = ~xr.ufuncs.isnan(dem)

# 1) daily presence (NaN on days w/o acquisition; NaN outside basin)
daily = sm.resample(time="1D").max().where(in_bounds).astype("float32")
# for the gufunc, make time a single chunk; keep spatial chunks small if you like
daily = daily.chunk({"time": -1})

# 2) unlimited gap bridging between consecutive observations
def fill_between_obs(arr_1d: np.ndarray) -> np.ndarray:
    # arr_1d in {0.0, 1.0, NaN}
    out = arr_1d.copy()
    obs_idx = np.flatnonzero(np.isfinite(out))  # positions with 0 or 1
    if obs_idx.size < 2:
        return out
    for left, right in zip(obs_idx[:-1], obs_idx[1:]):
        gap = right - left - 1
        if gap <= 0:
            continue
        a = out[left]
        b = out[right]
        if a == b:
            out[left+1:right] = a
        else:
            first_half = (gap + 1) // 2  # ceil(gap/2)
            out[left+1:left+1+first_half] = a
            out[left+1+first_half:right] = b
    return out

bridged = xr.apply_ufunc(
    fill_between_obs,
    daily,
    input_core_dims=[["time"]],
    output_core_dims=[["time"]],
    vectorize=True,
    dask="parallelized",
    dask_gufunc_kwargs={"output_sizes": {"time": daily.sizes["time"]}},
    output_dtypes=[np.float32],
)

# 3) yearly snow-day counts (keep float so NaN stays NaN)
YEAR_START, YEAR_END = "2018-01-01", "2024-12-31"
subset = bridged.sel(time=slice(YEAR_START, YEAR_END))

valid_days = subset.notnull().groupby("time.year").sum("time")
snow_days_per_year = (
    (subset == 1.0).astype("float32")   # float, not uint, so we can keep NaNs
    .groupby("time.year").sum("time")
).where(valid_days > 0)
snow_days_per_year = snow_days_per_year.rename("snow_days_per_year").astype("float32")

# 4) package & write (no hard-coded chunksizes)
out = xr.Dataset(
    data_vars=dict(
        snow_days_per_year=snow_days_per_year,  # (year, y, x) float32 with NaNs
        dem=dem,                                # (y, x) float32
    ),
    coords=dict(
        year=snow_days_per_year["year"],
        y=ds["y"],
        x=ds["x"],
        spatial_ref=ds.get("spatial_ref"),
    ),
    attrs=dict(
        Conventions="CF-1.9",
        description=(
            "Snow-day counts per pixel per calendar year from Sentinel-2 snowmask. "
            "Daily max, then unlimited gap-bridging between consecutive observations. "
            "NaN retained where no observations exist."
        ),
    ),
)

out["x"].attrs.update(standard_name="projection_x_coordinate", units="m")
out["y"].attrs.update(standard_name="projection_y_coordinate", units="m")
out["dem"].attrs.update(units="m", long_name="Elevation", grid_mapping="spatial_ref")
out["snow_days_per_year"].attrs.update(units="day", grid_mapping="spatial_ref")

encoding = {
    "snow_days_per_year": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "dem": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "x": {"zlib": True, "complevel": 4},
    "y": {"zlib": True, "complevel": 4},
}

tmp = OUT_NC.with_suffix(".nc.tmp")
out.to_netcdf(tmp, engine="netcdf4", mode="w", encoding=encoding)
os.replace(tmp, OUT_NC)
print("Wrote:", OUT_NC)
