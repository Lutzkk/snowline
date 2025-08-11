#snowline.py
#Python 3.10.2


#package imports
#----------
from pathlib import Path
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#----------

#Environment Setup


root = Path(__file__).resolve().parents[1]
data_path = root / "data" / "snowmask_dem_utm31n.nc"
outdir = root / "output" / "figs" / "snow_days_cal"
outdir.mkdir(parents=True, exist_ok=True)


#-----------------------------------------------------
"""
Since we only have very few actual acquisitions 
(due to the nature of cloudiness in the alps + 
the poor performance of Sen2Cor Cloud Masking (reason for a conservative cloud cover threshold))
We have enormous data gaps which makes assessing the duration of snow covered days difficult. However 

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
3. Calculate the amount of days a pixel is snow covered per calendar year. Then calc the average out of the 5 years available


"""
# STEP 1:

ds = xr.open_dataset(data_path, engine="netcdf4").astype("uint8")
NODATA=255 # manually set nodata value to stay in uint8
sm_raw=ds["snowmask"]
dem=ds["dem"]

#clip snowmask with dem
mask_valid_elev = (dem >= 897) & (dem <= 2670)
sm_clipped = xr.where(mask_valid_elev, sm_raw, NODATA).astype("uint8")

#-------------------------------------------
#STEP 2:
# Fill with daily NaN Dummies within timerange of valid pixels
# valid timerange: 2017-06-01 - 2025-06-01

time_range=pd.date_range(start="2017-06-01", end="2025-06-01", freq="D")
sm_float=sm_clipped.astype("float32")  # Convert to float to allow NaN values
sm_daily=sm_float.reindex(time=time_range)
sm_work = sm_daily.where(mask_valid_elev)  # outside becomes NaN for the operation 

observed=sm_work.isin([0,1]) #0=nosnow, 1=snow
last_obs=sm_work.where(observed).ffill(dim="time")
next_obs=sm_work.where(observed).bfill(dim="time")

# Fill snow gaps
fill_snow = sm_work.isnull() & (last_obs == 1) & (next_obs == 1)


# fill nonsnow gaps
fill_no_snow = sm_work.isnull() & (last_obs == 0) & (next_obs == 0)

# Combine both rules
to_fill = fill_snow | fill_no_snow

# Apply: use last_obs (or next_obs, both are same here by definition)
sm_filled_rule = xr.where(to_fill, last_obs, sm_work)

sm_filled_rule = xr.where(mask_valid_elev, sm_filled_rule, NODATA)

sm_filled_uint8 = sm_filled_rule.fillna(NODATA).astype("uint8")


