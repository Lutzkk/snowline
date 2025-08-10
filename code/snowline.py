#snowline.py
#Python 3.10.2


#package imports
#----------
import xarray as xr 
from pathlib import Path
import numpy as np 
import rioxarray as rxr 
import dask 
import os
#----------

#Environment Setup
print(os.getcwd())

project_dir = Path(__file__).resolve().parents[1] #crawl up one layer
os.chdir(project_dir)
data_dir = project_dir / "data"
utils_dir = project_dir / "utils"
output_dir = project_dir / "output"

#----
#load cube 
dc=xr.open_dataset(data_dir /"snowmask_dem_utm31n.nc", engine="netcdf4")

#---------
#calc the time a pixel is snowcovered per year. 

sm=dc["snowmask"]

#sanity check 
counts=sm.resample(time="1D").max()
assert counts.max().item() <= 1 # no error raise implies no duplicates
print("No duplicates found in time dimension.")

#------------------------------------------------------------------------
"""
Since we only have very few actual acquisitions 
(due to the nature of cloudiness in the alps + the poor performance of Sen2Cor Cloud Masking (reason for a conservative cloud cover threshold))
We have enormous data gaps which makes assessing the duration of snow covered days difficult.
"""
