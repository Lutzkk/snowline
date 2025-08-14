#snowline.py
#Python 3.10.2

"""
Github Link: https://github.com/Lutzkk/snowline

Snowline.py is the main script that processes the datacube generated from EarthEngine in cube_generation.py and was preprocessed in cube_preprocessing.py

Available Data: 
Xarray Dataset: snowmask_dem_utm31n.nc
   Digital Elevation Model: Scale 20M (y, x)
   Days Snowcovered per calendar per year: Scale 20M (year, y, x) - Years Available: 2018-2024
"""




#package imports
"""
Note: Fallback to Whitebox for slope aspect curvature calculation
to circumvent heavy library dependencies of xrspatial

"""
#----------
from pathlib import Path
import xarray as xr
import rioxarray as rxr
import whitebox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

wbt=whitebox.WhiteboxTools() # init whitebox
#----------

#Environment Setup


here = Path(__file__).resolve().parent
root=here.parent
print(root)
data_dir = root / "data" 
outdir = root / "output" / "figs" 
outdir.mkdir(parents=True, exist_ok=True)


#-----------------------------------------------------

ds=xr.open_dataset(data_dir / "snow_days_per_year.nc", engine="netcdf4") 
dem=ds["dem"]
snow_days_per_year=ds["snow_days_per_year"]


print(ds)
print("dem:", dem.dims, dem.sizes)
print("snow:", snow_days_per_year.dims, snow_days_per_year.sizes)
print("coords:", list(ds.coords))

"""
The print statements above serve as a sanity check, to ensure the cube is properly structured. 
Since x and y are matching for both
"""
#-----------------------------------------------
mean_days = snow_days_per_year.mean(dim="year") # calcs the avg number of snowcovered days per pixel across all years
std_days = snow_days_per_year.std(dim="year")   # calcs the std dev of snowcovered days per pixel -> Measure of fluctuation
#-----------------------------------------------

#create slope curvature and aspect
#1. temporarily save the dem array since whitebox works with paths
temp_dir=data_dir / "temp"
temp_dir.mkdir(exist_ok=True)

dem_path = temp_dir / "dem.tif"



wkt=str(ds.spatial_ref.values)
dem=dem.rio.write_crs(wkt) # write crs to dem for temp save
dem.rio.to_raster(dem_path)

slope_path  = temp_dir / "slope_deg.tif"
aspect_path = temp_dir / "aspect_deg.tif"
pcurv_path  = temp_dir / "profile_curv.tif"
hs_path = temp_dir / "hillshade.tif"


#slope
slope = wbt.slope(
    dem=str(dem_path),
    output=str(slope_path),
    zfactor=1.0,
    units="degrees"          # "degrees" or "radians"
)
print("Slope calculation done:", slope)

#aspect
aspect = wbt.aspect(
    dem=str(dem_path),
    output=str(aspect_path),
    zfactor=1.0
)

print("Aspect calculation done:", aspect)

profile_curvature = wbt.profile_curvature(
    dem=str(dem_path),
    output=str(pcurv_path),
    zfactor=1.0
)

print("Profile curvature calculation done:", profile_curvature)

wbt.hillshade(
    dem=str(dem_path),
    output=str(hs_path),
    azimuth=315.0,   
    altitude=45.0,   
    zfactor=1.0      
)

#----------------------------------------------

