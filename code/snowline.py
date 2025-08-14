#snowline.py
#Python 3.10.2
#author: Lutz Kleemann


"""
Github Link: https://github.com/Lutzkk/snowline NOTE: The GitHub Repo contains a yaml to rebuild the environment.
Its HIGHLY recommended to use a clone of the repo instead of the standalone script.

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The script below investigates the dimensions of yearly snowcover in the versoyen basin in the french alps. 
The Basin has an extent of x km² and an elevation range from 915 to 3006 m above sea level.
One of the main reasons for this basin is the absence of coniferous forests, which saves one step in the processing chain of masking these areas out.


More specifically it analyzes spatial distribution patterns and compares it to the altitutde, slope, aspect and curvature of the terrain without relying on regressions or machine learning.
Therefore it should rather be treated like an explorative analysis without actual statistical validation as required within the lectures regulations. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data:
The Datacube ("snow_days_per_year.nc") contains the two xarray data arrays:
    1. Days of Snow Cover per Calendar year:
    The "Days of Snow Cover per Calendar year" array provides information of the number of days with snow cover for each pixel in the study area,
    aggregated by calendar year for the years 2018, 2019, 2020, 2021, 2022, 2023 and 2024. 

    2. Digital Elevation Model (DEM):
    The DEM is derived from the french elevation model with a spatial resolution of 1 meter
    (https://developers.google.com/earth-engine/datasets/catalog/IGN_RGE_ALTI_1M_2_0).
    It is being resampled to match the Sentinel-2 Resolution of 20 Meters by averaging the cell values of the 1m native resolution.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data Acquisition and Preprocessing: 
The entire Datacube ("snow_days_per_year.nc") is delivered with the snowline.py script but can also be generated from these two scripts:
    1. cube_generation.py:
    This script generates the raw data cube from Sentinel-2 Images intersecting with the study area and a Cloudcover threshold of 10% using Google EarthEngine.
    Since the Sen2Cor Algorithm performs rather poorly over mountainous terrain, the script does not mask out clouds separately since the NDSI
    performs very well in differentiating between clouds and snow. In a more sophisticated approach a custom cloud masking could be employed.
    
    To keep the Data size manageable the xee extention (https://github.com/google/Xee) is used which does not only allow an immediate export to a local drive 
    but also allows the superior netcdf format aswell as work with the familiar xarray structure.

    The DEM is downloaded from Earthengine as a tif in the common batch.export way.

    2. cube_preprocessing.py
    The raw datacube generated within cube_generation.py now turns the binary integer arrays into continuous 
    variables representing the number of snow-covered days per pixel per calendar year. Since this involves a lot of CPU usage and ram, 
    this step is performed using Dask. Still a minimum of 32 GB of RAM is recommended (tested on Pop!_OS/Ubuntu 22.04) and a multicore Processor is highly recommended.

    Since there are major gaps between different sentinel-2 scenes, a very simple interpolation between these takes place: 
    
    1. Resample to daily resolution:
         1 == Snow,
         0 == No Snow,
         NaN == NoData
    
    2. Rule based gap filling betweeen two observations:
        - If both acquisitions have the same value for a pixel, fill the gap with that pixel value 
        - If they have different values (change in the snowcover extent), split the gap in the middle. 
        The first half of the gap gets the left value while the second half gets the right value. 
        - Gaps before the first/after the last observation remain NaN.
    
    3. Count snow days per year as the number of 1s. Pixels without data stay NaN.

    The final resulting xarray dataset consists of two xarray dataarrays as mentioned above. The Dimensions within these two data arrays differ:
    1. snow_days_per_year: (year, y, x)
    2. dem: (y, x)

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    




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
import os

wbt=whitebox.WhiteboxTools() # init whitebox  (more lightweight than xrspatial and therefore more reproducible)
#----------
#Environment Setup


#setup data dir, root dir, code dir
here = Path(__file__).resolve().parent
root = here.parent

data_dir = root / "data"
outdir   = root / "output"

print("Project root:", root)
made_any = False
if not data_dir.exists():
    data_dir.mkdir(parents=True, exist_ok=True)
    made_any = True
if not outdir.exists():
    outdir.mkdir(parents=True, exist_ok=True)
    made_any = True

print(f"data_dir : {data_dir} ({'created' if made_any and data_dir.exists() else 'ok'})")
print(f"outdir   : {outdir} ({'created' if made_any and outdir.exists() else 'ok'})")

#NOTE: ENSURE THE DATACUBE ("snow_days_per_year.nc") IS INSIDE THE DATA_DIR



#-----------------------------------------------------

ds=xr.open_dataset(data_dir / "snow_days_per_year.nc", engine="netcdf4") 
dem=ds["dem"]
snow_days_per_year=ds["snow_days_per_year"]

#sanityy check on the datacube
print(ds)
print("dem:", dem.dims, dem.sizes)
print("snow:", snow_days_per_year.dims, snow_days_per_year.sizes)
print("coords:", list(ds.coords))

#-----------------------------------------------
mean_days = snow_days_per_year.mean(dim="year") # calcs the avg number of snowcovered days per pixel across all years
std_days = snow_days_per_year.std(dim="year")   # calcs the std dev of snowcovered days per pixel -> Measure of fluctuation
#-----------------------------------------------

#create slope curvature and aspect using whitebox.
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

