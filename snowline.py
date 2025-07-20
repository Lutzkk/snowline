#snowline.py

#-----------------------

# This File is a standalone script that calculates the snowline altitude in an alpine catchment over multiple years
# It uses Cloud Masked Landsat 8/9 data to determine the change in snowline alt over the last 5 years 
# The Timespan of 5 years is chosen for multiple reasons: 
# 1: It keeps the data size manageable and easily reproducible without needing to fallback on EE (which wasnt part of the course)
# 2: Its more of a proof of concept than a full analysis, but can easily be extended to more years by changing the parameters

#-----------------------

# IMPORTANT: This Script utilizes Earth Observation Data Access Gateway (EODAG) instead of EE to preprocess data. 
# If EODAG is not installed, please install it and set up a
# configuration yaml as described in the EODAG documentation.
# https://eodag.readthedocs.io/en/stable/getting_started_guide/install.html

#-----------------------

# The Code aswell as a conda yaml can be found on the following GitHub repo:
# https://github.com/Lutzkk/snowline



# library imports
import os 
from pathlib import Path # Important for path handling indepondent of OS (code written on Ubuntu 22.04 system)
import ee
import pandas as pd
import xarray as xr
import geopandas as gpd
import numpy as np
#------------------------

# Environment Setup

project_dir = Path(__file__).resolve().parent
data_dir = project_dir / "data"
utils_dir = project_dir / "utils"
output_dir = project_dir / "output"

print(f"\nProject directory detected: {project_dir}")
print("The following folders will be created if they don't exist:")
print(f" - {data_dir.name}/")
print(f" - {utils_dir.name}/")
print(f" - {output_dir.name}/")


confirm = input("\nDo you want to proceed with creating these folders here? (y/n): ").strip().lower()

if confirm == "y":
    for folder in [data_dir, utils_dir, output_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    print("Folders created/already exist.")
else:
    print("No folders were created.")


#------------------------
#----PREPROCESSING----
#------------------------

#initialize earthengine api
ee.Authenticate()
ee.Initialize(project="ee-lutz-training", opt_url='https://earthengine-highvolume.googleapis.com') #highvolume url is necesarry for the xee (xarray earthengine extension) to work reliably


def scale_bands(img):
    """
    Scale S2 bands to reflectance floats (0-1 range) and reproject to 20 m.
    Ensures consistent resolution for all used bands (including SWIR).
    """
    bands_to_scale = ['B2', 'B3', 'B4', 'B11']
    
    # Define target projection using B11 (which is native 20 m)
    target_proj = img.select("B11").projection()
    
    # Scale bands and reproject them all to 20 m
    scaled = img.select(bands_to_scale) \
        .divide(10000) \
        .resample("bilinear") \
        .reproject(crs=target_proj)
    
    return img.addBands(scaled, overwrite=True)


def add_ndsi(img):
    """
    Calculate the Normalized Difference Snow Index (NDSI) for the image
    """
    ndsi = img.normalizedDifference(['B3', 'B11']).rename('NDSI')
    return img.addBands(ndsi)

def clip_to_aoi(image):
    """
    Clip sentinel-2 scene to the area of interest (aoi)
    """
    return image.clip(aoi)

def binarize_ndsi(img):
    """
    Binarize NDSI band to create a binary snow mask
    """
    snow_mask = img.select('NDSI').gt(0.42).rename('SnowMask')
    return img.addBands(snow_mask)

#------------------------

aoi_gdf = gpd.read_file("./data/basin.geojson") 

#Convert the geometry to ee.Geometry
aoi_geojson = aoi_gdf.geometry.values[0].__geo_interface__

#Create ee.Geometry
aoi = ee.Geometry(aoi_geojson)

#------------------------
#DEM ACQUISITION
#------------------------
srtm = ee.Image("CGIAR/SRTM90_V4").clip(aoi)

task = ee.batch.Export.image.toDrive(
    image=srtm,
    description='SRTM_Export',
    folder='EarthEngineExports',  # Google Drive folder name
    fileNamePrefix='SRTM_dem',
    region=aoi,
    scale=90,
    fileFormat='GeoTIFF'
)
task.start()
#------------------------
#BINARY SNOW MASK ACQUISITION
#------------------------
#query Sentinel-2 Scenes in timerange 2015-2025 with cloudcover < 5%, select bands B2,B3,B4 (for visual inspection in a testing env) and B11 (for NDSI calc)
# Apply prepocessing functions (band scaling, NDSI calculation, binarization)
s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterDate("2017-06-01", "2025-06-01") \
    .select(["B4", "B3", "B2", "B11"]) \
    .filterBounds(aoi) \
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10)) \
    .map(clip_to_aoi) \
    .map(scale_bands) \
    .map(add_ndsi) \
    .map(binarize_ndsi)


#------Conversion to xarray Dataset AND Download as NetCDF------


# Convert the ee.ImageCollection to an xarray Dataset using xee
proj_info = s2.first().select("SnowMask").projection().getInfo()

crs= proj_info['crs']
scale= proj_info['transform'][0]  # Assuming square pixels, scale is the first element of the transform

# NOTICE: Xarray cannot display the cube inside a python script (only in a notebook)
# However, to this date (2025-06-18) xee has a formatting inconsistency that does not match the NETCDF (and most other geospatial format) conventions
# Therefore, we need to transpose the dataset to match the expected format. Note that the issue is currently being worked on and might change in the future.
# For more info see: 
# https://github.com/google/Xee/discussions/196
# https://github.com/google/Xee/issues/171
# https://github.com/google/Xee/issues/230

snowmask=xr.open_dataset(
    s2.select("SnowMask"),
    engine="ee",
    geometry=aoi,
    crs=crs,
    scale=scale, 
).transpose("time", "Y", "X")

print(f"Current Dimension Order: \n {snowmask.dims}")


print(f"The Datacube contains {snowmask.sizes['time']} time steps and {snowmask.sizes['X']}*{snowmask.sizes['Y']} pixels.")


timestamps = snowmask["time"].values
formatted = "\n".join(str(ts) for ts in timestamps)
print(f"The following timestamps are available:\n{formatted}")


# Convert timestamps to pandas datetime (xarray stores them as np.datetime64)
timestamps = pd.to_datetime(snowmask['time'].values)

# Normalize to dates (strip time)
dates_only = timestamps.normalize()

# Get unique dates and their first occurrence
_, unique_indices = np.unique(dates_only, return_index=True)

# Select only those indices in xarray
snowmask_unique = snowmask.isel(time=unique_indices)
print(s2.first().select("SnowMask").projection().getInfo())

# Confirm result
print(f"Filtered to {snowmask_unique.sizes['time']} unique timestamps.")


#create output directory to hold temporary netcdf files
chunk_dir = data_dir / "chunks"
chunk_dir.mkdir(exist_ok=True)

print("Starting export of dataset from EE to NC for local processing...")

for i, ts in enumerate(snowmask_unique.time.values):
    ts_dt = pd.to_datetime(ts)
    date_str = ts_dt.strftime("%Y-%m-%d")
    file_path = chunk_dir / f"snowmask_{date_str}.nc"
    
    # Export one chunk at a time
    snowmask_unique.isel(time=i).compute().to_netcdf(file_path)
    print(f"Exported {file_path.name}")

print("All files exported successfully.")

chunk_dir = data_dir / "chunks"

nc_files=sorted(chunk_dir.glob("snowmask_*.nc")) #note that glob is not from the glob package but from pathlib.Path.glob

datasets=[]
for nc_file in nc_files:
    ds = xr.open_dataset(nc_file)
    datasets.append(ds)

#concatenate along the time dimension
combined=xr.concat(datasets, dim="time").rename({"X": "lon", "Y": "lat"}).sortby("time")

#save as datacube
combined.to_netcdf(output_dir / "Snowmask_datacube.nc")