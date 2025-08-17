#cube_generation.py
#python 3.10.2

#-----------------------

"""
This Script utilizes Google Earthengine + Xarray to generate a datacube consisting of snow cover information and a DEM for the Versoyen Basin  from the timespan 2018-2024.
"""


# The Code aswell as a conda yaml can be found on the following GitHub repo:
# https://github.com/Lutzkk/snowline



# library imports
import os 
from pathlib import Path # Important for path handling indepondent of OS 
import ee
import pandas as pd
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import numpy as np
from rasterio.enums import Resampling
import shutil

#------------------------

# Environment Setup

project_dir = Path(__file__).resolve().parents[1]
data_dir = project_dir / "data"
output_dir = project_dir / "output"

print(f"\nProject directory detected: {project_dir}")
print("The following folders will be created if they don't exist:")
print(f" - {data_dir.name}/")
print(f" - {output_dir.name}/")


for folder in [data_dir, output_dir]:
    folder.mkdir(parents=True, exist_ok=True)


#------------------------
#----PREPROCESSING----
#------------------------

#initialize earthengine api
ee.Authenticate()
#change project to your own 
ee.Initialize(project="ee-vorarlberg", opt_url='https://earthengine-highvolume.googleapis.com') #highvolume url is necesarry for the xee (xarray earthengine extension) to work reliably


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

aoi_gdf = gpd.read_file("./data/basin.geojson").to_crs(epsg=4326) 

#Convert the geometry to ee.Geometry
aoi_geojson = aoi_gdf.geometry.values[0].__geo_interface__

#Create ee.Geometry
aoi = ee.Geometry(aoi_geojson)

#------------------------
#DEM ACQUISITION
#------------------------
ds = (
    ee.ImageCollection("IGN/RGE_ALTI/1M/2_0")
    .select("MNT")
    .map(lambda img: img.clip(aoi).rename("elevation"))
    .mosaic()
)

task = ee.batch.Export.image.toDrive(
    image=ds,
    description='RGE_ALTI_mosaic_export',
    folder='earthengine_exports',     # Folder Google Drive
    fileNamePrefix='rge_alti_mosaic', 
    region=aoi,
    fileFormat='GeoTIFF'
)

task.start() #download from drive after task (approx 2 mins)


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

print(f"Filtered to {snowmask_unique.sizes['time']} unique timestamps.")

#------------------------
#EXPORT
#------------------------

chunk_dir = data_dir / "chunks"
chunk_dir.mkdir(exist_ok=True)

print("Starting export of dataset from EE to NC for local processing...")

# Export one chunk at a time as DataArray with 'time' dimension preserved
for i, ts in enumerate(snowmask_unique.time.values):
    ts_dt = pd.to_datetime(ts)
    date_str = ts_dt.strftime("%Y-%m-%d")
    file_path = chunk_dir / f"snowmask_{date_str}.nc"
    
    snowmask_da = snowmask_unique["SnowMask"].isel(time=i).compute()
    

    #Retain time dimension (otherwise it will be a 2D slice)
    snowmask_da = snowmask_da.expand_dims(time=[ts_dt])
    
    snowmask_da.to_netcdf(file_path)
    print(f"Exported {file_path.name}")

print("All files exported successfully.")

#-------------------------
#DATACUBE CREATION
#-------------------------
nc_files = sorted(chunk_dir.glob("snowmask_*.nc")) #glob from the pathlib module (not glob module) -> ensures compatibility with Path objects
arrays = [xr.open_dataarray(nc_file) for nc_file in nc_files] #Loop through filelist and open them as standalone datta arrays

# Concatenate along the time dimension and rename coordinates
combined = xr.concat(arrays, dim="time").rename({"X": "lon", "Y": "lat"}) #concat along the time dimension to form a datacube

#sort by time to ensure correct order
combined = combined.sortby("time") 

# Convert bounds to tuple for compatibility with xarray
# Note: This is a workaround for the xarray issue where bounds are stored as lists. (XEE unwanted interaction)
# See: https://discourse.pangeo.io/t/xarray-to-netcdf-valueerror-the-truth-value-of-an-array-with-more-than-one-element-is-ambiguous-use-a-any-or-a-all/4785/5
combined.attrs["bounds"]=tuple(combined.attrs["bounds"]) 
combined.to_netcdf(data_dir / "Snowmask_datacube.nc")

# align names & grids before merge 



# combined dims are ("time", "lon/lat" or "X/Y") → make them ("time","y","x")
if {"lon","lat"}.issubset(combined.dims):
    sm = combined.rename({"lon":"x","lat":"y"}).transpose("time","y","x")
elif {"X","Y"}.issubset(combined.dims):
    sm = combined.rename({"X":"x","Y":"y"}).transpose("time","y","x")
else:
    raise ValueError(f"Unexpected dims for combined: {combined.dims}")

# Tell rioxarray the spatial dims + CRS (UTM 31N)
sm = sm.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
sm = sm.rio.write_crs("EPSG:32631", inplace=False)

# Open DEM (the GeoTIFF you exported) and reproject to match the snowmask grid
dem_path = data_dir / "rge_alti_mosaic.tif"


dem = rxr.open_rasterio(dem_path).squeeze().rename("dem")
print(dem)
ref = sm.isel(time=0)
dem_on_sm = dem.rio.reproject_match(ref, resampling=Resampling.average).astype("float32")

# sanityy prints
print("SM:", sm.sizes, "CRS:", sm.rio.crs.to_string())
print("DEM on SM:", dem_on_sm.sizes, "CRS:", dem_on_sm.rio.crs.to_string())


#-------------------------
#DATACUBE MERGE DEM + SNOWMASK
#NOTE: The following metadata munging is based on agreeing the netcdf conventions,
#especially this section: https://cfconventions.org/cf-conventions/cf-conventions.html#cell-boundaries

#without respect to the conventions the datacube is read only and cant be written to a drive.
#-------------------------


#cast mask to compact dtype
snowmask_u8 = sm.astype("uint8")  

#build xarray dataset
ds_out = xr.Dataset(
    data_vars=dict(
        snowmask=(("time","y","x"), snowmask_u8.data, {"long_name":"Snow mask (0/1)", "grid_mapping":"spatial_ref"}),
        dem=(("y","x"), dem_on_sm.data, {"long_name":"Elevation", "units":"m", "grid_mapping":"spatial_ref"}),
    ),
    coords=dict(
        time=sm["time"],
        y=sm["y"], x=sm["x"],
        spatial_ref=xr.DataArray(sm.rio.crs.to_wkt()), #grid mapping variable
    ),
    attrs={"Conventions":"CF-1.9"},
)

#coord metadata
ds_out["x"].attrs.update(standard_name="projection_x_coordinate", units="m")
ds_out["y"].attrs.update(standard_name="projection_y_coordinate", units="m")

# drop array-valued 'bounds' attrs if present (CF writer quirk)
for name in ds_out.variables:
    if "bounds" in ds_out[name].attrs and not isinstance(ds_out[name].attrs["bounds"], str):
        ds_out[name].attrs.pop("bounds")

#write with compression
out_nc = data_dir / "elevation_snowmask_cube.nc"
encoding = {
    "snowmask": {"zlib": True, "complevel": 4, "dtype": "uint8"},
    "dem": {"zlib": True, "complevel": 4, "dtype": "float32"},
    "x": {"zlib": True, "complevel": 4},
    "y": {"zlib": True, "complevel": 4},
    "time": {"zlib": True, "complevel": 4},
    "spatial_ref": {"zlib": True, "complevel": 4},
}

#save as netcdf
ds_out.to_netcdf(out_nc, engine="netcdf4", encoding=encoding)
print("Wrote:", out_nc)


#-------------------------
#CLEANUP
#-------------------------

#remove chunks
if chunk_dir.exists() and chunk_dir.is_dir():
    print(f"Removing temporary folder: {chunk_dir}")
    #shutil.rmtree(chunk_dir) #THIS LINE REMOVE THE CHUNK DIRECTORY!! UNCOMMENT AND RUN BUT BE ABSOLUTELY SURE ITS POINTING TOO THE CHUNK FOLDER
    #print("Chunks folder removed.")
else:
    print("No chunk folder found, nothing to clean up.")
