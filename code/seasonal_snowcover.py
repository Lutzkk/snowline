#snowline.py
#Python 3.10.2
#author: Lutz Kleemann


"""
Github Link: https://github.com/Lutzkk/snowline 

NOTE: The GitHub Repo contains a yaml to rebuild the environment.
It is HIGHLY recommended to use a clone of the repo instead of the standalone script.
This script follows a notebook-like, narrative structure rather than a modular utility layout. A one script submission prevents a modular approach.

NOTE: 
project_root/
├── code/       all scripts (cube_generation.py, cube_preprocessing.py, seasonal_snowcover.py  <- (main script))
├── data/       consists of datacube (iif datacube is rebuilt, raw data will be stored there)
└── output/     generated results and plots

PLEASE ENSURE THAT THE DATA STRUCTURE IS MAINTAINED AS DESCRIBED ABOVE and ALL SCRIPTS ARE PLACED IN THE CODE DIRECTORY.
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

The script below investigates the dimensions of yearly snowcover in the versoyen basin in the french alps in regards to elevation attributes (elevation, slope, curvature, aspect) 
The Basin has a size of 115 km² and an elevation range from 915 to 3006 m above sea level. It is derived from the HYDROSHEDS Dataset (https://www.hydrosheds.org/).
One of the main reasons to choose this basin is the absence of coniferous forests, which saves one step in the processing chain of masking these areas out.

The Script does not utilize any machine learning or regression techniques and should be treated as a simple exploratory analysis.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Data:
The Datacube ("snow_days_per_year.nc") contains two xarray data arrays:
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
    Since the Sen2Cor Algorithm performs rather poorly over mountainous terrain, the script does not mask out clouds separately.
    However, since the NDSI distinguishes Clouds and Snow really well, only false negatives might be the outcome (no fp). 
    In a more sophisticated approach a custom cloud masking algorithm could be employed but this would result in another script or module import.

    A binary Snowmask is created using the NDSI threshholding Method with a threshhold of 0.42 - Bands 3 and 11 of Sentinel-2 are being used.
    (Threshold value inherited from: https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/ndsi/)

    To keep the Data size manageable the xee extention (https://github.com/google/Xee) is used which does not only allow an immediate export EE objects to a local drive
    but also allows for the usage of the superior netcdf format aswell as work with the familiar xarray structure which circumvents earthengine computation limitations more easily.
    In total 143 binary snow masks are being used.

    The DEM is downloaded from Earthengine as a tif in the common batch.export way and later added to the datacube

    2. cube_preprocessing.py
    The raw datacube generated within cube_generation.py now turns the binary integer arrays into continuous 
    variables representing the number of snow-covered days per pixel per calendar year. 
    
    Since this involves a lot of CPU usage and ram, 
    this step is performed using Dask. Still a minimum of 32 GB of RAM is recommended (tested on Pop!_OS/Ubuntu 22.04) 

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

    The final resulting xarray dataset consists of two xarray dataarrays as mentioned above.
    The Dimensions within these two data arrays differ but they perfectly align spatially:
    1. snow_days_per_year: (year, y, x)
    2. dem: (y, x)


Comment: 
It is important to mention that the snow mask generation process is highly dependent on the resolution and quality of the input data.
Since the input data is derived from Sentinel-2 imagery with very little cloudcover, the timegaps between acquisitions can sometimes span across several weeks.
The "simple" Interpolation between acquisitions is simplifying the actual snow dynamics and may not capture all relevant changes! 
It should be therefore be treated as a proof of concept rather than a real representation of snow dynamics.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
"""

#package imports
from pathlib import Path
import xarray as xr
import xrspatial
from xrspatial import hillshade, slope, aspect, curvature
from xrspatial.classify import reclassify
from xrspatial.zonal import stats
import datashader.transfer_functions as tf
from datashader.colors import Elevation
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MultipleLocator
import pandas as pd
import os

#----------

#Environment Setup

#setup data dir, root dir, code dir
root     = Path(__file__).resolve().parents[1]

data_dir = root / "data"
outdir   = root / "output"

print(f"\nProject directory detected: {root}")
print("The following folders will be created if they don't exist:")
print(f" - {data_dir.name}/")
print(f" - {outdir.name}/")


#NOTE: ENSURE THE DATACUBE ("snow_days_per_year.nc") IS INSIDE THE DATA_DIR AND THE SCRIPT IS IN THE CODES FOLDER.

#-----------------------------------------------------

#open dataset
ds=xr.open_dataset(data_dir / "snow_days_per_year.nc", engine="netcdf4") 

#extract data arrays out of the dataset and ensure float for dem (for xrspatial dem processing)
dem=ds["dem"]
dem32 = dem.astype("float32")

snow_days_per_year=ds["snow_days_per_year"]

#sanity check on the datacube
print(ds)
print("dem:", dem.dims, dem.sizes)
print("snow:", snow_days_per_year.dims, snow_days_per_year.sizes)
print("coords:", list(ds.coords))

#-----------------------------------------------
mean_days = snow_days_per_year.mean(dim="year") 
std_days = snow_days_per_year.std(dim="year")   
#-----------------------------------------------

#calc terrain attributes: hillshade, slope, aspect, curvature using xrspatial
hs = hillshade(dem32, azimuth=315.0, angle_altitude=45.0) # for aesthetics
slope_xrs = slope(dem32, name='slope')
aspect_xrs = aspect(dem32, name='aspect')
curv_xrs  = curvature(dem32, name='curvature')
print("created hillshade, slope, aspect, curvature")

#-----------------------------------------------

#OVERVIEW MAP 
# Extent from raster bounds
xmin, ymin, xmax, ymax = dem32.rio.bounds()
extent = [xmin, xmax, ymin, ymax]


# Using the Elevation Colormap from xrspatial (slightly adapted)
Elevation_own = [ "sandybrown", "limegreen", "green", "green", "darkgreen", "saddlebrown", "gray", "white"]  # noqa: E501
relief_cont = LinearSegmentedColormap.from_list("Relief_cont", Elevation_own, N=512)

#
fig, ax = plt.subplots(figsize=(9, 10), constrained_layout=True)

#Hillshade background
ax.imshow(hs, cmap="gray", extent=extent, origin="lower")

#Relief DEM overlay
im = ax.imshow(dem32, cmap=relief_cont, alpha=0.55,
               extent=extent, origin="lower")

# Continuous colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Elevation (m)")

# Colorbar ticks every 200 m, minor ticks every 100 m
major_step = 200
minor_step = 100
cbar.ax.yaxis.set_major_locator(MultipleLocator(major_step))
cbar.ax.yaxis.set_minor_locator(MultipleLocator(minor_step))
cbar.ax.tick_params(which="both", length=6)
cbar.ax.tick_params(which="minor", length=3)

# ----- Map grid -----
xr = xmax - xmin
yr = ymax - ymin

def pick_step(span, target_lines=12):
    candidates = np.array([1,2,5,10,20,50,100,200,500,1000,2000,5000], dtype=float)
    return float(candidates[np.argmin(np.abs(candidates - span/target_lines))])

x_major = pick_step(xr, 5)
y_major = pick_step(yr, 5)
x_minor = x_major / 2
y_minor = y_major / 2

ax.xaxis.set_major_locator(MultipleLocator(x_major))
ax.yaxis.set_major_locator(MultipleLocator(y_major))
ax.xaxis.set_minor_locator(MultipleLocator(x_minor))
ax.yaxis.set_minor_locator(MultipleLocator(y_minor))

ax.grid(True, which="major", alpha=0.45, linewidth=0.8)
ax.grid(True, which="minor", alpha=0.20, linewidth=0.5)

# Labels & limits
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.savefig(outdir / "dem_relief_hillshade_continuous.png", dpi=300)
print("Overview Map created")
#NOTE: Please check the output directory for the generated map.
#Displaying Graphs in Python Scripts is not good practise. 

#-----------------------------------------------

#Reclassify terrain attributes to discrete classes -> Makes it easier to explore the Impact of terrain attributes on snowcover dynamics.

#slope
bins=[5,10,15,20,25,30,35,40,45,50,55,60,np.inf]
new_values=[1,2,3,4,5,6,7,8,9,10,11,12,13]
slope_class = reclassify(slope_xrs, bins=bins, new_values=new_values, name="slope_class")

slope_class.attrs["class_ranges"] = {
    1: "0–5°",
    2: "5–10°",
    3: "10–15°",
    4: "15–20°",
    5: "20–25°",
    6: "25–30°",
    7: "30–35°",
    8: "35–40°",
    9: "40–45°",
    10: "45–50°",
    11: "50–55°",
    12: "55–60°",
    13: ">60°"
}
print("reclassified slope (0–60° in 5° steps)")

#---

#aspect
#flatmask since aspect does not matter in areas with very little slope (xrspatial has it inherited in the function but its a double check)
flat_mask = (slope_xrs < 1.0) | ~np.isfinite(aspect_xrs)
aspect_masked = aspect_xrs.where(~flat_mask)

#see: https://xarray-spatial.readthedocs.io/en/latest/reference/_autosummary/xrspatial.aspect.aspect.html

bins = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, np.inf]
new_values = [1, 2, 3, 4, 5, 6, 7, 8, 1]  # 1=N, 2=NE, ... 8=NW
aspect_class = reclassify(aspect_masked, bins=bins, new_values=new_values, name="aspect_classes")
aspect_class = aspect_class.where(~flat_mask)

aspect_labels = {1: "N", 2: "NE", 3: "E", 4: "SE", 5: "S", 6: "SW", 7: "W", 8: "NW"}
aspect_class.attrs["labels"] = aspect_labels
print("reclassified aspect (N, NE, E, SE, S, SW, W, NW)")

#---

#curvature

#xrspatial.curvature returnes curvature in m⁻¹,

# classes: 
# curv < -0.05 -> concave
# curv >= -0.05 AND <= 0.05 -> planar
# curv > 0.05 -> convex
bins=[-0.05,0.05, np.inf]
new_values=[1,2,3]
curv_class=reclassify(curv_xrs, bins=bins, new_values=new_values, name="curvature_classes")

labels={1: "concave", 2: "planar", 3: "convex"}
curv_class.attrs["labels"] = labels
print("reclassified curvature (concave, planar, convex)")

#---

#elevation
highest=3551
lowest=813
#create bins in 100m increments
elev_bins=list(range(lowest, highest + 100, 100))
#class id for each bin
new_values=list(range(1, len(elev_bins)+1))

#apply reclassification
elev_class=reclassify(dem32, bins=elev_bins, new_values=new_values, name="elevation_classes")

#labels
labels = {i: f"{start}–{end} m" for i, (start, end) in enumerate(zip(
    [lowest] + elev_bins[:-1], elev_bins
), start=1)}

elev_class.attrs["labels"] = labels
print("reclassified elevation (100m steps)")

#----------------------------------------------

# Plot reclassified terrain attributes

# Helper Function for discrete colormaps
def make_discrete_cmap(cmap_name, n, labels, bad_color=(0,0,0,0)):
    cmap = plt.cm.get_cmap(cmap_name, n)  #quantized
    cmap = cmap.with_extremes(bad=bad_color)
    norm = mcolors.BoundaryNorm(np.arange(0.5, n+1.5, 1), n)
    return cmap, norm

fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)

#Slope
slope_labels = ["0–5°", "5–10°", "10–15°", "15–20°", "20–25°", "25–30°",
                "30–35°", "35–40°", "40–45°", "45–50°", "50–55°", "55–60°", ">60°"]
cmap_slope, norm_slope = make_discrete_cmap("viridis", len(slope_labels), slope_labels)
axes[0,0].imshow(hs, cmap="gray", extent=extent, origin="lower")
im0 = axes[0,0].imshow(slope_class, cmap=cmap_slope, norm=norm_slope,
                       extent=extent, origin="lower", alpha=0.7)
cbar0 = fig.colorbar(im0, ax=axes[0,0], fraction=0.046, pad=0.04,
                     ticks=np.arange(1, len(slope_labels)+1))
cbar0.ax.set_yticklabels(slope_labels)
axes[0,0].set_title("Slope classes")
axes[0,0].axis("off")

#Aspect
aspect_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
cmap_aspect = mcolors.ListedColormap([
    "#FF0000", "#FF7F00", "#FFFF00", "#7FFF00",
    "#00FF00", "#00FFFF", "#0000FF", "#8B00FF"
])
cmap_aspect = cmap_aspect.with_extremes(bad=(0,0,0,0))
norm_aspect = mcolors.BoundaryNorm(np.arange(0.5, len(aspect_labels)+1.5, 1),
                                   cmap_aspect.N)
axes[0,1].imshow(hs, cmap="gray", extent=extent, origin="lower")
im1 = axes[0,1].imshow(aspect_class, cmap=cmap_aspect, norm=norm_aspect,
                       extent=extent, origin="lower", alpha=0.7)
cbar1 = fig.colorbar(im1, ax=axes[0,1], fraction=0.046, pad=0.04,
                     ticks=np.arange(1, len(aspect_labels)+1))
cbar1.ax.set_yticklabels(aspect_labels)
axes[0,1].set_title("Aspect classes")
axes[0,1].axis("off")

#Curvature
curv_labels = ["concave", "planar", "convex"]
cmap_curv = mcolors.ListedColormap(["#2166AC", "#FFFFFF", "#B2182B"])
cmap_curv = cmap_curv.with_extremes(bad=(0,0,0,0))
norm_curv = mcolors.BoundaryNorm(np.arange(0.5, len(curv_labels)+1.5, 1), cmap_curv.N)
axes[1,0].imshow(hs, cmap="gray", extent=extent, origin="lower")
im2 = axes[1,0].imshow(curv_class, cmap=cmap_curv, norm=norm_curv,
                       extent=extent, origin="lower", alpha=0.7)
cbar2 = fig.colorbar(im2, ax=axes[1,0], fraction=0.046, pad=0.04,
                     ticks=np.arange(1, len(curv_labels)+1))
cbar2.ax.set_yticklabels(curv_labels)
axes[1,0].set_title("Curvature classes")
axes[1,0].axis("off")

#Elevation
elev_label_map = elev_class.attrs["labels"]  
elev_labels = [elev_label_map[i] for i in sorted(elev_label_map.keys())]

relief_cont = LinearSegmentedColormap.from_list("Relief_cont", Elevation_own, N=30) #overwrite N
cmap_elev, norm_elev = make_discrete_cmap(relief_cont, len(elev_labels), elev_labels)
axes[1,1].imshow(hs, cmap="gray", extent=extent, origin="lower")
im3 = axes[1,1].imshow(elev_class, cmap=cmap_elev, norm=norm_elev,
                       extent=extent, origin="lower", alpha=0.7)
cbar3 = fig.colorbar(im3, ax=axes[1,1], fraction=0.046, pad=0.04,
                     ticks=np.arange(1, len(elev_labels)+1))
cbar3.ax.set_yticklabels(elev_labels)
axes[1,1].set_title("Elevation classes")
axes[1,1].axis("off")

plt.savefig(outdir / "terrain_classes_2x2_discrete.png", dpi=300)

#---

#PLOT HISTOGRAMS OF TERRAIN ATTRIBUTES TO CHECK DATA DISTRIBUTION:

#countt values

slope_counts = slope_class.values.ravel()
slope_counts = slope_counts[np.isfinite(slope_counts)]
unique_slope, counts_slope = np.unique(slope_counts.astype(int), return_counts=True)

aspect_counts = aspect_class.values.ravel()
aspect_counts = aspect_counts[np.isfinite(aspect_counts)]
unique_aspect, counts_aspect = np.unique(aspect_counts.astype(int), return_counts=True)

curv_counts = curv_class.values.ravel()
curv_counts = curv_counts[np.isfinite(curv_counts)]
unique_curv, counts_curv = np.unique(curv_counts.astype(int), return_counts=True)

elev_counts = elev_class.values.ravel()
elev_counts = elev_counts[np.isfinite(elev_counts)]
unique_elev, counts_elev = np.unique(elev_counts.astype(int), return_counts=True)

#---

#Plot
fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
fig.suptitle("Amount of Pixels per Terrain Class", fontsize=16)

#slope
axes[0,0].bar([slope_labels[i-1] for i in unique_slope], counts_slope)
axes[0,0].set_title("Histogram – Slope classes")
axes[0,0].set_ylabel("Pixel count")
axes[0,0].tick_params(axis="x", rotation=45)

#aspect
axes[0,1].bar([aspect_labels[i-1] for i in unique_aspect], counts_aspect)
axes[0,1].set_title("Histogram – Aspect classes")
axes[0,1].set_ylabel("Pixel count")
axes[0,1].tick_params(axis="x", rotation=45)

#curvature
axes[1,0].bar([curv_labels[i-1] for i in unique_curv], counts_curv)
axes[1,0].set_title("Histogram – Curvature classes")
axes[1,0].set_ylabel("Pixel count")
axes[1,0].tick_params(axis="x", rotation=45)

#elevation
axes[1,1].bar(
    [elev_class.attrs["labels"].get(int(i), str(int(i))) for i in unique_elev],
    counts_elev
)
axes[1,1].set_title("Histogram – Elevation classes")
axes[1,1].set_ylabel("Pixel count")
axes[1,1].tick_params(axis="x", rotation=45)

plt.savefig(outdir / "terrain_class_histograms_2x2_counts.png", dpi=300)
print("Saved histogram panel")

#----------------------------------------------
# We now reclassified all terrain attributes into discrete classes and visualized them.
# Now we can check out the spatial distribution + variation of snow cover days in the basin.


## Plotting mean snow days


#Since the resolution and extent of all datasets is the same, it doesnt matter from which array we take the extent metrics
xmin, ymin, xmax, ymax = hs.rio.bounds()
extent = [xmin, xmax, ymin, ymax]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
axes[0].imshow(hs, cmap="gray", extent=extent, origin= "lower")

im0 = axes[0].imshow(mean_days, extent=extent, cmap="viridis", origin="lower",
                     alpha=0.7)
axes[0].set_title("Mean snow-covered days per year (averaged from 2018 to 2024)")
axes[0].axis("off")
cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.set_label("days")


axes[1].imshow(hs, cmap="gray", origin= "lower",
               extent=extent)
im1 = axes[1].imshow(std_days, extent=extent, origin="lower",
                     alpha=0.7)
axes[1].set_title("Interannual std. of snow-covered days")
axes[1].axis("off")
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label("days")


plt.savefig(outdir / "snowcoverdays_mean_std.png", dpi=400)

print("SnowMap created")

"""
We can already see from first glance, that the snowcover seems to be the longest per year in the higher altitudes in the north and east.
The standard deviation is also lower in these regions indicating a very consistent snow cover pattern in these regions.
At the same time areas located at steep slopes tend to have a high standard deviation indicating higher interannual variability of snow persistence there.
"""


#------------------------------------------------------------

#xrspatial zonalstats
zone_layers={
    "Elevation": elev_class,
    "Aspect": aspect_class,
    "Curvature": curv_class,
    "Slope": slope_class
}

all_results=[]

for name, zones in zone_layers.items():
    stat = stats(zones=zones, values=mean_days)
    stat["zone_type"] = name
    
    #attach readable labels from attrs
    labels_dict = zones.attrs.get("labels") or zones.attrs.get("class_ranges")
    if labels_dict:
        stat["zone_label"] = stat["zone"].map(labels_dict)
    else:
        stat["zone_label"] = stat["zone"]  #fallback
    
    all_results.append(stat)

df_all = pd.concat(all_results, ignore_index=True)
df_all.to_csv(outdir / "snowcoverdays_stats.csv", index=False)

#----

#visualize results
orders={
    "Aspect":   aspect_labels,
    "Curvature": curv_labels,
    "Slope":    slope_labels,
    "Elevation": elev_labels
}
fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
fig.suptitle("Zonal Statistics: Mean Snow-Covered Days per Terrain Attribute Class", fontsize=16)
axes = axes.ravel()

for i, zone_type in enumerate(["Slope","Aspect","Curvature","Elevation"]):
    sub = df_all[df_all["zone_type"] == zone_type].copy()

    # Use label if present; fallback to numeric zone id
    label_col = "zone_label" if "zone_label" in sub.columns else "zone"
    
    # Sort per type
    if zone_type in orders:
        cat_order = orders[zone_type]
        sub[label_col] = sub[label_col].astype("category")
        sub[label_col] = sub[label_col].cat.set_categories(cat_order, ordered=True)
        sub = sub.sort_values(label_col)
    else:
        # Slope/Elev: sort by numeric zone id
        sub = sub.sort_values("zone")

    # Plot bars with error bars (std)
    axes[i].bar(sub[label_col].astype(str), sub["mean"], yerr=sub["std"], capsize=3)
    axes[i].set_title(f"{zone_type}: mean snow-covered days")
    axes[i].set_ylabel("days")
    axes[i].set_xlabel("")
    axes[i].tick_params(axis="x", rotation=45)

plt.savefig(outdir / "zonal_means_4x_bars.png", dpi=300)

"""

"""
