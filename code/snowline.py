#snowline.py
#Python 3.10.2
#author: Lutz Kleemann


"""
Github Link: https://github.com/Lutzkk/snowline 

NOTE: The GitHub Repo contains a yaml to rebuild the environment.
Its HIGHLY recommended to use a clone of the repo instead of the standalone script.
This script follows a notebook-like, narrative structure rather than a modular utility layout. A one script submission prevents a modular approach.

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

    A binary Snowmask is created using the NDSI threshholding Method with a threshhold of 0.42 - Bands 3 and 11 of Sentinel-2 are being used.

    To keep the Data size manageable the xee extention (https://github.com/google/Xee) is used which does not only allow an immediate export to a local drive
    but also allows for the superior netcdf format aswell as work with the familiar xarray structure which also circcumvents earthengine computation limitations more easily.

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
import xrspatial
from xrspatial import hillshade, slope, aspect, curvature
from xrspatial.classify import reclassify
from xrspatial.zonal import stats
import datashader.transfer_functions as tf
from datashader.colors import Elevation
import rioxarray as rxr
import whitebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.ticker import MultipleLocator
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
dem32 = dem.astype("float32")
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

#calc terrain attributees
hs = hillshade(dem32, azimuth=315.0, angle_altitude=45.0)
slope_xrs = slope(dem32, name='slope')
aspect_xrs = aspect(dem32, name='aspect')
aspect_xrs_exp=aspect_xrs.rio.write_crs("EPSG:32631")
aspect_xrs_exp.rio.to_raster(root / "asepct_xrspatial.tif")
curv_xrs  = curvature(dem32, name='curvature')
print("created hillshade, slope, aspect, curvature")

#-----------------------------------------------

#OVERVIEW MAP 
# Prepare DEM & hillshade



# Extent from raster bounds
xmin, ymin, xmax, ymax = dem32.rio.bounds()
extent = [xmin, xmax, ymin, ymax]

# Stretch hillshade for better contrast
hs_vmin = np.nanpercentile(hs, 2)
hs_vmax = np.nanpercentile(hs, 98)

# Continuous Relief colormap
relief_cont = LinearSegmentedColormap.from_list("Relief_cont", Elevation, N=512)

# ----- Figure -----
fig, ax = plt.subplots(figsize=(9, 10), constrained_layout=True)

# Hillshade background
ax.imshow(hs, cmap="gray", vmin=hs_vmin, vmax=hs_vmax,
          extent=extent, origin="lower")

# Relief DEM overlay (continuous)
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
#reclassify terrain attributes 

#---RECLASSIFY

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
#flatmask since aspect does not matter in areas with very little slope
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
#while in thiis case the values span around +/-0.4
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
# --- helper for discrete colormaps ---
def make_discrete_cmap(cmap_name, n, labels, bad_color=(0,0,0,0)):
    cmap = plt.cm.get_cmap(cmap_name, n)  # quantized
    cmap = cmap.with_extremes(bad=bad_color)
    norm = mcolors.BoundaryNorm(np.arange(0.5, n+1.5, 1), n)
    return cmap, norm

fig, axes = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)

# === 1. Slope ===
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

# === 2. Aspect ===
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

# === 3. Curvature ===
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

# === 4. Elevation ===
elev_labels = [f"{int(lo)}–{int(hi)} m" for lo, hi in zip(elev_bins[:-1], elev_bins[1:])]
cmap_elev, norm_elev = make_discrete_cmap("terrain", len(elev_labels), elev_labels)
axes[1,1].imshow(hs, cmap="gray", extent=extent, origin="lower")
im3 = axes[1,1].imshow(elev_class, cmap=cmap_elev, norm=norm_elev,
                       extent=extent, origin="lower", alpha=0.7)
cbar3 = fig.colorbar(im3, ax=axes[1,1], fraction=0.046, pad=0.04,
                     ticks=np.arange(1, len(elev_labels)+1))
cbar3.ax.set_yticklabels(elev_labels)
axes[1,1].set_title("Elevation classes")
axes[1,1].axis("off")

plt.savefig(outdir / "terrain_classes_2x2_discrete.png", dpi=300)



#----------------------------------------------

#at first we look at the general spatial distribution of snow days and its respective standard deviation
## Plotting mean snow days


#since the resolution and extent of all datasets is the same, we can take the extent from one random dataset
xmin, ymin, xmax, ymax = hs.rio.bounds()
extent = [xmin, xmax, ymin, ymax]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
axes[0].imshow(hs, cmap="gray", vmin=0, vmax=32767, extent=extent, origin= "lower")
im0 = axes[0].imshow(mean_days, extent=extent, cmap="viridis", origin="lower",
                     alpha=0.7)
axes[0].set_title("Mean snow-covered days (multi-year)")
axes[0].axis("off")
cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.set_label("days")

axes[1].imshow(hs, cmap="gray", vmin=0, vmax=32767, origin= "lower",
               extent=extent)
im1 = axes[1].imshow(std_days, extent=extent, origin="lower",
                     alpha=0.7)
axes[1].set_title("Interannual std. of snow-covered days")
axes[1].axis("off")
cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label("days")


plt.savefig(outdir / "snowcoverdays_mean_std.png", dpi=400)

"""
We can already see from first gleance, that the snowcover seems to be the longest per year in the higher altitudes in the north and east. 
"""
print("SnowMap created")
#------------------------------------------------------------


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
    
    # attach readable labels from attrs
    labels_dict = zones.attrs.get("labels") or zones.attrs.get("class_ranges")
    if labels_dict:
        stat["zone_label"] = stat["zone"].map(labels_dict)
    else:
        stat["zone_label"] = stat["zone"]  # fallback to numeric ID
    
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
