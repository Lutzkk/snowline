## SNOWSTATS  

This repository was created as part of the **Advanced Spatial Python** course at the *Earth Observation Research Cluster*.  
It is written in **Python 3.10.2** and explores the relationship between **terrain attributes** and **snow-cover duration** in the Versoyen Basin (French Alps).

---

## Data

The analysis is based on a [NetCDF datacube](https://github.com/pydata/xarray) containing:

- **Snow-cover duration**  
  Number of days with snow cover per pixel, aggregated annually for 2018–2024.  
- **Digital Elevation Model (DEM)**  
  Derived from [IGN RGE ALTI 1m](https://developers.google.com/earth-engine/datasets/catalog/IGN_RGE_ALTI_1M_2_0), resampled to 20 m to match Sentinel-2.



---

## Preprocessing

The datacube is generated in two steps:

1. **`cube_generation.py`**  
   - Collects Sentinel-2 imagery (≤ 10% cloud cover) using Google Earth Engine.  
   - Applies NDSI thresholding (`B3`, `B11`, threshold = 0.42) to create binary snow masks.  
   - Exports to NetCDF using [Xee](https://github.com/google/Xee).

2. **`cube_preprocessing.py`**  
   - Interpolates gaps between acquisitions (rule-based).  
   - Aggregates daily snow masks to annual counts of snow-covered days.  
   - Adds DEM layer and prepares final datacube.

---

## Analysis 
The analysis links snow-cover duration with topographic features derived from the DDEM. Terrain variables such as elevation, slope, aspect and curvature are extracted and compared against the number of snow-covered days per year.  
This is done by generating classes for the different terrain features and then performing zonal statistics.  
**No Regression _OR_ Machine Learning is being used**.

[terrain features](output/terrain_classes_2x2_discrete.png)

## Caveats

- Interpolation between acquisitions simplifies snow dynamics and missees short-term events.  
- Sentinel-2 cloud masking is limited in complex terrain.Misclassification is possible.  
- Results should be considered **exploratory** rather than definitive.  

---