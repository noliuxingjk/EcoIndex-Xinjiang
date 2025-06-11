import os
import numpy as np
import rasterio
import rasterio.mask
import geopandas as gpd
from tqdm import tqdm

# ==========================================
# Define data directories (customize here)
# ==========================================
ndvi_dir = r"D:\your_project\data\NDVI_cleaned"
wue_dir = r"D:\your_project\data\WUE_cleaned"
output_dir = r"D:\your_project\results\quadrant_classification"
shapefile_path = r"D:\your_project\shapefiles\region_boundary.shp"

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# Load study area shapefile
# ==========================================
gdf = gpd.read_file(shapefile_path)
geoms = gdf.geometry.values

# ==========================================
# Define quadrant classification logic
# ==========================================
def classify_quadrants(delta_wue, delta_ndvi):
    """
    Classify pixel-wise changes into 4 quadrants:
        I   (+ΔWUE, +ΔNDVI)
        II  (-ΔWUE, +ΔNDVI)
        III (-ΔWUE, -ΔNDVI)
        IV  (+ΔWUE, -ΔNDVI)
    """
    quadrant = np.full(delta_wue.shape, 0, dtype=np.uint8)
    quadrant[(delta_wue > 0) & (delta_ndvi > 0)] = 1  # Quadrant I
    quadrant[(delta_wue < 0) & (delta_ndvi > 0)] = 2  # Quadrant II
    quadrant[(delta_wue < 0) & (delta_ndvi < 0)] = 3  # Quadrant III
    quadrant[(delta_wue > 0) & (delta_ndvi < 0)] = 4  # Quadrant IV
    return quadrant

# ==========================================
# Process interannual changes and classify
# ==========================================
years = list(range(2000, 2023))  # Exclude final year to compare with next

for year in tqdm(years, desc="Quadrant classification"):
    next_year = year + 1

    ndvi_path1 = os.path.join(ndvi_dir, f"{year}_NDVI_cleaned.tif")
    ndvi_path2 = os.path.join(ndvi_dir, f"{next_year}_NDVI_cleaned.tif")
    wue_path1 = os.path.join(wue_dir, f"{year}_WUE_cleaned.tif")
    wue_path2 = os.path.join(wue_dir, f"{next_year}_WUE_cleaned.tif")

    with rasterio.open(ndvi_path1) as src1, rasterio.open(ndvi_path2) as src2:
        ndvi1, _ = rasterio.mask.mask(src1, geoms, crop=False)
        ndvi2, _ = rasterio.mask.mask(src2, geoms, crop=False)
        meta = src1.meta.copy()

    with rasterio.open(wue_path1) as src1, rasterio.open(wue_path2) as src2:
        wue1, _ = rasterio.mask.mask(src1, geoms, crop=False)
        wue2, _ = rasterio.mask.mask(src2, geoms, crop=False)

    # Calculate yearly differences
    delta_ndvi = ndvi2[0] - ndvi1[0]
    delta_wue = wue2[0] - wue1[0]

    # Apply valid pixel mask
    valid_mask = (~np.isnan(delta_ndvi)) & (~np.isnan(delta_wue))
    delta_ndvi[~valid_mask] = np.nan
    delta_wue[~valid_mask] = np.nan

    # Perform quadrant classification
    quadrant_map = classify_quadrants(delta_wue, delta_ndvi)

    # Save output raster
    meta.update({
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "nodata": 0
    })

    save_path = os.path.join(output_dir, f"{next_year}_Quadrant.tif")
    with rasterio.open(save_path, "w", **meta) as dst:
        dst.write(quadrant_map, 1)

print("✅ All quadrant classification maps generated successfully.")
