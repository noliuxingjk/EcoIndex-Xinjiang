import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from tqdm import tqdm

# ==========================================
# Define input/output paths (customize here)
# ==========================================
ndvi_dir = r"D:\your_project\data\NDVI_cleaned"
wue_dir = r"D:\your_project\data\WUE_cleaned"
output_dir = r"D:\your_project\results\ESI"
shapefile_path = r"D:\your_project\shapefiles\study_region.shp"

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# Load shapefile for masking
# ==========================================
shapefile = gpd.read_file(shapefile_path)

# ==========================================
# Collect annual NDVI and WUE data
# ==========================================
years = list(range(2000, 2024))
ndvi_stack = []
wue_stack = []

for year in tqdm(years, desc="Loading NDVI and WUE"):
    ndvi_path = os.path.join(ndvi_dir, f"{year}_NDVI_cleaned.tif")
    wue_path = os.path.join(wue_dir, f"{year}_WUE_cleaned.tif")

    with rasterio.open(ndvi_path) as src_ndvi:
        ndvi, _ = mask(src_ndvi, shapefile.geometry, crop=False)
        ndvi = ndvi[0]
        ndvi_meta = src_ndvi.meta.copy()

    with rasterio.open(wue_path) as src_wue:
        wue, _ = mask(src_wue, shapefile.geometry, crop=False)
        wue = wue[0]

    ndvi_stack.append(ndvi)
    wue_stack.append(wue)

# ==========================================
# Stack into 3D arrays: [time, rows, cols]
# ==========================================
ndvi_stack = np.stack(ndvi_stack)
wue_stack = np.stack(wue_stack)

# ==========================================
# Normalize within valid pixels only
# ==========================================
valid_mask = (ndvi_stack != ndvi_meta['nodata']) & (wue_stack != ndvi_meta['nodata'])

ndvi_valid = ndvi_stack[valid_mask]
wue_valid = wue_stack[valid_mask]

ndvi_min, ndvi_max = np.min(ndvi_valid), np.max(ndvi_valid)
wue_min, wue_max = np.min(wue_valid), np.max(wue_valid)

ndvi_range = ndvi_max - ndvi_min if ndvi_max != ndvi_min else 1
wue_range = wue_max - wue_min if wue_max != wue_min else 1

ndvi_norm = (ndvi_stack - ndvi_min) / ndvi_range
wue_norm = (wue_stack - wue_min) / wue_range

# ==========================================
# Compute ESI (cosine similarity) per year
# ==========================================
for i, year in enumerate(years):
    ndvi = ndvi_norm[i]
    wue = wue_norm[i]

    numerator = ndvi * wue
    denominator = np.sqrt(ndvi**2 + wue**2)
    esi = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    # Apply valid mask
    esi[~valid_mask[i]] = ndvi_meta['nodata']

    # Save output
    output_path = os.path.join(output_dir, f"{year}_ESI.tif")
    ndvi_meta.update({
        "dtype": "float32",
        "count": 1,
        "nodata": ndvi_meta['nodata']
    })

    with rasterio.open(output_path, "w", **ndvi_meta) as dst:
        dst.write(esi.astype(np.float32), 1)

print("✅ All yearly ESI rasters generated successfully.")
