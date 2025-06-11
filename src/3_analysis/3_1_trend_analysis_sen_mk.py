import os
import numpy as np
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import pymannkendall as mk
from tqdm import tqdm

# ===================================
# Define input/output and mask paths
# ===================================
ecoindex_dir = r"D:\your_project\data\EcoIndex"
esi_dir = r"D:\your_project\data\ESI"
output_dir = r"D:\your_project\results\TrendMaps"
shapefile_path = r"D:\your_project\shapefiles\study_region.shp"

os.makedirs(output_dir, exist_ok=True)

# ===================================
# Load shapefile geometries
# ===================================
shapefile = gpd.read_file(shapefile_path)
geoms = shapefile.geometry.values

# ===================================
# Load multiyear raster time series
# ===================================
def load_raster_series(folder, keyword):
    files = sorted([f for f in os.listdir(folder) if keyword in f and f.endswith('.tif')])
    files = [f for f in files if 'map' not in f and 'mosaic' not in f]  # Exclude non-yearly tiles
    stack = []
    for f in files:
        with rasterio.open(os.path.join(folder, f)) as src:
            img, _ = mask(src, geoms, crop=False)
            stack.append(img[0])
            transform = src.transform
            crs = src.crs
    return np.array(stack), transform, crs

eco_stack, transform, crs = load_raster_series(ecoindex_dir, 'EcoIndex')
esi_stack, _, _ = load_raster_series(esi_dir, 'ESI')
years = np.arange(2000, 2024)

# ===================================
# Calculate Sen's slope
# ===================================
def compute_sen_slope(ts):
    ts = np.array(ts)
    valid = ~np.isnan(ts)
    if np.sum(valid) < 2:
        return np.nan
    slopes = [(ts[j] - ts[i]) / (years[j] - years[i])
              for i in range(len(ts)) for j in range(i + 1, len(ts))
              if valid[i] and valid[j]]
    return np.median(slopes) if slopes else np.nan

# ===================================
# Calculate Mann-Kendall p-value
# ===================================
def compute_mk_pvalue(ts):
    ts = np.array(ts)
    valid = ~np.isnan(ts)
    if np.sum(valid) < 6:
        return np.nan
    result = mk.original_test(ts[valid])
    return result.p

# ===================================
# Apply trend analysis to each pixel
# ===================================
def trend_analysis(data_stack, label):
    height, width = data_stack.shape[1:]
    sen_slope = np.full((height, width), np.nan)
    mk_pvalue = np.full((height, width), np.nan)

    for i in tqdm(range(height), desc=f"Analyzing {label}"):
        for j in range(width):
            ts = data_stack[:, i, j]
            if np.all(np.isnan(ts)):
                continue
            sen_slope[i, j] = compute_sen_slope(ts)
            mk_pvalue[i, j] = compute_mk_pvalue(ts)

    return sen_slope, mk_pvalue

eco_sen, eco_p = trend_analysis(eco_stack, "EcoIndex")
esi_sen, esi_p = trend_analysis(esi_stack, "ESI")

# ===================================
# Save trend raster outputs
# ===================================
def save_raster(array, path, transform, crs, dtype='float32'):
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(array.astype(dtype), 1)

save_raster(eco_sen, os.path.join(output_dir, 'EcoIndex_SenSlope.tif'), transform, crs)
save_raster(eco_p, os.path.join(output_dir, 'EcoIndex_MK_pvalue.tif'), transform, crs)
save_raster(esi_sen, os.path.join(output_dir, 'ESI_SenSlope.tif'), transform, crs)
save_raster(esi_p, os.path.join(output_dir, 'ESI_MK_pvalue.tif'), transform, crs)

print("✅ Trend analysis completed and results saved.")
