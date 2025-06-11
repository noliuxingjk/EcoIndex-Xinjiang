import os
import numpy as np
import rasterio
from rasterio.mask import mask
import fiona
from sklearn.decomposition import PCA

# ============================================================
# Configurable Paths (replace with your actual project folders)
# ============================================================
ndvi_dir = r"D:\your_project\data\NDVI_cleaned"
wue_dir = r"D:\your_project\data\WUE_cleaned"
output_dir = r"D:\your_project\results\EcoIndex_PCA"
shapefile_path = r"D:\your_project\shapefiles\region_boundary.shp"

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Load study area shapefile
# ============================================================
with fiona.open(shapefile_path, "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]

# ============================================================
# Read yearly NDVI and WUE raster files
# ============================================================
years = range(2000, 2024)
ndvi_list = []
wue_list = []

for year in years:
    ndvi_path = os.path.join(ndvi_dir, f"{year}_NDVI_cleaned.tif")
    wue_path = os.path.join(wue_dir, f"{year}_WUE_cleaned.tif")

    with rasterio.open(ndvi_path) as src_ndvi:
        ndvi_data, _ = mask(src_ndvi, shapes, crop=False, filled=True, nodata=np.nan)
    with rasterio.open(wue_path) as src_wue:
        wue_data, _ = mask(src_wue, shapes, crop=False, filled=True, nodata=np.nan)

    ndvi_list.append(ndvi_data.squeeze())
    wue_list.append(wue_data.squeeze())

ndvi_stack = np.stack(ndvi_list)  # shape: (years, height, width)
wue_stack = np.stack(wue_list)

# ============================================================
# Normalize within valid region
# ============================================================
valid_mask = (~np.isnan(ndvi_stack)) & (~np.isnan(wue_stack)) & (ndvi_stack > 0) & (wue_stack > 0)
ndvi_valid = ndvi_stack[valid_mask]
wue_valid = wue_stack[valid_mask]

ndvi_mean, ndvi_std = np.mean(ndvi_valid), np.std(ndvi_valid)
wue_mean, wue_std = np.mean(wue_valid), np.std(wue_valid)

print(f"✅ Normalization parameters:")
print(f"NDVI: mean={ndvi_mean:.4f}, std={ndvi_std:.4f}")
print(f"WUE:  mean={wue_mean:.4f}, std={wue_std:.4f}")

ndvi_norm = (ndvi_stack - ndvi_mean) / ndvi_std
wue_norm = (wue_stack - wue_mean) / wue_std

# ============================================================
# Principal Component Analysis (PCA) on valid pixels
# ============================================================
X_valid = np.vstack([ndvi_norm[valid_mask], wue_norm[valid_mask]]).T
pca = PCA(n_components=1)
pca.fit(X_valid)
coeff_ndvi, coeff_wue = pca.components_[0]

print(f"✅ PCA coefficients: NDVI={coeff_ndvi:.4f}, WUE={coeff_wue:.4f}")

# ============================================================
# Compute PCA-based EcoIndex for each year and save GeoTIFF
# ============================================================
for idx, year in enumerate(years):
    ndvi_img = ndvi_norm[idx]
    wue_img = wue_norm[idx]

    ecoindex = np.full(ndvi_img.shape, np.nan, dtype=np.float32)
    valid_pixels = (~np.isnan(ndvi_img)) & (~np.isnan(wue_img)) & (ndvi_img > -999) & (wue_img > -999)
    ecoindex[valid_pixels] = coeff_ndvi * ndvi_img[valid_pixels] + coeff_wue * wue_img[valid_pixels]

    # Read metadata from reference NDVI raster
    ref_path = os.path.join(ndvi_dir, f"{year}_NDVI_cleaned.tif")
    with rasterio.open(ref_path) as src:
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": ecoindex.shape[0],
            "width": ecoindex.shape[1],
            "transform": src.transform,
            "crs": src.crs,
            "count": 1,
            "dtype": "float32",
            "nodata": np.nan
        })

    # Save output
    output_path = os.path.join(output_dir, f"{year}_EcoIndex.tif")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(ecoindex, 1)

print("✅ All annual EcoIndex maps generated successfully.")
