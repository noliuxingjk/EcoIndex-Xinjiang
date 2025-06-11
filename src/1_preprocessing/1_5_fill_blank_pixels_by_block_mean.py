import os
import numpy as np
import rasterio
import rasterio.features
import fiona
from shapely.geometry import shape
from tqdm import tqdm

# =============================================
# Configurable Paths (Edit only these)
# =============================================
input_root = r"D:\your_project\data\resized"
output_root = r"D:\your_project\data\filled"
shapefile_dir = r"D:\your_project\shapefiles\boundary"
block_rows = 17
block_cols = 17

# =============================================
# Load boundary geometry (e.g., Xinjiang)
# =============================================
shp_files = [os.path.join(shapefile_dir, f) for f in os.listdir(shapefile_dir) if f.endswith('.shp')]
if not shp_files:
    raise FileNotFoundError("No shapefile found in the specified directory.")
shp_path = shp_files[0]

with fiona.open(shp_path, 'r') as shapefile:
    geometries = [shape(feature['geometry']) for feature in shapefile]

# =============================================
# Discover all input GeoTIFF files
# =============================================
tif_files = []
for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith('.tif') and not file.endswith('.tif.ovr'):
            tif_files.append(os.path.join(root, file))

os.makedirs(output_root, exist_ok=True)

# =============================================
# Batch process each GeoTIFF file
# =============================================
for tif_path in tqdm(tif_files, desc="Processing rasters", unit="file"):
    filename = os.path.basename(tif_path).replace("_resize2", "").replace("_clip", "")
    category = os.path.basename(os.path.dirname(tif_path)).replace("4_", "").replace("_clip", "")
    output_dir = os.path.join(output_root, f"filled_{category}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}_filled.tif")

    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(np.float32)
        transform = src.transform
        nodata = src.nodata
        height, width = data.shape

        shp_mask = rasterio.features.geometry_mask(
            geometries,
            transform=transform,
            invert=True,
            out_shape=(height, width)
        )

    # Data-specific preprocessing
    if 'Nightlight' in filename:
        data = np.where((data < 0) & (shp_mask == 1), 0, data)
    if 'CLCD' in filename.upper():
        data = np.clip(data, 1, 9)

    # Identify invalid (blank) pixels
    def is_blank(x):
        if nodata is not None:
            return np.isnan(x) or (x == nodata) or (x < -1e30)
        else:
            return np.isnan(x) or (x < -1e30)

    blank_mask = np.vectorize(is_blank)(data)
    replace_mask = (shp_mask == 1) & (blank_mask == 1)
    valid_mask = (shp_mask == 1) & (~blank_mask)
    global_mean = np.nanmean(data[valid_mask]) if np.any(valid_mask) else 0

    # Compute local means by block
    local_mean_map = np.full((block_rows, block_cols), np.nan)
    bh, bw = height // block_rows, width // block_cols

    for i in range(block_rows):
        for j in range(block_cols):
            row_start = i * bh
            row_end = (i + 1) * bh if i < block_rows - 1 else height
            col_start = j * bw
            col_end = (j + 1) * bw if j < block_cols - 1 else width

            block_data = data[row_start:row_end, col_start:col_end]
            block_mask = (shp_mask[row_start:row_end, col_start:col_end] == 1) & (~np.isnan(block_data)) & (block_data > -1e30)
            local_mean_map[i, j] = np.nanmean(block_data[block_mask]) if np.any(block_mask) else global_mean

    # Replace missing values by corresponding block average
    filled_data = np.copy(data)
    rows, cols = np.where(replace_mask)
    for r, c in zip(rows, cols):
        bi = min(r // bh, block_rows - 1)
        bj = min(c // bw, block_cols - 1)
        filled_data[r, c] = local_mean_map[bi, bj]

    # Final postprocessing
    filled_data = np.where(
        (np.isnan(filled_data)) | (filled_data < -1e30),
        -9999,
        filled_data
    )
    if 'CLCD' in filename.upper():
        filled_data = np.clip(filled_data, 1, 9)
    if 'Nightlight' in filename:
        filled_data = np.clip(filled_data, 0, None)

    # Save final raster
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()

    meta.update(dtype='float32', nodata=-9999)
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(filled_data.astype(np.float32), 1)

print(f"\n✅ All raster cleaning completed. Output directory: {output_root}")
