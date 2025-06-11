import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from tqdm import tqdm

# =====================================
# User-defined input and output folders
# (Modify these two paths as needed)
# =====================================
input_root = r"D:\your_project\data\reprojected"
output_root = r"D:\your_project\data\resampled"

# =====================================
# Target spatial resolution (in meters)
# =====================================
target_resolution = (823.25, 823.25)  # (x_res, y_res)

# =====================================
# Prepare output directory
# =====================================
os.makedirs(output_root, exist_ok=True)

# =====================================
# Collect all GeoTIFF files for processing
# =====================================
tif_files = []
for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".tif") and not file.endswith(".tif.ovr"):
            tif_files.append(os.path.join(root, file))

# =====================================
# Perform resampling for each raster
# =====================================
for tif_path in tqdm(tif_files, desc="Resampling progress"):
    # Maintain relative folder structure
    relative_path = os.path.relpath(os.path.dirname(tif_path), input_root)
    output_folder = os.path.join(output_root, f"resampled_{relative_path}")
    os.makedirs(output_folder, exist_ok=True)

    filename, _ = os.path.splitext(os.path.basename(tif_path))
    output_path = os.path.join(output_folder, f"{filename}_resampled.tif")

    with rasterio.open(tif_path) as src:
        # Compute scaling factors and new shape
        scale_x = src.res[0] / target_resolution[0]
        scale_y = src.res[1] / target_resolution[1]
        new_width = int(src.width * scale_x)
        new_height = int(src.height * scale_y)

        # Compute new affine transform
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        # Update metadata
        meta = src.meta.copy()
        meta.update({
            'height': new_height,
            'width': new_width,
            'transform': transform
        })

        # Resample and write to output
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest
                )

print("✅ All rasters successfully resampled and saved.")
