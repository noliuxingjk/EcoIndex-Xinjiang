import os
import rasterio
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling

# =======================================
# Define input and output root directories
# (Update these two paths as needed)
# =======================================
input_root = r"D:\your_project\raw_data"
output_root = r"D:\your_project\processed_data\reprojected"

# =======================================
# Define target projection: Albers Equal Area
# Modify parameters based on your regional needs
# =======================================
target_crs = {
    'proj': 'aea',
    'lat_1': 25,
    'lat_2': 47,
    'lat_0': 0,
    'lon_0': 105,
    'x_0': 0,
    'y_0': 0,
    'datum': 'WGS84',
    'units': 'm',
    'no_defs': True
}

# =======================================
# Batch reproject all .tif files in folder
# =======================================
for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".tif") and not file.endswith(".tif.ovr"):
            input_path = os.path.join(root, file)

            # Generate output subdirectory based on relative path
            relative_subfolder = os.path.relpath(root, input_root)
            output_subfolder = os.path.join(output_root, f"reproj_{relative_subfolder}")
            os.makedirs(output_subfolder, exist_ok=True)

            # Define output file path
            filename_no_ext, _ = os.path.splitext(file)
            output_path = os.path.join(output_subfolder, f"{filename_no_ext}_reproj.tif")

            # Open source raster and reproject
            with rasterio.open(input_path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds)

                metadata = src.meta.copy()
                metadata.update({
                    'crs': target_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
                })

                with rasterio.open(output_path, 'w', **metadata) as dst:
                    for i in range(1, src.count + 1):
                        reproject(
                            source=rasterio.band(src, i),
                            destination=rasterio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=Resampling.nearest
                        )

print("✅ All raster files have been successfully reprojected and saved.")
