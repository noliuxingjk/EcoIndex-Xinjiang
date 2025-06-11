import os
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from tqdm import tqdm

# ============================================
# User-defined paths (modify only these)
# ============================================
input_root = r"D:\your_project\data\resampled"
output_root = r"D:\your_project\data\clipped"
shapefile_dir = r"D:\your_project\shapefiles\region_boundary"

# ============================================
# Nodata value to be enforced in all outputs
# ============================================
custom_nodata_value = -9999

# ============================================
# Load clipping geometry from shapefile
# ============================================
shapefiles = [os.path.join(shapefile_dir, f) for f in os.listdir(shapefile_dir) if f.endswith(".shp")]
if not shapefiles:
    raise FileNotFoundError("No .shp files found in the specified directory.")

gdf = gpd.read_file(shapefiles[0])
geometries = gdf.geometry.values

# ============================================
# Collect all GeoTIFF files to process
# ============================================
tif_files = []
for root, _, files in os.walk(input_root):
    for file in files:
        if file.endswith(".tif") and not file.endswith(".tif.ovr"):
            tif_files.append(os.path.join(root, file))

print(f"🛰️ Found {len(tif_files)} raster files to clip.\n")

# ============================================
# Perform batch clipping and nodata standardization
# ============================================
for tif_path in tqdm(tif_files, desc="Clipping progress", unit="file"):
    file_name = os.path.basename(tif_path)
    base_name = file_name.replace("_resize", "").replace("_repro", "").replace(".tif", "")
    parent_folder = os.path.basename(os.path.dirname(tif_path)).replace("2_", "")

    output_dir = os.path.join(output_root, f"clipped_{parent_folder}")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"{base_name}_clipped.tif")

    try:
        with rasterio.open(tif_path) as src:
            # Determine nodata value (default to user-defined if missing)
            nodata = src.nodata if src.nodata is not None else custom_nodata_value

            # Clip the raster using geometry
            clipped_image, clipped_transform = mask(
                src,
                geometries,
                crop=True,
                filled=True,
                nodata=nodata
            )

            # Update metadata
            meta = src.meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": clipped_image.shape[1],
                "width": clipped_image.shape[2],
                "transform": clipped_transform,
                "nodata": nodata
            })

        # Save clipped raster
        with rasterio.open(output_file, "w", **meta) as dst:
            dst.write(clipped_image)

    except ValueError as e:
        print(f"⚠️ Skipped {file_name}: No spatial intersection with clipping geometry. ({e})")

print(f"\n✅ All rasters successfully clipped. Output saved in: {output_root}")
