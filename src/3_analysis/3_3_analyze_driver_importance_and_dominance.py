import os
import numpy as np
import rasterio
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from rasterio.mask import mask

# =========================
# Define file paths
# =========================
driver_raster_dir = r"D:\project\outputs\driver_attribution"
output_dir = r"D:\project\outputs\region_stats"
os.makedirs(output_dir, exist_ok=True)

# Region shapefiles (province-wide and sub-regions)
region_shapefiles = {
    'Overall': r"D:\project\data\shapefiles\region_boundary.shp",
    'Northern Xinjiang': r"D:\project\data\shapefiles\north_region.shp",
    'Southern Xinjiang': r"D:\project\data\shapefiles\south_region.shp",
    'Eastern Xinjiang': r"D:\project\data\shapefiles\east_region.shp"
}

# Raster filenames for driver importance
driver_rasters = {
    'PR': 'PR_importance_fast.tif',
    'SOIL': 'SOIL_importance_fast.tif',
    'TEMP': 'TEMP_importance_fast.tif',
    'NL': 'NL_importance_fast.tif',
    'CLCD': 'CLCD_importance_fast.tif'
}
dominance_raster = 'Driver_Dominance_fast.tif'

# =========================
# Utility: Mask raster with region shapefile
# =========================
def read_masked_data(raster_path, shapefile_path):
    with rasterio.open(raster_path) as src:
        gdf = gpd.read_file(shapefile_path)
        masked, _ = mask(src, gdf.geometry, crop=False)
        data = masked[0]
        data = np.where((data == src.nodata) | (np.isnan(data)), np.nan, data)
    return data

# =========================
# Part 1: Statistics of driver importance
# =========================
importance_results = []

for region_name, shp_path in region_shapefiles.items():
    region_stats = {'Region': region_name}
    for var, filename in driver_rasters.items():
        raster_path = os.path.join(driver_raster_dir, filename)
        data = read_masked_data(raster_path, shp_path)
        valid = data[~np.isnan(data)]
        region_stats[f'{var}_Mean'] = np.nanmean(valid)
        region_stats[f'{var}_Median'] = np.nanmedian(valid)
    importance_results.append(region_stats)

importance_df = pd.DataFrame(importance_results)
importance_df.to_csv(os.path.join(output_dir, 'Driver_Importance_Statistics.csv'), index=False)

print("✅ Driver importance statistics completed.")

# =========================
# Part 2: Dominant driver classification ratio
# =========================
dominance_results = []

for region_name, shp_path in region_shapefiles.items():
    raster_path = os.path.join(driver_raster_dir, dominance_raster)
    data = read_masked_data(raster_path, shp_path)
    valid = data[~np.isnan(data)]
    total = len(valid)
    climate = np.sum(valid == 1)
    human = np.sum(valid == 2)
    mixed = np.sum(valid == 3)

    dominance_results.append({
        'Region': region_name,
        'Climate_Dominated_%': climate / total * 100,
        'Human_Dominated_%': human / total * 100,
        'Mixed_Influence_%': mixed / total * 100
    })

dominance_df = pd.DataFrame(dominance_results)
dominance_df.to_csv(os.path.join(output_dir, 'Driver_Dominance_Statistics.csv'), index=False)

print("✅ Driver dominance classification statistics completed.")

# =========================
# Optional: Visualization - histogram example
# =========================
def plot_histogram(data, title, output_path):
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=30, edgecolor='black')
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Example: Histogram for PR importance in Northern Xinjiang
sample_data = read_masked_data(
    os.path.join(driver_raster_dir, driver_rasters['PR']),
    region_shapefiles['Northern Xinjiang']
)
valid_values = sample_data[~np.isnan(sample_data)]

plot_histogram(
    valid_values,
    'PR Importance Distribution - Northern Xinjiang',
    os.path.join(output_dir, 'Histogram_Northern_PR.png')
)

print("✅ Example histogram generated.")
