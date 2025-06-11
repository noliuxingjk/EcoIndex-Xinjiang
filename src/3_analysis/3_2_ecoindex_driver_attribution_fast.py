import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

# ===============================
# Directory and configuration
# ===============================
ecoindex_dir = r"D:\project\data\EcoIndex"
driver_dir = r"D:\project\data\Drivers"
output_dir = r"D:\project\results\Attribution_Fast"
shapefile_path = r"D:\project\shapefiles\region_boundary.shp"
os.makedirs(output_dir, exist_ok=True)

years = np.arange(2000, 2024)

# Define driver variables and their folder/key mapping
driver_mapping = {
    'PR': ('Precipitation', 'TerraClimate_pr'),
    'SOIL': ('SoilMoisture', 'TerraClimate_soil'),
    'TEMP': ('Temperature', 'TerraClimate_AvgTemp'),
    'NL': ('Nightlight', 'Nightlight'),
    'CLCD': ('LandCover', 'CLCD')
}

# ===============================
# Load shapefile geometry
# ===============================
shapefile = gpd.read_file(shapefile_path)
geoms = shapefile.geometry.values

# ===============================
# Resample raster to lower resolution
# ===============================
def resample_raster(input_path, output_path, scale_factor, is_categorical=False):
    with rasterio.open(input_path) as src:
        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)

        if is_categorical:
            data = src.read(1)
            resampled = data[::int(1 / scale_factor), ::int(1 / scale_factor)][np.newaxis, :, :]
        else:
            resampled = src.read(
                out_shape=(src.count, new_height, new_width),
                resampling=Resampling.average
            )

        transform = src.transform * src.transform.scale(
            src.width / resampled.shape[-1],
            src.height / resampled.shape[-1]
        )

        profile = src.profile
        profile.update({
            'height': resampled.shape[1],
            'width': resampled.shape[2],
            'transform': transform
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(resampled)

# ===============================
# Load and preprocess annual rasters
# ===============================
def load_stack(folder, keyword, years, scale_factor=0.25, is_index=False, is_categorical=False):
    stack = []
    for year in years:
        if is_index:
            path = os.path.join(folder, f"{year}_EcoIndex.tif")
        elif keyword in ['Nightlight', 'CLCD']:
            path = os.path.join(folder, f"{keyword}_{year}.tif_remove.tif")
        else:
            path = os.path.join(folder, f"{year}_{keyword}.tif_remove.tif")

        resampled_path = path.replace(".tif", "_resampled.tif")
        if not os.path.exists(resampled_path):
            resample_raster(path, resampled_path, scale_factor, is_categorical=is_categorical)

        with rasterio.open(resampled_path) as src:
            img, _ = mask(src, geoms, crop=False)
            stack.append(img[0])
            transform = src.transform
            crs = src.crs
    return np.array(stack), transform, crs

# Load EcoIndex stack
eco_stack, transform, crs = load_stack(ecoindex_dir, None, years, scale_factor=0.25, is_index=True)

# Load drivers
driver_stacks = {}
for var, (subfolder, keyword) in driver_mapping.items():
    full_path = os.path.join(driver_dir, subfolder)
    is_categorical = (var == 'CLCD')
    driver_stacks[var], _, _ = load_stack(full_path, keyword, years, scale_factor=0.25, is_categorical=is_categorical)

# ===============================
# Calculate standardized anomalies
# ===============================
def calc_anomalies(stack, categorical=False):
    if categorical:
        return stack
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    return (stack - mean) / (std + 1e-6)

eco_anomaly = calc_anomalies(eco_stack)
driver_anomalies = {
    var: calc_anomalies(driver_stacks[var], categorical=(var == 'CLCD'))
    for var in driver_stacks
}

# ===============================
# Sample training data for RF
# ===============================
height, width = eco_anomaly.shape[1:]
np.random.seed(42)
valid_pixels = np.argwhere(~np.isnan(eco_anomaly[0]))
selected_idx = valid_pixels[np.random.choice(valid_pixels.shape[0], size=20000, replace=False)]

X_train, y_train = [], []
for idx in selected_idx:
    i, j = idx
    y_series = eco_anomaly[:, i, j]
    X_series = np.stack([driver_anomalies[v][:, i, j] for v in driver_mapping], axis=1)
    mask = ~np.isnan(y_series) & ~np.isnan(X_series).any(axis=1)
    if np.sum(mask) < 10:
        continue
    y_train.append(y_series[mask])
    X_train.append(X_series[mask])

X_train_all = np.vstack(X_train)
y_train_all = np.hstack(y_train)

# ===============================
# Train baseline Random Forest
# ===============================
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_all, y_train_all)
print("✅ Random Forest model trained.")

# ===============================
# Attribution: pixel-wise feature importance
# ===============================
importance_array = np.full((height, width, len(driver_mapping)), np.nan)

for i in tqdm(range(height), desc="Pixel-wise Attribution"):
    for j in range(width):
        y = eco_anomaly[:, i, j]
        X = np.stack([driver_anomalies[v][:, i, j] for v in driver_mapping], axis=1)
        if np.isnan(y).all() or np.isnan(X).all():
            continue
        mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
        if np.sum(mask) < 10:
            continue
        rf_pixel = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_pixel.fit(X[mask], y[mask])
        importance_array[i, j, :] = rf_pixel.feature_importances_

# ===============================
# Save feature importance maps
# ===============================
for idx, var in enumerate(driver_mapping):
    output_path = os.path.join(output_dir, f"{var}_importance_fast.tif")
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype='float32',
        crs=crs,
        transform=transform,
        nodata=np.nan
    ) as dst:
        dst.write(importance_array[:, :, idx], 1)

# ===============================
# Generate driver dominance classification
# ===============================
dominance_map = np.full((height, width), np.nan)

for i in range(height):
    for j in range(width):
        importance = importance_array[i, j, :]
        if np.isnan(importance).all():
            continue
        climate_score = np.sum(importance[0:3])   # PR, SOIL, TEMP
        human_score = np.sum(importance[3:5])     # NL, CLCD

        if abs(climate_score - human_score) <= 0.05:
            dominance_map[i, j] = 3  # Mixed influence
        elif climate_score > human_score:
            dominance_map[i, j] = 1  # Climate-dominated
        else:
            dominance_map[i, j] = 2  # Human-dominated

out_path = os.path.join(output_dir, "Driver_Dominance_fast.tif")
with rasterio.open(
    out_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=1,
    dtype='uint8',
    crs=crs,
    transform=transform,
    nodata=0
) as dst:
    dst.write(dominance_map.astype('uint8'), 1)

print("🏁 Completed attribution and classification mapping.")
