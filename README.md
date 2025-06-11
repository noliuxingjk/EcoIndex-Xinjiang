# EcoIndex-Xinjiang

A reproducible workflow for eco-hydrological index calculation, spatiotemporal trend detection, and driver attribution across Xinjiang from 2000 to 2023. This repository implements a full pipeline for preprocessing remote sensing datasets, computing ecological indicators (e.g., EcoIndex and ESI), and analyzing the dominant climate and anthropogenic drivers using machine learning methods.

---

## 🧭 1_Script Overview
This repository contains a modular pipeline for preprocessing remote sensing data, computing ecological indices, and performing trend and driver attribution analysis across Xinjiang (2000–2023). Scripts are organized into three main stages:

### 📦 1.1_preprocessing/ — Data Preparation & Cleaning

| Script                                   | Description                                                              |
| ---------------------------------------- | ------------------------------------------------------------------------ |
| `1_1_plot_regional_eco_variables.py`     | Visualizes time-series eco-variables (NDVI, GPP, ET, etc.) by subregion. |
| `1_2_batch_reproject_rasters_albers.py`  | Reprojects all rasters to Albers Equal Area Conic projection.            |
| `1_3_batch_downsample_rasters.py`        | Downsamples rasters to reduce spatial resolution and data size.          |
| `1_4_clip_rasters_by_boundary.py`        | Clips rasters based on administrative boundaries using shapefiles.       |
| `1_5_fill_blank_pixels_by_block_mean.py` | Fills missing pixels using block-wise local mean interpolation.          |


### 📊 1.2_index_calculation/ — Index Derivation
| Script            | Description                                                                    |
| ----------------- | ------------------------------------------------------------------------------ |
| `2_1_ecoindex.py` | Constructs a composite EcoIndex using PCA on normalized NDVI and WUE.          |
| `2_2_quadrant.py` | Classifies year-to-year NDVI–WUE changes into 4 ecohydrological quadrants.     |
| `2_3_ESI.py`      | Calculates the Ecohydrological Similarity Index (ESI) using cosine similarity. |


### 📈 1.3_analysis/ — Trend & Attribution Analysis
| Script                                           | Description                                                                    |
| ------------------------------------------------ | ------------------------------------------------------------------------------ |
| `3_1_trend_analysis_sen_mk.py`                   | Applies Sen's slope estimator and Mann–Kendall test for trend detection.       |
| `3_2_ecoindex_driver_attribution_fast.py`        | Performs pixel-wise RF regression to attribute EcoIndex variations to drivers. |
| `3_3_analyze_driver_importance_and_dominance.py` | Aggregates driver importances and visualizes climate/human dominance patterns. |

---

## ⚙️ 2_Installation & Dependencies
This repository is written in Python 3.8+ and requires the following core libraries:

```
numpy
pandas
matplotlib
tqdm
rasterio
geopandas
fiona
shapely
scikit-learn
pymannkendall
```

You can install all required packages using:

```
pip install -r requirements.txt
```

ℹ️ Note:
For geospatial operations (rasterio, geopandas, fiona, etc.), it is recommended to use a conda-based environment (e.g., Anaconda) to avoid binary compatibility issues. You can set up a clean environment via:

---

## 📦 3_Data Overview
The analysis in this repository is based on a collection of multi-source, long-term ecological and climatic remote sensing datasets covering Xinjiang from 2000 to 2023. The major data categories include:
### 🛰️ 3_1 Remote Sensing Inputs
| Variable | Description                                | Temporal Resolution | Source                          |
| -------- | ------------------------------------------ | ------------------- | ------------------------------- |
| NDVI     | Normalized Difference Vegetation Index     | Monthly             | MODIS MOD13Q1 / GIMMS / CLCD    |
| WUE      | Water Use Efficiency (derived from GPP/ET) | Annual              | Based on MOD17 + ET datasets    |
| PR       | Precipitation                              | Monthly             | TerraClimate / CHIRPS           |
| TEMP     | Mean Air Temperature                       | Monthly             | TerraClimate                    |
| SOIL     | Surface Soil Moisture                      | Monthly             | ESA CCI / SMAP                  |
| NL       | Nighttime Light Intensity                  | Yearly              | DMSP-OLS / VIIRS                |
| CLCD     | Land Cover Classification                  | Yearly              | CLCD product (China Land Cover) |
### 📐 3_2 Spatial Information
 - Projection: All rasters are reprojected to Albers Equal Area Conic for regional analysis consistency.
 - Resolution: Standardized to 0.05° (~5 km) for ecological modeling and visualization.
 - Masking: A shapefile of Xinjiang province and its three subregions (Northern, Southern, Eastern Xinjiang) is used to clip and subset all datasets.

### 📦 3_3 Derived Products
 - EcoIndex: Composite index reflecting vegetation productivity and water efficiency.
 - ESI (Ecohydrological Similarity Index): Cosine-based similarity measure between NDVI and WUE dynamics.
 - Trend Layers: Pixel-wise Sen's slope and Mann–Kendall p-values for change detection.
 - Driver Layers: Variable importance maps from Random Forest models (e.g., PR, TEMP, SOIL).
 - Dominance Maps: Classified maps identifying climate-, human-, or mixed-dominated regions.

🔧 All intermediate files and output products are saved as GeoTIFFs and can be visualized in GIS software or Python-based mapping tools.

---

## 📊 Output Preview *(to be added)*

- Annual EcoIndex and ESI maps (.tif)  
- Trend significance layers  
- Driver importance rasters and summary tables  
- Dominance classification maps (climate, human, mixed)  

---

## 📄 License

This repository is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 📬 Contact

For questions or collaboration requests, feel free to contact:

**Your Name**  
Email: *your_email@domain.com*  
GitHub: [your_username](https://github.com/your_username)
