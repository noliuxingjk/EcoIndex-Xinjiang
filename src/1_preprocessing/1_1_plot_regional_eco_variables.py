import pandas as pd
import matplotlib.pyplot as plt

# Define file paths (replace with actual paths as needed)
north_path = r'path_to_data/Northern_Xinjiang_Statistics.csv'
south_path = r'path_to_data/Southern_Xinjiang_Statistics.csv'
east_path  = r'path_to_data/Eastern_Xinjiang_Statistics.csv'

# Load datasets for the three subregions
north_df = pd.read_csv(north_path)
south_df = pd.read_csv(south_path)
east_df  = pd.read_csv(east_path)

# Rename columns for consistency across datasets
columns = ['Year', 'ET', 'GPP', 'MODIS_NDVI', 'Nightlight', 'PR', 'SOIL', 'TEMP']
north_df.columns = columns
south_df.columns = columns
east_df.columns  = columns

# Ensure 'Year' is of integer type
for df in [north_df, south_df, east_df]:
    df['Year'] = df['Year'].astype(int)

# Define units for axis labeling
units = {
    'ET': ' (mm)',
    'GPP': ' (gC/m²/day)',
    'MODIS_NDVI': '',
    'Nightlight': ' (nW/cm²/sr)',
    'PR': ' (mm)',
    'SOIL': ' (%)',
    'TEMP': ' (°C)'
}

# Regional configuration: DataFrame and associated color
regions = {
    'Northern Xinjiang': (north_df, 'orange'),
    'Southern Xinjiang': (south_df, '#20B2AA'),
    'Eastern Xinjiang':  (east_df,  '#BA55D3')
}

# Set up multi-panel figure: 3 columns (regions) × 7 rows (variables)
fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(24, 28), sharex=True)
variables = columns[1:]

# Plot each variable for each subregion
for col_idx, (region_name, (df, color)) in enumerate(regions.items()):
    for row_idx, var in enumerate(variables):
        ax = axes[row_idx, col_idx]
        ax.plot(df['Year'], df[var], marker='o', linewidth=2.5, color=color, markersize=6)

        # Y-axis label for the first column
        if col_idx == 0:
            ax.set_ylabel(f'{var}{units[var]}', fontname='Times New Roman', fontsize=28, fontweight='bold')

        # Column title
        if row_idx == 0:
            ax.set_title(region_name, fontname='Times New Roman', fontsize=30, fontweight='bold')

        # Axis configuration
        ax.set_xlim([2000, 2024])
        ax.set_xticks(list(range(2000, 2025, 3)))
        ax.tick_params(axis='both', labelsize=24)
        ax.grid(True)

        # Uniform font settings
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Times New Roman')

        # X-axis label for bottom row
        if row_idx == len(variables) - 1:
            ax.set_xlabel('Year', fontname='Times New Roman', fontsize=25, fontweight='bold')

# Layout adjustment
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()
