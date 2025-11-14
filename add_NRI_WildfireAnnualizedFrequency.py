import pandas as pd
import os

# --- Configuration ---
IMAGE_DATA_FILE = 'data/NRI_Table_Counties/image_county_data.csv'
NRI_DATA_FILE = 'data/NRI_Table_Counties/NRI_Table_Counties.csv'
OUTPUT_FILE = 'data/NRI_Table_Counties/image_county_data_with_frequency.csv'

# Column names in NRI table to be extracted
NRI_COUNTY_COL = 'COUNTY'
NRI_STATE_COL = 'STATE'
# NEW COLUMN TO EXTRACT
NRI_FREQ_COL = 'WFIR_AFREQ'

# New column name for the merged output
NEW_COLUMN_NAME = 'Wildfire_Annualized_Frequency'

# --- 1. Load Image Data ---
print(f"Loading image data from: {IMAGE_DATA_FILE}")
df_images = pd.read_csv(IMAGE_DATA_FILE)

if 'county' not in df_images.columns or 'state' not in df_images.columns:
    print("Error: Image CSV must contain 'county' and 'state' columns.")
    # Execution will stop here if columns are missing

# --- 2. Load NRI Data ---
print(f"Loading NRI data from: {NRI_DATA_FILE}")
df_nri = pd.read_csv(NRI_DATA_FILE)
    
# Check required columns
required_cols = [NRI_COUNTY_COL, NRI_STATE_COL, NRI_FREQ_COL]
if not all(col in df_nri.columns for col in required_cols):
    print(f"Error: NRI file must contain columns: {required_cols}")
    # Execution will stop here if columns are missing

# --- 3. Standardize County/State Names for Merge ---

# Standardize NRI table: combine COUNTY + STATE as join key
df_nri['join_key'] = (
    df_nri[NRI_COUNTY_COL].str.replace(' County', '', regex=False)
    .str.strip().str.title() + ', ' +
    df_nri[NRI_STATE_COL].str.strip().str.title()
)

# Standardize Image table: combine county + state as join key
df_images['join_key'] = (
    df_images['county'].str.replace(' County', '', regex=False)
    .str.strip().str.title() + ', ' +
    df_images['state'].str.strip().str.title()
)

# Select only the join key and the new wildfire column from NRI, drop duplicates
df_nri_final = df_nri[['join_key', NRI_FREQ_COL]].rename(
    columns={NRI_FREQ_COL: NEW_COLUMN_NAME}
)
df_nri_final = df_nri_final.drop_duplicates(subset=['join_key'])


# --- 4. Merge ---
df_merged = pd.merge(
    df_images,
    df_nri_final,
    on='join_key', # Merge on the new, standardized 'join_key' in both DataFrames
    how='left'
)

# Drop temporary join column
df_merged = df_merged.drop(columns=['join_key'])

# --- 5. Save Output ---
df_merged.to_csv(OUTPUT_FILE, index=False)

print("\n--- Merge Complete ---")
print(f"Total rows: {len(df_merged)}")
print(f"New column added: '{NEW_COLUMN_NAME}'")
print(f"Output saved to: {OUTPUT_FILE}")
print("\nSample Data (first 5 rows with new column):")
print(df_merged.head())