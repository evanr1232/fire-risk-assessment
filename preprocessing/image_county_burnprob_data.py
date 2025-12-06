import pandas as pd
import os

# --- Configuration ---
IMAGE_DATA_FILE = 'data/NRI_Table_Counties/image_county_data.csv'
NRI_DATA_FILE = 'data/NRI_Table_Counties/NRI_Table_Counties.csv'
OUTPUT_FILE = 'data/NRI_Table_Counties/image_county_data_with_burnprob.csv'

# Column names in NRI table
NRI_COUNTY_COL = 'COUNTY'
NRI_STATE_COL = 'STATE'
NRI_PROB_COL = 'WFIR_RISKS'
NEW_COLUMN_NAME = 'Wildfire_Risk_Score_NRI'  # new column for merged score

# --- 1. Load Image Data ---
print(f"Loading image data from: {IMAGE_DATA_FILE}")
try:
    df_images = pd.read_csv(IMAGE_DATA_FILE)
    # Check for 'county' and 'state' columns which are needed for the corrected merge
    if 'county' not in df_images.columns or 'state' not in df_images.columns:
        raise ValueError("Image CSV must contain 'county' and 'state' columns.")
except FileNotFoundError:
    print(f"Error: File '{IMAGE_DATA_FILE}' not found.")
    exit()
except ValueError as e:
    print(f"Error: {e}")
    exit()

# --- 2. Load NRI Data ---
print(f"Loading NRI data from: {NRI_DATA_FILE}")
try:
    df_nri = pd.read_csv(NRI_DATA_FILE)
except FileNotFoundError:
    print(f"Error: File '{NRI_DATA_FILE}' not found.")
    exit()

# Check required columns
required_cols = [NRI_COUNTY_COL, NRI_STATE_COL, NRI_PROB_COL]
if not all(col in df_nri.columns for col in required_cols):
    print(f"Error: NRI file must contain columns: {required_cols}")
    exit()

# --- 3. Standardize County/State Names for Merge (FIXED) ---

# Standardize NRI table: combine COUNTY + STATE as join key
# The format will be 'County Name, State Name' (e.g., 'Autauga, Alabama')
df_nri['join_key'] = (
    df_nri[NRI_COUNTY_COL].str.replace(' County', '', regex=False) # Remove ' County' if it exists in NRI data
    .str.strip().str.title() + ', ' +
    df_nri[NRI_STATE_COL].str.strip().str.title()
)

# Standardize Image table: combine county + state as join key
# The format will match: 'County Name, State Name'
df_images['join_key'] = (
    df_images['county'].str.replace(' County', '', regex=False) # Remove ' County' from image data
    .str.strip().str.title() + ', ' +
    df_images['state'].str.strip().str.title()
)

# Select only the join key and wildfire score from NRI, drop duplicates
df_nri_final = df_nri[['join_key', NRI_PROB_COL]].rename(columns={NRI_PROB_COL: NEW_COLUMN_NAME})
df_nri_final = df_nri_final.drop_duplicates(subset=['join_key'])


# --- 4. Merge (Updated to use 'join_key' from both tables) ---
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
# Note: os.path.abspath(OUTPUT_FILE) is for running in a real environment
print(f"Output saved to: {OUTPUT_FILE}") 
print("\nSample Data (first 5 rows):")
print(df_merged.head())