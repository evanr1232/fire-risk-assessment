import pandas as pd
import os

# --- Configuration ---
IMAGE_DATA_FILE = 'image_county_data.csv'
NRI_DATA_FILE = 'data/NRI_Table_Counties/NRI_Table_Counties.csv'
OUTPUT_FILE = 'image_county_data_with_burnprob.csv'

# Based on the sample data, these are the columns to use:
NRI_COUNTY_COL = 'COUNTY'
NRI_STATE_COL = 'STATE'
# WFIR_RISKS is the Wildfire Risk Score, which is the best proxy for burn probability.
NRI_PROB_COL = 'WFIR_RISKS'
NEW_COLUMN_NAME = 'Wildfire_Risk_Score_NRI' # Clearer column name for the score

print(f"Loading image data from: {IMAGE_DATA_FILE}")

# 1. Load the generated image data CSV
try:
    df_images = pd.read_csv(IMAGE_DATA_FILE)
    # Ensure the column used for joining is present
    if 'county' not in df_images.columns:
        print("Error: 'image_county_data.csv' must contain a 'county' column in the 'County, State' format.")
        exit()
except FileNotFoundError:
    print(f"Error: The file '{IMAGE_DATA_FILE}' was not found. Please ensure the original script ran successfully.")
    exit()

print(f"Loading NRI data from: {NRI_DATA_FILE}")

# 2. Load the NRI data CSV
try:
    df_nri = pd.read_csv(NRI_DATA_FILE)
except FileNotFoundError:
    print(f"Error: The file '{NRI_DATA_FILE}' was not found. Please ensure the name is correct.")
    exit()

# --- 3. Prepare NRI Data for Merging ---

# Check if the required columns exist in the NRI data
required_cols = [NRI_COUNTY_COL, NRI_STATE_COL, NRI_PROB_COL]
if not all(col in df_nri.columns for col in required_cols):
    print("\nERROR: One or more required columns (COUNTY, STATE, WFIR_RISKS) were not found in the NRI file.")
    print(f"Please confirm your NRI file headers match the expected format.")
    exit()

# CREATE THE JOIN KEY: Combine COUNTY and STATE to match the format in df_images ('Autauga, Alabama')
df_nri['join_key'] = (
    df_nri[NRI_COUNTY_COL].astype(str).str.strip() + ', ' +
    df_nri[NRI_STATE_COL].astype(str).str.strip()
)

# Select only the necessary columns from the NRI data and rename the score column
df_nri_final = df_nri[['join_key', NRI_PROB_COL]].rename(
    columns={NRI_PROB_COL: NEW_COLUMN_NAME}
)

# Handle potential duplicate entries for a county (shouldn't happen with NRI data, but good practice)
df_nri_final = df_nri_final.drop_duplicates(subset=['join_key'])

# --- 4. Perform the Merge ---

# Perform a left merge: keep all image rows and add the corresponding risk score
df_merged = pd.merge(
    df_images,
    df_nri_final,
    left_on='county',
    right_on='join_key',
    how='left'
)

# Drop the temporary 'join_key' column
df_merged = df_merged.drop(columns=['join_key'])


# --- 5. Generate CSV Output ---
df_merged.to_csv(OUTPUT_FILE, index=False)

print("\n--- Processing Complete ---")
print(f"Total rows in final data: {len(df_merged)}")
print(f"New column added: '{NEW_COLUMN_NAME}' (from NRI column {NRI_PROB_COL})")
print(f"Output saved to: {os.path.abspath(OUTPUT_FILE)}")
print("\nSample Data (First 5 Rows with new column):")
print(df_merged.head())