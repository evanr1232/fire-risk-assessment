import os
import re
from pathlib import Path
import reverse_geocoder as rg
import pandas as pd
from tqdm import tqdm

# --- 1. Configuration and Setup ---

# Define the root directory (adjust this path if your 'data' folder is elsewhere)
ROOT_DIR = Path('data/firerisk')

# Define the folders to process
SUB_FOLDERS = ['train', 'val']

# Define the regex pattern to reliably capture the coordinates from the filename
# Example: 27032281_4_-103.430441201095_44.2804260315038.png
# The pattern captures the Longitude and Latitude values.
COORDINATE_PATTERN = re.compile(r'_([-+]?\d+(?:\.\d+)?)_([-+]?\d+(?:\.\d+)?)\.png$')

# List to store dictionaries of results
results = []

# --- 2. Helper Functions ---

def extract_coords_from_filename(filename: str) -> tuple[float, float] | None:
    """
    Extracts the longitude and latitude from the filename.
    Returns (latitude, longitude) as floats
    """
    match = COORDINATE_PATTERN.search(filename)
    if match:
        try:
            # Group 1 is Longitude, Group 2 is Latitude (based on your example format)
            lon = float(match.group(1))
            lat = float(match.group(2))
            return lat, lon # Return as (Latitude, Longitude) for contus
        except ValueError:
            return None
    return None

def get_county_from_coords(lat: float, lon: float) -> str:
    """
    Uses the reverse_geocoder library to map coordinates to the nearest city/state.
    """
    try:
        results = rg.search([(lat, lon)])  # must be a list of tuples
        place = results[0]
        return f"{place['name']}, {place['admin1']}, {place['cc']}"
    except Exception as e:
        return f"ERROR: {e}"

# --- 3. Main Processing Loop ---

if __name__ == "__main__":
    print(f"Starting processing in root directory: {ROOT_DIR}\n")

    for folder_name in SUB_FOLDERS:
        current_path = ROOT_DIR / folder_name
        print(f"Processing folder: {current_path.name.upper()}...")

        image_files = list(current_path.rglob('*.png'))
        coords_list = []
        filenames = []

        # Collect coordinates first
        for file_path in tqdm(image_files, desc=f"Reading {folder_name}", unit="file"):
            filename = file_path.name
            coords = extract_coords_from_filename(filename)
            if coords:
                filenames.append(filename)
                coords_list.append(coords)
            else:
                tqdm.write(f"Skipped: {filename} - Coordinates not found in name.")

        # Batch reverse-geocode all valid coordinates at once
        try:
            results_geo = rg.search(coords_list)
        except Exception as e:
            print(f"Batch reverse geocode failed: {e}")
            results_geo = [None] * len(coords_list)

        # Merge results
        for filename, (lat, lon), geo in zip(filenames, coords_list, results_geo):
            if geo:
                place = f"{geo['name']}, {geo['admin1']}, {geo['cc']}"
            else:
                place = "ERROR: reverse geocode failed"
            results.append({
                'folder': folder_name,
                'filename': filename,
                'latitude': lat,
                'longitude': lon,
                'county': place
            })

    # 4. Save CSV
    df = pd.DataFrame(results)
    output_file = 'data/NRI_Table_Counties/image_county_data.csv'
    df.to_csv(output_file, index=False)
    print("\n--- Processing Complete ---")
    print(f"Total images processed: {len(df)}")
    print(f"Output saved to: {os.path.abspath(output_file)}")
    print("\nSample Data:")
    print(df.head())