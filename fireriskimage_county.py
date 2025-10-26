import os
import re
from pathlib import Path
import contus
import pandas as pd

# --- 1. Configuration and Setup ---

# Define the root directory (adjust this path if your 'data' folder is elsewhere)
ROOT_DIR = Path('data/firerisk')

# Define the folders to process
SUB_FOLDERS = ['train', 'val']

# Define the regex pattern to reliably capture the coordinates from the filename
# Example: 27032281_4_-103.430441201095_44.2804260315038.png
# The pattern captures the Longitude and Latitude values.
COORDINATE_PATTERN = re.compile(r'_([-+]?\d+\.\d+)_([-+]?\d+\.\d+)\.png$')

# List to store dictionaries of results
results = []

# --- 2. Helper Functions ---

def extract_coords_from_filename(filename: str) -> tuple[float, float] | None:
    """
    Extracts the longitude and latitude from the filename.
    Returns (latitude, longitude) as floats for direct use with contus.
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
    Uses the contus library to reverse geocode coordinates to a county name.
    """
    try:
        # contus.get_fips requires (latitude, longitude)
        fips_data = contus.get_fips(lat, lon)
        
        # contus returns a list of matching geographies. We take the first one.
        if fips_data:
            county = fips_data[0]['county_name']
            state = fips_data[0]['state_name']
            return f"{county}, {state}"
        else:
            return "County Not Found (Outside US/Territories)"
    except Exception:
        # Handle API or connection errors gracefully
        return "ERROR: Contus lookup failed"

# --- 3. Main Processing Loop ---

print(f"Starting processing in root directory: {ROOT_DIR}\n")

for folder_name in SUB_FOLDERS:
    current_path = ROOT_DIR / folder_name
    print(f"Processing folder: {current_path.name.upper()}...")

    # Recursively iterate through files in the current directory and its subdirectories
    for file_path in current_path.rglob('*.png'):
        filename = file_path.name
        
        # 1. Extract Coordinates
        coords = extract_coords_from_filename(filename)
        
        if coords:
            lat, lon = coords
            
            # 2. Get County
            county_info = get_county_from_coords(lat, lon)
            
            # 3. Store Result
            results.append({
                'folder': folder_name,
                'filename': filename,
                'latitude': lat,
                'longitude': lon,
                'county': county_info
            })
            # print(f"  {filename} -> {county_info}") # Uncomment for verbose output
        else:
            print(f"  Skipped: {filename} - Coordinates not found in name.")

# --- 4. Generate CSV Output ---

# Convert the list of results into a pandas DataFrame
df = pd.DataFrame(results)

# Define the output file name
output_file = 'image_county_data.csv'

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)

print("\n--- Processing Complete ---")
print(f"Total images processed: {len(df)}")
print(f"Output saved to: {os.path.abspath(output_file)}")
print("\nSample Data:")
print(df.head())