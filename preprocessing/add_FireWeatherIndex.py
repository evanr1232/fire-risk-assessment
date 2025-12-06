from scipy.io import netcdf_file
import numpy as np
import pandas as pd



# --- Configuration ---
NETCDF_FILE = "data/FWI_GEOS5.nc"
CSV_FILE = "data/NRI_Table_Counties/image_county_data_with_burnprob.csv"
FWI_VAR_NAME = 'GEOS-5_FWI'
NEW_COLUMN_NAME = 'Max_FWRI_2020'


def extract_fwi_and_merge(netcdf_file_path: str, csv_file_path: str) -> pd.DataFrame:
    """
    1. Loads NetCDF data and calculates the maximum FWI at each grid point in 2020.
    2. Loads the CSV file.
    3. Finds the nearest NetCDF grid value for each lat/lon in the CSV.
    4. Merges the extracted FWI values into the DataFrame.
    """
    
    # --- 1. Load and Process NetCDF Data ---

    try:
        with netcdf_file(netcdf_file_path, 'r') as file2read:

            # Extract data arrays

            fwi_data = file2read.variables[FWI_VAR_NAME][:].copy()
            lat = file2read.variables['lat'][:].copy()
            lon = file2read.variables['lon'][:].copy()

            print(lat[:10])
            print(lon[:10])



    except FileNotFoundError:
        print(f"Error: NetCDF file '{netcdf_file_path}' not found.")
        return pd.DataFrame()

    except KeyError as e:
        print(f"Error: Required variable {e} not found in NetCDF file.")
        return pd.DataFrame()

    print("NetCDF data loaded successfully.")


    # ASSUMPTION: The entire fwi_data array is for the year 2020.
    # If not, time slicing (fwi_data[start_idx:end_idx, :, :]) must be done here.
    fwi_2020 = fwi_data


    # Calculate the maximum FWI at each (lat, lon) location (collapse the time dimension)
    # This results in a 2D array (lat, lon)
    max_spatial_fwi = np.nanmax(fwi_2020, axis=0)



    # --- 2. Load CSV Data ---
    try:
        df = pd.read_csv(csv_file_path)

    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file_path}' not found.")
        return pd.DataFrame()

    print(f"CSV data loaded. Processing {len(df)} locations...")



    # --- 3. Mapping and Extraction ---

    # Find the indices of the nearest grid cell for each lat/lon point
    # np.argmin and np.abs find the index where the difference is smallest.
    

    def get_nearest_fwi(row):
        # Find the index of the nearest latitude in the NetCDF grid
        lat_idx = np.argmin(np.abs(lat - row['latitude']))

        # Find the index of the nearest longitude in the NetCDF grid
        lon_idx = np.argmin(np.abs(lon - row['longitude']))

        return max_spatial_fwi[lat_idx, lon_idx]

    # Apply the function to the DataFrame to create the new column

    # This is the most time-consuming step

    df[NEW_COLUMN_NAME] = df.apply(get_nearest_fwi, axis=1)

    return df

# --- Execution ---

df_augmented = extract_fwi_and_merge(NETCDF_FILE, CSV_FILE)

if not df_augmented.empty:

    output_file_name = CSV_FILE.replace('.csv', f'_with_{NEW_COLUMN_NAME}.csv')
    df_augmented.to_csv(output_file_name, index=False)
    print("\nProcessing Complete.")
    print(f"Augmented data saved to: {output_file_name}")
    print("\nSample of new DataFrame:")
    print(df_augmented[['filename', 'latitude', 'longitude', NEW_COLUMN_NAME]].head())