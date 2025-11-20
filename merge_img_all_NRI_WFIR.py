import pandas as pd
import re

# ---------- Load ----------
image_df = pd.read_csv("data/NRI_Table_Counties/image_county_data.csv", dtype=str)
nri_df   = pd.read_csv("data/NRI_Table_Counties/NRI_table_counties.csv", dtype=str)

# ---------- Clean County Names ----------
def clean_county(name):
    if pd.isna(name): 
        return ""

    name = name.strip()

    # Remove ONLY suffixes like " County", " Parish", etc.
    name = re.sub(r"\s+(County|Parish|Borough|Municipality)$", "", name, flags=re.IGNORECASE)

    return name

def clean_state(name):
    if pd.isna(name):
        return ""
    return name.strip()

image_df["county_clean"] = image_df["county"].apply(clean_county)
image_df["state_clean"]  = image_df["state"].apply(clean_state)

nri_df["COUNTY_clean"] = nri_df["COUNTY"].apply(clean_county)
nri_df["STATE_clean"]  = nri_df["STATE"].apply(clean_state)

# ---------- Select WFIR columns ----------
wfir_cols = [c for c in nri_df.columns if c.startswith("WFIR_") and not c.startswith("WFIR_EVNTS")]

# ----- IMPORTANT: choose only COUNTY rows, exclude CITY rows -----
nri_counties_only = nri_df[nri_df["COUNTYTYPE"].str.lower() == "county"]

# Now uniqueness is guaranteed: one row per (state, county)
nri_unique = nri_counties_only.drop_duplicates(
    subset=["STATE_clean", "COUNTY_clean"], keep="first"
)

# ---------- Merge ----------
merged = image_df.merge(
    nri_unique[["STATE_clean", "COUNTY_clean"] + wfir_cols],
    how="left",
    left_on=["state_clean", "county_clean"],
    right_on=["STATE_clean", "COUNTY_clean"],
    validate="m:1"        # many images → one county row
)

merged = merged.drop(columns=["STATE_clean", "COUNTY_clean"])

# ---------- Save ----------
merged.to_csv("data/NRI_Table_Counties/image_to_all_NRI_WFIR.csv", index=False)
print("Saved image_to_all_NRI_WFIR.csv with shape:", merged.shape)