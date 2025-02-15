import os
import pandas as pd
from datetime import datetime

# Define file paths
input_csv = "/Users/giovi/Documents/StudioBereikbaar/SB455 - Impact assessment of accessibility/education_roland_manus/data/analysis/education_locations.csv"
output_csv = "/Users/giovi/Documents/StudioBereikbaar/SB455 - Impact assessment of accessibility/education_roland_manus/data/analysis/education_locations_fixed.csv"

# ✅ Step 1: Check if file exists
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"❌ Error: The file '{input_csv}' does not exist. Check the file path.")

print(f"✅ File found: {input_csv}")

# ✅ Step 2: Load the CSV file
df = pd.read_csv(input_csv, dtype=str)  # Load all columns as strings

print(f"✅ Loaded {len(df)} rows from {input_csv}")

# ✅ Step 3: Function to convert date format
def convert_date_format(date_str):
    """Convert date from DD-MM-YYYY to YYYY-MM-DD format."""
    if pd.isna(date_str) or date_str.strip() == "":
        return None  # Keep NULL values intact
    try:
        return datetime.strptime(date_str, "%d-%m-%Y").strftime("%Y-%m-%d")
    except ValueError:
        return date_str  # Return original value if conversion fails

# ✅ Step 4: Convert date columns if they exist
if "start_date" in df.columns:
    df["start_date"] = df["start_date"].apply(convert_date_format)

if "end_date" in df.columns:
    df["end_date"] = df["end_date"].apply(convert_date_format)

print("✅ Date columns formatted correctly.")

# ✅ Step 5: Save the fixed CSV
df.to_csv(output_csv, index=False)

print(f"✅ CSV fixed and saved as: {output_csv}")