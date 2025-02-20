#!/usr/bin/env python3
import os
import csv
import json

def read_raw_csv(file_path):
    """Read a CSV file and return a list of dictionaries."""
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def main():
    raw_dir = os.path.join("data", "raw")
    output_file = os.path.join("data", "processed", "education_data_all_intermediate.json")
    all_data = []
    
    # Process each CSV file from the raw directory
    for filename in os.listdir(raw_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(raw_dir, filename)
            print(f"Processing {file_path}...")
            data = read_raw_csv(file_path)
            # --- Removed main/branch location logic here ---
            all_data.extend(data)
    
    # Save the complete, unmodified data to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)
    
    print(f"Created intermediate JSON file: {output_file}")

    # Print the first 10 entries of the output file
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data[:10]:
            print(entry)
            
if __name__ == "__main__":
    main()