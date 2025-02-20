#!/usr/bin/env python3
import os
import argparse
import geopandas as gpd
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Iterative spatial matching between geocoded education data and reference dataset."
    )
    parser.add_argument(
        "--max_distance",
        type=float,
        default=None,
        help="Optional maximum allowed distance (in meters) for a valid match."
    )
    args = parser.parse_args()
    
    # Define file paths
    education_geojson_path = os.path.join("data", "processed", "education_data_all_geocoded.geojson")
    reference_geojson_path = os.path.join("data", "reference", "onderwijs_basis.geojson")
    output_csv_path = os.path.join("data", "processed", "matching_results_iterative.csv")
    
    # Load the geocoded education data and the reference dataset
    edu_gdf = gpd.read_file(education_geojson_path)
    ref_gdf = gpd.read_file(reference_geojson_path)
    
    # Reproject both to a metric CRS for accurate distance calculations (EPSG:28992 is common for NL)
    edu_gdf = edu_gdf.to_crs(epsg=28992)
    ref_gdf = ref_gdf.to_crs(epsg=28992)
    
    # Remove any records with missing geometry in the education dataset
    edu_gdf = edu_gdf[edu_gdf.geometry.notnull()]
    
    # Perform a nearest-neighbor spatial join.
    # The reference dataset’s geometry will be appended with a suffix (typically "geometry_right").
    joined = gpd.sjoin_nearest(edu_gdf, ref_gdf, how="left", distance_col="distance")
    
    # If a maximum allowed distance was specified, filter the results
    if args.max_distance is not None:
        initial_count = len(joined)
        joined = joined[joined["distance"] <= args.max_distance]
        print(f"Filtered {initial_count - len(joined)} records with distance > {args.max_distance} m.")
    
    # Attempt to detect an identifier in the education data – for example, the 'NR_ADMINISTRATIE' field.
    id_field = None
    for col in edu_gdf.columns:
        if "NR_ADMINISTRATIE" in col:
            id_field = col
            break
    if id_field is None:
        id_field = "index"
    
    # If sjoin_nearest did not append the reference geometry, create a placeholder.
    if "geometry_right" not in joined.columns:
        joined["geometry_right"] = None
    
    # Prepare output DataFrame. We'll include:
    # - education identifier
    # - computed distance in meters
    # - education geometry (converted to WKT for CSV)
    # - reference geometry (WKT)
    output_df = joined[[id_field, "distance", "geometry", "geometry_right"]].copy()
    output_df = output_df.rename(columns={
        id_field: "education_id",
        "distance": "distance_m",
        "geometry": "education_geometry",
        "geometry_right": "reference_geometry"
    })
    
    # Convert geometries to Well-Known Text (WKT) for CSV export
    output_df["education_geometry"] = output_df["education_geometry"].apply(
        lambda geom: geom.wkt if geom is not None else None
    )
    output_df["reference_geometry"] = output_df["reference_geometry"].apply(
        lambda geom: geom.wkt if geom is not None else None
    )
    
    # Save the matching results
    output_df.to_csv(output_csv_path, index=False)
    print(f"Matching results saved to: {output_csv_path}")

if __name__ == "__main__":
    main()