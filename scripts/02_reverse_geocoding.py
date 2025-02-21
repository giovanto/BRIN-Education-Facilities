"""
Step 2: Reverse Geocoding with optimized local Nominatim settings
Processes validated school locations from onderwijs_basis.geojson
"""

import geopandas as gpd
import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import requests

# Project structure setup
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [PROCESSED_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Nominatim configuration
NOMINATIM_HOST = "localhost"
NOMINATIM_PORT = 8080
BATCH_SIZE = 100  # Process in smaller batches
MAX_WORKERS = 8   # Reduced from 12 to 8 for stability
RATE_LIMIT = 0.05 # 50ms between requests based on test results

def setup_logging() -> logging.Logger:
    """Configure logging with both file and console handlers"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'reverse_geocoding_{timestamp}.log'
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_geojson_data(logger: logging.Logger) -> gpd.GeoDataFrame:
    """Load and validate the input GeoJSON file"""
    try:
        geojson_path = REFERENCE_DIR / "onderwijs_basis.geojson"
        gdf = gpd.read_file(geojson_path)
        
        logger.info(f"Loaded GeoJSON data: {len(gdf)} locations")
        logger.info(f"Columns: {gdf.columns.tolist()}")
        logger.info(f"CRS: {gdf.crs}")
        
        if 'geometry' not in gdf.columns:
            raise ValueError("GeoJSON file missing geometry column")
            
        return gdf
        
    except Exception as e:
        logger.error(f"Error loading GeoJSON: {str(e)}")
        raise

def reverse_geocode_location(coords: Tuple[float, float]) -> Optional[Dict]:
    """
    Reverse geocode a single coordinate pair using direct HTTP request
    """
    url = f"http://{NOMINATIM_HOST}:{NOMINATIM_PORT}/reverse"
    params = {
        "lat": coords[0],
        "lon": coords[1],
        "format": "json",
        "addressdetails": 1,
        "accept-language": "nl"  # Dutch language results
    }
    
    try:
        time.sleep(RATE_LIMIT)  # Rate limiting
        response = requests.get(url, params=params, timeout=5)
        
        if response.status_code == 200:
            return response.json()
        return None
        
    except requests.exceptions.RequestException:
        return None

def process_batch(
    coordinates: List[Tuple[float, float]], 
    start_idx: int,
    logger: logging.Logger
) -> List[Dict]:
    """Process a batch of coordinates"""
    results = []
    
    for i, coords in enumerate(coordinates):
        try:
            result = reverse_geocode_location(coords)
            if result:
                location_data = {
                    'original_index': start_idx + i,
                    'coordinates': {
                        'latitude': coords[0],
                        'longitude': coords[1]
                    },
                    'geocoded_address': result
                }
                results.append(location_data)
            else:
                logger.warning(f"Failed to geocode location {start_idx + i}: {coords}")
                
        except Exception as e:
            logger.error(f"Error processing location {start_idx + i}: {str(e)}")
    
    return results

def process_locations(gdf: gpd.GeoDataFrame, logger: logging.Logger) -> List[Dict]:
    """Process all locations using batched parallel execution"""
    processed_locations = []
    coordinates = [(geom.y, geom.x) for geom in gdf.geometry]
    total_locations = len(coordinates)
    
    logger.info(f"Starting reverse geocoding for {total_locations} locations")
    
    # Process in batches
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        
        for i in range(0, total_locations, BATCH_SIZE):
            batch = coordinates[i:i + BATCH_SIZE]
            future = executor.submit(process_batch, batch, i, logger)
            futures.append(future)
        
        # Process results as they complete
        with tqdm(total=len(futures), desc="Processing batches") as pbar:
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    processed_locations.extend(batch_results)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
    
    return processed_locations

def analyze_results(
    processed_locations: List[Dict],
    total_locations: int,
    logger: logging.Logger
) -> Dict:
    """Analyze the reverse geocoding results"""
    analysis = {
        'total_locations': total_locations,
        'successfully_geocoded': len(processed_locations),
        'failed_geocoding': total_locations - len(processed_locations),
        'success_rate': len(processed_locations) / total_locations * 100,
        'address_completeness': {}
    }
    
    # Analyze address components
    if processed_locations:
        address_fields = set()
        
        # Collect all possible address fields
        for location in processed_locations:
            if 'address' in location['geocoded_address']:
                address_fields.update(location['geocoded_address']['address'].keys())
        
        # Calculate completeness for each field
        for field in address_fields:
            field_count = sum(
                1 for loc in processed_locations
                if 'address' in loc['geocoded_address'] 
                and field in loc['geocoded_address']['address']
            )
            analysis['address_completeness'][field] = {
                'count': field_count,
                'percentage': field_count / len(processed_locations) * 100
            }
    
    return analysis

def main():
    """Main execution function for reverse geocoding"""
    logger = setup_logging()
    logger.info("Starting reverse geocoding process...")
    
    try:
        # Load validated locations
        gdf = load_geojson_data(logger)
        
        # Process locations
        processed_locations = process_locations(gdf, logger)
        
        # Analyze results
        analysis = analyze_results(processed_locations, len(gdf), logger)
        
        # Prepare output
        output = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'settings': {
                    'batch_size': BATCH_SIZE,
                    'max_workers': MAX_WORKERS,
                    'rate_limit_ms': RATE_LIMIT * 1000
                },
                'analysis': analysis
            },
            'locations': processed_locations
        }
        
        # Save results
        output_file = PROCESSED_DIR / 'geocoded_locations.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully saved geocoded data to {output_file}")
        logger.info(f"Success rate: {analysis['success_rate']:.2f}%")
        logger.info("Address completeness:")
        for field, stats in analysis['address_completeness'].items():
            logger.info(f"  {field}: {stats['percentage']:.2f}% ({stats['count']} locations)")
        
    except Exception as e:
        logger.error(f"Error in reverse geocoding process: {str(e)}")
        raise

if __name__ == "__main__":
    main()