#!/usr/bin/env python3
import json
import os
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from time import sleep
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry  # type: ignore
from collections import defaultdict
import sys
from datetime import datetime

##############################
# Advanced Geocoding Classes #
##############################

class GeocodingQualityMetrics:
    """Track and analyze geocoding quality metrics"""
    
    def __init__(self):
        self.total_addresses = 0
        self.successful_geocodes = 0
        self.failed_geocodes = 0
        self.quality_scores = []
        self.method_counts = defaultdict(int)
        self.error_types = defaultdict(int)
        self.processing_times = []
        self.postcode_success_rate = defaultdict(lambda: {'success': 0, 'total': 0})
        
    def record_attempt(self, success: bool, method: str, quality_score: float,
                      processing_time: float, postcode: str = None,
                      error_type: str = None):
        self.total_addresses += 1
        if success:
            self.successful_geocodes += 1
            self.quality_scores.append(quality_score)
            self.method_counts[method] += 1
            if postcode:
                self.postcode_success_rate[postcode]['success'] += 1
        else:
            self.failed_geocodes += 1
            if error_type:
                self.error_types[error_type] += 1
        if postcode:
            self.postcode_success_rate[postcode]['total'] += 1
        self.processing_times.append(processing_time)
    
    def generate_report(self) -> Dict[str, Any]:
        success_rate = (self.successful_geocodes / self.total_addresses * 100 
                        if self.total_addresses > 0 else 0)
        avg_quality = (sum(self.quality_scores) / len(self.quality_scores) 
                       if self.quality_scores else 0)
        avg_processing_time = (sum(self.processing_times) / len(self.processing_times) 
                               if self.processing_times else 0)
        problematic_postcodes = [
            {'postcode': pc, 'stats': stats}
            for pc, stats in self.postcode_success_rate.items()
            if stats['success'] / stats['total'] < 0.5 and stats['total'] >= 5
        ]
        return {
            'summary': {
                'total_addresses': self.total_addresses,
                'successful_geocodes': self.successful_geocodes,
                'failed_geocodes': self.failed_geocodes,
                'success_rate_percent': round(success_rate, 2),
                'average_quality_score': round(avg_quality, 2),
                'average_processing_time_ms': round(avg_processing_time * 1000, 2)
            },
            'method_distribution': dict(self.method_counts),
            'error_analysis': dict(self.error_types),
            'quality_distribution': {
                'min_quality': min(self.quality_scores) if self.quality_scores else 0,
                'max_quality': max(self.quality_scores) if self.quality_scores else 0,
                'quality_quartiles': (
                    pd.Series(self.quality_scores).quantile([0.25, 0.5, 0.75]).to_dict()
                    if self.quality_scores else {}
                )
            },
            'problematic_areas': {
                'postcodes_with_low_success': problematic_postcodes,
                'common_error_patterns': {
                    error: count 
                    for error, count in self.error_types.items() 
                    if count >= 5
                }
            }
        }

class LocalNominatimGeocoder:
    """Enhanced geocoder with quality metrics and robust error handling"""
    
    def __init__(
        self,
        nominatim_url: str = "http://localhost:8080",
        max_retries: int = 2,
        timeout: int = 5,
        workers: int = 12,
        rate_limit: float = 0.1
    ):
        self.nominatim_url = nominatim_url.rstrip('/')
        self.timeout = timeout
        self.workers = workers
        self.rate_limit = rate_limit
        self.quality_metrics = GeocodingQualityMetrics()
        
        self.setup_logging()
        self.session = self.setup_session(max_retries)
        self._test_connection()
    
    def setup_logging(self):
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'geocoding_{timestamp}.log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_session(self, max_retries: int) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _test_connection(self) -> None:
        try:
            response = self.session.get(
                f"{self.nominatim_url}/status.php",
                timeout=self.timeout
            )
            response.raise_for_status()
            self.logger.info("Successfully connected to local Nominatim instance")
        except Exception as e:
            self.logger.error(f"Failed to connect to Nominatim: {str(e)}")
            raise ConnectionError("Could not connect to local Nominatim instance")

    def clean_address(self, address: str) -> str:
        address = address.replace('.0', '')
        abbreviations = {
            'str.': 'straat',
            'ln.': 'laan',
            'ln': 'laan',
            'str': 'straat',
            'Kon.': 'Koningin',
            'Dr.': 'Doctor',
            'St.': 'Sint',
            'Prof.': 'Professor',
            'Jr.': 'Junior',
            'Sr.': 'Senior',
            'v.': 'van',
            'v/d': 'van de',
            'pr.': 'prins',
            'bs.': 'basisschool',
            'obs.': 'openbare basisschool'
        }
        for abbr, full in abbreviations.items():
            address = address.replace(f' {abbr} ', f' {full} ')
        address = address.replace('.', ' ').replace('  ', ' ').strip()
        return address

    def format_house_number(self, number: Any, addition: Optional[str] = None) -> str:
        try:
            if pd.isna(number) or number is None:
                self.logger.debug(f"Missing house number, using default")
                return "1"
            float_num = float(str(number).replace(',', '.'))
            if pd.isna(float_num):
                return "1"
            base_number = str(int(float_num))
            if addition and not pd.isna(addition):
                addition = str(addition).strip().strip('-').strip()
                if addition and addition.lower() != 'nan':
                    return f"{base_number}-{addition}"
            return base_number
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error formatting house number {number}: {str(e)}")
            return "1"

    def geocode_with_postcode(self, postcode: str, house_number: str, start_time: float) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            postcode = postcode.replace(" ", "").upper()
            query = f"{postcode} {house_number}, Netherlands"
            params = {
                'q': query,
                'format': 'json',
                'addressdetails': 1,
                'limit': 1,
                'countrycodes': 'nl'
            }
            response = self.session.get(
                f"{self.nominatim_url}/search.php",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            results = response.json()
            if results:
                result = results[0]
                return {
                    'lat': float(result['lat']),
                    'lon': float(result['lon']),
                    'confidence': float(result.get('importance', 0)),
                    'method': 'postcode',
                    'raw_response': result
                }, None
            return None, "no_results"
        except Exception as e:
            self.logger.debug(f"Postcode geocoding failed: {str(e)}")
            return None, str(e)

    def geocode_with_street(self, street: str, house_number: str, city: str, start_time: float) -> Tuple[Optional[Dict], Optional[str]]:
        street = self.clean_address(street)
        city = self.clean_address(city)
        queries = [
            f"{street} {house_number}, {city}, Netherlands",
            f"{street} {house_number}, {city}",
            f"{street} {house_number}, Netherlands"
        ]
        for query in queries:
            try:
                params = {
                    'q': query,
                    'format': 'json',
                    'addressdetails': 1,
                    'limit': 1,
                    'countrycodes': 'nl'
                }
                response = self.session.get(
                    f"{self.nominatim_url}/search.php",
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                results = response.json()
                if results:
                    result = results[0]
                    return {
                        'lat': float(result['lat']),
                        'lon': float(result['lon']),
                        'confidence': float(result.get('importance', 0)),
                        'method': 'street',
                        'raw_response': result
                    }, None
            except Exception as e:
                self.logger.debug(f"Street geocoding failed for {query}: {str(e)}")
                continue
        return None, "no_results"

    def geocode_address(self, address: Dict, inst_id: str = "") -> Optional[Dict]:
        start_time = datetime.now().timestamp()
        # Expecting address to have: 'postcode', 'number', 'street', 'city'
        if not all(key in address for key in ['postcode', 'number', 'street', 'city']):
            self.quality_metrics.record_attempt(
                False, '', 0.0, datetime.now().timestamp() - start_time,
                error_type='incomplete_address'
            )
            return None
        try:
            house_number = self.format_house_number(address['number'], address.get('number_addition'))
            result, error = self.geocode_with_postcode(address['postcode'], house_number, start_time)
            if result:
                self.quality_metrics.record_attempt(
                    True, 'postcode', result['confidence'],
                    datetime.now().timestamp() - start_time,
                    address['postcode']
                )
                return result
            sleep(self.rate_limit)
            result, error = self.geocode_with_street(address['street'], house_number, address['city'], start_time)
            if result:
                self.quality_metrics.record_attempt(
                    True, 'street', result['confidence'],
                    datetime.now().timestamp() - start_time
                )
                return result
            self.quality_metrics.record_attempt(
                False, '', 0.0,
                datetime.now().timestamp() - start_time,
                address['postcode'],
                error
            )
            return None
        except Exception as e:
            self.logger.error(f"Geocoding failed for {address}: {str(e)}")
            self.quality_metrics.record_attempt(
                False, '', 0.0,
                datetime.now().timestamp() - start_time,
                error_type=str(e)
            )
            return None

###############################
# Helper: Build Address Dict  #
###############################

def build_address_dict(record: Dict) -> Optional[Dict]:
    """
    Build an address dictionary from a record.
    Uses physical address fields if available; otherwise, falls back to correspondence.
    Expected keys for geocoding: 'street', 'number', 'postcode', 'city', and optionally 'number_addition'.
    """
    # Use physical (vest) address if available
    if record.get('NAAM_STRAAT_VEST', '').strip() and record.get('POSTCODE_VEST', '').strip() and record.get('NAAM_PLAATS_VEST', '').strip():
        return {
            'street': record.get('NAAM_STRAAT_VEST', '').strip(),
            'number': record.get('NR_HUIS_VEST', '').strip(),
            'postcode': record.get('POSTCODE_VEST', '').strip(),
            'city': record.get('NAAM_PLAATS_VEST', '').strip(),
            'number_addition': record.get('NR_HUIS_TOEV_VEST', '').strip() or None
        }
    # Fallback to correspondence address
    elif record.get('NAAM_STRAAT_CORR', '').strip() and record.get('POSTCODE_CORR', '').strip() and record.get('NAAM_PLAATS_CORR', '').strip():
        return {
            'street': record.get('NAAM_STRAAT_CORR', '').strip(),
            'number': record.get('NR_HUIS_CORR', '').strip(),
            'postcode': record.get('POSTCODE_CORR', '').strip(),
            'city': record.get('NAAM_PLAATS_CORR', '').strip(),
            'number_addition': record.get('NR_HUIS_TOEV_CORR', '').strip() or None
        }
    else:
        return None

##################################
# Process a Single Record        #
##################################

def process_record(record: Dict, geocoder: LocalNominatimGeocoder) -> Dict:
    """
    Process a single record: build its address dictionary, geocode it,
    and update the record with geocoding results.
    """
    inst_id = record.get('\ufeffNR_ADMINISTRATIE', 'unknown')
    address = build_address_dict(record)
    if not address:
        geocoder.logger.info(f"Record {inst_id}: No valid address found.")
        record["latitude"] = None
        record["longitude"] = None
        record["geocode_method"] = None
        record["confidence"] = None
        return record
    result = geocoder.geocode_address(address, inst_id)
    if result:
        record["latitude"] = result['lat']
        record["longitude"] = result['lon']
        record["geocode_method"] = result['method']
        record["confidence"] = result['confidence']
        geocoder.logger.info(f"Record {inst_id}: Geocoded '{address['street']} {address['number']}, {address['postcode']} {address['city']}' to ({result['lat']}, {result['lon']})")
    else:
        record["latitude"] = None
        record["longitude"] = None
        record["geocode_method"] = None
        record["confidence"] = None
        geocoder.logger.info(f"Record {inst_id}: Could not geocode address: '{address}'")
    return record

###############################
# Main Processing Function    #
###############################

def main():
    input_file = os.path.join("data", "processed", "education_data_all_intermediate.json")
    output_file = os.path.join("data", "processed", "education_data_all_geocoded.geojson")
    
    # Load intermediate JSON data
    with open(input_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    
    # Initialize the advanced geocoder (adjust nominatim_url if needed)
    geocoder = LocalNominatimGeocoder(nominatim_url="http://localhost:8080", workers=12, rate_limit=0.1)
    
    # Process records in parallel
    updated_records = []
    with ThreadPoolExecutor(max_workers=geocoder.workers) as executor:
        futures = {executor.submit(process_record, record, geocoder): record for record in records}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Geocoding records"):
            try:
                updated_record = future.result()
                updated_records.append(updated_record)
            except Exception as e:
                geocoder.logger.error(f"Error processing record: {str(e)}")
    
    # Build GeoJSON features
    features = []
    for rec in updated_records:
        if rec.get("latitude") is not None and rec.get("longitude") is not None:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [rec["longitude"], rec["latitude"]]
                },
                "properties": rec
            }
        else:
            feature = {
                "type": "Feature",
                "geometry": None,
                "properties": rec
            }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=4)
    
    geocoder.logger.info(f"Created geocoded GeoJSON file: {output_file}")
    # Optionally, print the quality report
    report = geocoder.quality_metrics.generate_report()
    geocoder.logger.info("Geocoding Quality Report:")
    geocoder.logger.info(json.dumps(report, indent=4))
    
if __name__ == "__main__":
    main()