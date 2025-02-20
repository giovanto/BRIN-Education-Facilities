#!/usr/bin/env python3
"""
Script: 02_geocode_addresses.py
Purpose: Geocode institution addresses with enhanced quality metrics and house number handling
"""

import json
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
        """Record a geocoding attempt"""
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
        """Generate comprehensive quality report"""
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
        
        # Configure logging
        self.setup_logging()
        
        # Configure session
        self.session = self.setup_session(max_retries)
        
        # Test connection
        self._test_connection()
    
    def setup_logging(self):
        """Setup detailed logging configuration"""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'geocoding_{timestamp}.log'
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def setup_session(self, max_retries: int) -> requests.Session:
        """Setup session with retry strategy"""
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
        """Test connection to local Nominatim instance"""
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
        """Clean address string for better geocoding results"""
        # Remove decimal points from house numbers
        address = address.replace('.0', '')
        
        # Expand common Dutch abbreviations
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
        
        # Clean up whitespace and periods
        address = address.replace('.', ' ').replace('  ', ' ').strip()
        
        return address

    def format_house_number(self, number: Any, addition: Optional[str] = None) -> str:
        """Format house number with enhanced error handling"""
        try:
            # Handle NaN/None values
            if pd.isna(number) or number is None:
                self.logger.debug(f"Missing house number, using default")
                return "1"  # Default value for missing numbers
            
            # Convert to float first to handle string inputs
            float_num = float(str(number).replace(',', '.'))
            if pd.isna(float_num):
                return "1"
                
            base_number = str(int(float_num))
            
            # Handle addition
            if addition and not pd.isna(addition):
                addition = str(addition).strip().strip('-').strip()
                if addition and addition.lower() != 'nan':
                    return f"{base_number}-{addition}"
            
            return base_number
            
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Error formatting house number {number}: {str(e)}")
            return "1"

    def geocode_with_postcode(
        self,
        postcode: str,
        house_number: str,
        start_time: float
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Geocode using postcode and house number"""
        try:
            # Clean and format postcode
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

    def geocode_with_street(
        self,
        street: str,
        house_number: str,
        city: str,
        start_time: float
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """Geocode using street name and house number"""
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

    def geocode_address(
        self,
        address: Dict,
        inst_id: str = "",
        location_type: str = 'main'
    ) -> Optional[Dict]:
        """Geocode a single address with quality metrics"""
        start_time = datetime.now().timestamp()
        
        if not all(key in address for key in ['postcode', 'number', 'street', 'city']):
            self.quality_metrics.record_attempt(
                False, '', 0.0, 
                datetime.now().timestamp() - start_time,
                error_type='incomplete_address'
            )
            return None
        
        try:
            # Format house number
            house_number = self.format_house_number(
                address['number'],
                address.get('number_addition')
            )
            
            # Try postcode first (most reliable for NL)
            result, error = self.geocode_with_postcode(
                address['postcode'],
                house_number,
                start_time
            )
            
            if result:
                self.quality_metrics.record_attempt(
                    True, 'postcode', result['confidence'],
                    datetime.now().timestamp() - start_time,
                    address['postcode']
                )
                return result
            
            # Fallback to street method
            sleep(self.rate_limit)
            result, error = self.geocode_with_street(
                address['street'],
                house_number,
                address['city'],
                start_time
            )
            
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

    def batch_geocode(self, addresses: List[Dict]) -> List[Dict]:
        """Geocode multiple addresses with parallel processing"""
        results = []
        
        # Group by postcode for cache optimization
        postcode_groups = defaultdict(list)
        for addr in addresses:
            postcode = str(addr.get('postcode', '')).replace(' ', '').upper()
            if postcode:
                postcode_groups[postcode].append(addr)
            else:
                self.logger.warning(f"Missing postcode for address: {addr}")
        
        total_groups = len(postcode_groups)
        self.logger.info(f"Processing {total_groups} postcode groups")
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = []
            
            with tqdm(total=total_groups, desc="Geocoding by postcode") as pbar:
                for postcode, addr_group in postcode_groups.items():
                    future = executor.submit(
                        self._process_postcode_group,
                        postcode,
                        addr_group
                    )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Error processing batch: {str(e)}")
        
        return results

    def _process_postcode_group(self, postcode: str, addresses: List[Dict]) -> List[Dict]:
        """Process a group of addresses with the same postcode"""
        results = []
        start_time = datetime.now().timestamp()
        
        try:
            # First, try a single geocoding request for the postcode
            base_location, error = self.geocode_with_postcode(postcode, '', start_time)
            
            if base_location:
                # Use base location as reference and adjust for house numbers
                base_lat = float(base_location['lat'])
                base_lon = float(base_location['lon'])
                
                for addr in addresses:
                    try:
                        # Enhanced number handling
                        raw_number = addr.get('number')
                        if pd.isna(raw_number):
                            self.logger.debug(f"Missing house number for address: {addr}")
                            continue
                        
                        house_number = self.format_house_number(
                            raw_number,
                            addr.get('number_addition')
                        )
                        
                        # Skip invalid house numbers
                        if house_number == "1" and raw_number != 1:
                            continue
                        
                        # Adjust coordinates slightly based on house number
                        base_number = int(house_number.split('-')[0])
                        adjusted_location = {
                            'lat': base_lat + (base_number * 0.0001),
                            'lon': base_lon,
                            'confidence': base_location['confidence'] * 0.9,  # Reduce confidence for approximated locations
                            'method': 'postcode_batch',
                            'raw_response': base_location['raw_response']
                        }
                        
                        results.append({
                            'address': addr,
                            'geocode': adjusted_location,
                            'approximated': True  # Flag for approximated locations
                        })
                        
                        self.quality_metrics.record_attempt(
                            True, 'postcode_batch',
                            adjusted_location['confidence'],
                            datetime.now().timestamp() - start_time,
                            postcode
                        )
                        
                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Error processing address in batch: {str(e)}")
                        self.quality_metrics.record_attempt(
                            False, '', 0.0,
                            datetime.now().timestamp() - start_time,
                            postcode,
                            str(e)
                        )
                        continue
            else:
                # Fallback to individual geocoding if postcode not found
                for addr in addresses:
                    result = self.geocode_address(addr)
                    if result:
                        results.append({
                            'address': addr,
                            'geocode': result,
                            'approximated': False
                        })
        except Exception as e:
            self.logger.error(f"Error processing postcode group {postcode}: {str(e)}")
        
        return results

    def process_education_data(
        self,
        input_file: Path,
        output_json_file: Path,
        output_geojson_file: Path
    ) -> None:
        """Process education data with optimized batch geocoding"""
        # Load input data
        self.logger.info(f"Loading data from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Collect all addresses
        all_addresses = []
        address_mapping = {}
        
        for edu_type, edu_data in data.items():
            for inst in edu_data['institutions']:
                # Main address
                if 'main_address' in inst:
                    addr_id = f"main_{inst['institution_id']}"
                    all_addresses.append({
                        'id': addr_id,
                        **inst['main_address']
                    })
                    address_mapping[addr_id] = ('main', inst)
                
                # Branch addresses
                if 'branches' in inst:
                    for branch in inst['branches']:
                        if 'address' in branch:
                            addr_id = f"branch_{inst['institution_id']}_{branch['branch_id']}"
                            all_addresses.append({
                                'id': addr_id,
                                **branch['address']
                            })
                            address_mapping[addr_id] = ('branch', branch)
        
        # Process addresses in batches
        self.logger.info(f"Processing {len(all_addresses)} addresses in batches")
        geocoded_results = self.batch_geocode(all_addresses)
        
        # Update original data structure
        for result in geocoded_results:
            if not result.get('geocode'):
                continue
            
            addr_id = result['address']['id']
            location_type, target = address_mapping[addr_id]
            
            # Store geocoding result
            target['geocode'] = result['geocode']
            if result.get('approximated'):
                target['geocode']['approximated'] = True
        
        # Generate quality report
        quality_report = self.quality_metrics.generate_report()
        
        # Save quality report
        quality_report_file = output_json_file.parent / 'geocoding_quality_report.json'
        with open(quality_report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Quality report saved to {quality_report_file}")
        
        # Log quality summary
        self.logger.info("Geocoding Quality Summary:")
        self.logger.info(f"Total addresses processed: {quality_report['summary']['total_addresses']}")
        self.logger.info(f"Successful geocodes: {quality_report['summary']['successful_geocodes']}")
        self.logger.info(f"Success rate: {quality_report['summary']['success_rate_percent']}%")
        self.logger.info(f"Average quality score: {quality_report['summary']['average_quality_score']}")
        
        # Save results
        self.logger.info(f"Saving geocoded JSON to {output_json_file}")
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Convert to GeoJSON
        self.logger.info("Converting to GeoJSON")
        geojson_data = self.convert_to_geojson(data)
        
        self.logger.info(f"Saving GeoJSON to {output_geojson_file}")
        with open(output_geojson_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)
    
    def convert_to_geojson(self, data: Dict) -> Dict:
        """Convert education data to GeoJSON format"""
        features = []
        
        for edu_type, edu_data in data.items():
            for inst in edu_data['institutions']:
                # Add main location if geocoded
                if 'geocode' in inst:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                inst['geocode']['lon'],
                                inst['geocode']['lat']
                            ]
                        },
                        "properties": {
                            "institution_id": inst['institution_id'],
                            "name": inst['name'],
                            "type": edu_type,
                            "location_type": "main",
                            "address": inst['main_address'],
                            "geocoding_method": inst['geocode'].get('method'),
                            "confidence": inst['geocode'].get('confidence'),
                            "approximated": inst['geocode'].get('approximated', False)
                        }
                    })
                
                # Add branch locations if geocoded
                if 'branches' in inst:
                    for branch in inst['branches']:
                        if 'geocode' in branch:
                            features.append({
                                "type": "Feature",
                                "geometry": {
                                    "type": "Point",
                                    "coordinates": [
                                        branch['geocode']['lon'],
                                        branch['geocode']['lat']
                                    ]
                                },
                                "properties": {
                                    "institution_id": f"{inst['institution_id']}_{branch['branch_id']}",
                                    "name": f"{inst['name']} (Branch)",
                                    "type": edu_type,
                                    "location_type": "branch",
                                    "address": branch['address'],
                                    "geocoding_method": branch['geocode'].get('method'),
                                    "confidence": branch['geocode'].get('confidence'),
                                    "approximated": branch['geocode'].get('approximated', False)
                                }
                            })
        
        return {
            "type": "FeatureCollection",
            "features": features
        }

def main():
    """Main execution function"""
    # Configure paths
    project_root = Path(__file__).parent.parent
    processed_data_dir = project_root / 'data' / 'processed'
    
    input_file = processed_data_dir / 'education_data_all.json'
    output_json_file = processed_data_dir / 'education_data_all_geocoded.json'
    output_geojson_file = processed_data_dir / 'education_data_all.geojson'
    
    # Create processed data directory if it doesn't exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize geocoder with optimized settings for local Nominatim
    geocoder = LocalNominatimGeocoder(
        nominatim_url="http://localhost:8080",
        max_retries=2,      # Reduced for faster failure
        timeout=5,          # Lower timeout
        workers=12,         # Match core count
        rate_limit=0.1      # Reduced for local instance
    )
    
    # Process data
    geocoder.process_education_data(
        input_file,
        output_json_file,
        output_geojson_file
    )

if __name__ == "__main__":
    main()