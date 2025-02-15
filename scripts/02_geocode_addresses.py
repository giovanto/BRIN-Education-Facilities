import json
import pandas as pd
import requests
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from time import sleep
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry # type: ignore
from collections import defaultdict
import sys
from datetime import datetime

class GeocodingAnalyzer:
    """Analyzer for geocoding results and data quality"""
    
    def __init__(self):
        self.stats = defaultdict(int)
        self.quality_metrics = defaultdict(list)
        self.failures = []
        
    def record_success(self, method: str, confidence: float):
        """Record successful geocoding attempt"""
        self.stats[f'success_{method}'] += 1
        self.quality_metrics[method].append(confidence)
        
    def record_failure(self, institution_id: str, address: Dict, reason: str):
        """Record failed geocoding attempt"""
        self.stats['total_failures'] += 1
        self.failures.append({
            'institution_id': institution_id,
            'address': address,
            'reason': reason
        })
        
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        report = {
            'summary': {
                'total_processed': sum(v for k, v in self.stats.items() if k.startswith('success_')),
                'total_failures': self.stats['total_failures'],
                'success_by_method': {
                    k.replace('success_', ''): v 
                    for k, v in self.stats.items() 
                    if k.startswith('success_')
                }
            },
            'quality_metrics': {
                method: {
                    'mean_confidence': sum(scores)/len(scores) if scores else 0,
                    'min_confidence': min(scores) if scores else 0,
                    'max_confidence': max(scores) if scores else 0
                }
                for method, scores in self.quality_metrics.items()
            },
            'failure_analysis': {
                'total_failures': len(self.failures),
                'failure_samples': self.failures[:10]  # First 10 failures for review
            }
        }
        
        return report

class LocalNominatimGeocoder:
    """Enhanced geocoder with multiple strategies and detailed analysis"""
    
    def __init__(
        self,
        nominatim_url: str = "http://localhost:8080",
        max_retries: int = 3,
        timeout: int = 10,
        workers: int = 4,
        rate_limit: float = 1.0
    ):
        self.nominatim_url = nominatim_url.rstrip('/')
        self.timeout = timeout
        self.workers = workers
        self.rate_limit = rate_limit
        self.analyzer = GeocodingAnalyzer()
        
        # Configure logging with both file and console output
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
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
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

    def geocode_with_postcode(
        self,
        postcode: str,
        house_number: str
    ) -> Optional[Dict]:
        """Geocode using postcode and house number"""
        query = f"{postcode} {house_number}, Netherlands"
        self.logger.debug(f"Attempting postcode geocoding: {query}")
        
        try:
            self.logger.debug(f"Trying street query: {query}")
            params = {
                'q': query,
                'format': 'json',
                'addressdetails': 1,
                'limit': 1,
                'countrycodes': 'nl'  # Restrict to Netherlands
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
                }
        except Exception as e:
            self.logger.debug(f"Postcode geocoding failed: {str(e)}")
        
        return None

    def clean_address(self, address: str) -> str:
        """Clean address string for better geocoding results"""
        # Remove decimal points from house numbers
        address = address.replace('.0', '')
        
        # Expand common abbreviations
        abbreviations = {
            'str.': 'straat',
            'ln.': 'laan',
            'ln': 'laan',
            'str': 'straat',
            'Kon.': 'Koningin',
            'Dr.': 'Doctor',
            'St.': 'Sint',
            'Prof.': 'Professor'
        }
        
        for abbr, full in abbreviations.items():
            address = address.replace(f' {abbr} ', f' {full} ')
            
        # Remove periods from initials
        address = address.replace('.', ' ').replace('  ', ' ')
        
        return address.strip()

    def geocode_with_street(
        self,
        street: str,
        house_number: str,
        city: str
    ) -> Optional[Dict]:
        """Geocode using street name and house number"""
        # Clean and format address
        house_number = str(house_number).replace('.0', '')
        street = self.clean_address(street)
        city = self.clean_address(city)
        
        # Try different query formats
        queries = [
            f"{street} {house_number}, {city}, Netherlands",
            f"{street} {house_number}, Netherlands",  # Try without city
            f"{street}, {city}, Netherlands",  # Try without house number
            f"{street} {house_number}, {city}"  # Try without country
        ]
        
        for query in queries:
            self.logger.debug(f"Attempting street geocoding: {query}")
        
        try:
            params = {
                'q': query,
                'format': 'json',
                'addressdetails': 1,
                'limit': 1
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
                }
        except Exception as e:
            self.logger.debug(f"Street geocoding failed: {str(e)}")
        
        return None

    def _geocode_institution(
        self,
        institution: Dict
    ) -> Tuple[str, Optional[Dict]]:
        """Geocode a single institution with multiple strategies"""
        inst_id = institution['institution_id']
        address = institution.get('main_address', {})
        
        if not all(key in address for key in ['postcode', 'number', 'street', 'city']):
            self.logger.warning(f"Incomplete address for institution {inst_id}")
            self.analyzer.record_failure(inst_id, address, "incomplete_address")
            return inst_id, None
            
        # Try postcode first
        sleep(self.rate_limit)
        result = self.geocode_with_postcode(
            address['postcode'],
            address['number']
        )
        
        if result:
            self.logger.debug(f"Successfully geocoded {inst_id} using postcode")
            self.analyzer.record_success('postcode', result['confidence'])
            return inst_id, result
            
        # Fallback to street name
        sleep(self.rate_limit)
        result = self.geocode_with_street(
            address['street'],
            address['number'],
            address['city']
        )
        
        if result:
            self.logger.debug(f"Successfully geocoded {inst_id} using street name")
            self.analyzer.record_success('street', result['confidence'])
            return inst_id, result
            
        self.logger.warning(f"Failed to geocode {inst_id} with all methods")
        self.analyzer.record_failure(inst_id, address, "all_methods_failed")
        return inst_id, None

    def convert_to_geojson(self, data: Dict) -> Dict:
        """Convert education data to GeoJSON format"""
        features = []
        
        for edu_type, edu_data in data.items():
            for inst in edu_data['institutions']:
                if 'geocode' in inst:
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [
                                inst['geocode']['lon'],
                                inst['geocode']['lat']
                            ]
                        },
                        "properties": {
                            **inst,
                            "education_type": edu_type
                        }
                    }
                    # Remove geometry info from properties
                    feature['properties'].pop('geocode', None)
                    features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }

    def process_education_data(
        self,
        input_file: Path,
        output_json_file: Path,
        output_geojson_file: Path
    ) -> None:
        """Process education data with detailed analysis"""
        # Load data
        self.logger.info(f"Loading data from {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Process each education type
        for edu_type, edu_data in data.items():
            self.logger.info(f"\nProcessing {edu_type}")
            institutions = edu_data['institutions']
            
            # Create progress bar
            pbar = tqdm(
                total=len(institutions),
                desc=f"Geocoding {edu_type}",
                ncols=100
            )
            
            # Process institutions with thread pool
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_inst = {
                    executor.submit(self._geocode_institution, inst): inst
                    for inst in institutions
                }
                
                for future in as_completed(future_to_inst):
                    inst_id, geocode_result = future.result()
                    
                    for inst in institutions:
                        if inst['institution_id'] == inst_id:
                            if geocode_result:
                                inst['geocode'] = geocode_result
                            break
                            
                    pbar.update(1)
            
            pbar.close()
        
        # Generate analysis report
        report = self.analyzer.generate_report()
        self.logger.info("\nGeocoding Analysis Report:")
        self.logger.info(json.dumps(report, indent=2))
        
        # Save original JSON with geocoding
        self.logger.info(f"\nSaving geocoded JSON to {output_json_file}")
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        # Convert and save GeoJSON
        self.logger.info(f"Converting to GeoJSON")
        geojson_data = self.convert_to_geojson(data)
        
        self.logger.info(f"Saving GeoJSON to {output_geojson_file}")
        with open(output_geojson_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, ensure_ascii=False, indent=2)

def main():
    # Configure paths
    project_root = Path(__file__).parent.parent
    processed_data_dir = project_root / 'data' / 'processed'
    
    input_file = processed_data_dir / 'education_data_all.json'
    output_json_file = processed_data_dir / 'education_data_geocoded.json'
    output_geojson_file = processed_data_dir / 'education_data.geojson'
    
    # Initialize geocoder
    geocoder = LocalNominatimGeocoder(
        nominatim_url="http://localhost:8080",
        max_retries=3,
        timeout=10,
        workers=4,
        rate_limit=1.0
    )
    
    # Process data
    geocoder.process_education_data(
        input_file,
        output_json_file,
        output_geojson_file
    )

if __name__ == "__main__":
    main()