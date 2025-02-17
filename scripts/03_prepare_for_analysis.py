import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
import csv
import logging
import re

class NameValidator:
    """Validate school names against their declared types"""
    
    NAME_PATTERNS = {
        'ROC': [  # Place ROC first to catch before VOS
            r'\broc\b',
            r'regionaal\s+opleidingen?\s*centrum',
            r'\bmbo\s+college\b',
            r'graafschap\s+college',
            r'alfa-?college',
            r'koning\s+willem\s+i\s+college',
            r'da\s+vinci\s+college',
            r'friesland\s+college',
            r'(regionaal\s+opleidingen?\s*centrum.*college)',  # ROC specific colleges
            r'vakschool(?!\s+voor\s+praktijk)'
        ],
        'PROS': [  # Place PROS before VOS to catch praktijk patterns
            r'praktijkonderwijs',
            r'praktijkschool',
            r'\bpro\b(?!\s+education)',  # Not Pro Education
            r'praktijk\s*college',
            r'school\s+voor\s+praktijk',
            r'voor\s+praktijkonderwijs',
            r'pro\d+\s+college',  # e.g., PRO33 college
            r'.*\s+voor\s+.*\s+praktijkonderwijs'
        ],
        'SPEC': [
            r'speciaal\s+onderwijs',
            r'\bvso\b',
            r'\bzmlk\b',
            r'mytylschool',
            r'tyltylschool',
            r'cluster\s+[1234IViv]',
            r'zeer\s+moeilijk\s+(lerende|opvoedbare)',
            r'moeilijk\s+lerende\s+kinderen',
            r'voortgezet\s+speciaal\s+onderwijs',
            r'meforta\s+college',  # Specific SPEC institutions
            r'schreuder\s+college',
            r'w\.h\.\s+suringar.*college'
        ],
        'BAS': [
            r'\bbasisschool\b',
            r'\bobs\b',
            r'\bcbs\b',
            r'\bpcbs\b',
            r'\brkbs\b',
            r'\bbs\b(?!o)',  # Not BSO
            r'\bdaltonschool\b',
            r'\bmontessorischool\b',
            r'school\s+voor\s+basisonderwijs',
            r'basis\s+onderwijs(?!\s+speciaal)',  # Not special
            r'junior.*school'  # Include junior schools in basic
        ],
        'VOS': [
            r'(?<!praktijk)college(?!\s+(voor|van)\s+(mbo|praktijk))',
            r'\blyceum\b',
            r'\bgymnasium\b',
            r'\b(vmbo|havo|vwo)\b',
            r'middelbare\s+school',
            r'scholengemeenschap(?!\s+voor\s+praktijk)',
            r'\batheneum\b',
            r'voortgezet\s+onderwijs(?!\s+.*praktijk)',
            r'mavo(?!\s+praktijk)'
        ],
        'SBAS': [
            r'speciale\s+basisschool',
            r'school\s+voor\s+speciaal\s+basisonderwijs',
            r'\bsbo\b',
            r'speciaal\s+basis',
            r'basis\s+speciaal\s+onderwijs'
        ],
        'UNIV': [
            r'\buniversiteit\b',
            r'\buniversity\b(?!\s+of\s+applied\s+sciences)',
            r'\bfaculteit\b',
            r'\bfaculty\b',
            r'\btu\s',
            r'technische\s+universiteit',
            r'school\s+of\s+medicine(?!\s+and\s+applied\s+sciences)'
        ],
        'HBOS': [
            r'\bhogeschool\b',
            r'\bhbo\b',
            r'\bacademie\b(?!.*junior)',  # Not junior academie
            r'\bconservatorium\b',
            r'\bkunstacademie\b',
            r'university\s+of\s+applied\s+sciences',
            r'school\s+of\s+arts',
            r'business\s+school'
        ]
    }
    
    def __init__(self):
        """Initialize the validator with compiled regex patterns"""
        self.patterns = {
            type_: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for type_, patterns in self.NAME_PATTERNS.items()
        }
        self.mismatches = defaultdict(list)
        self.type_stats = defaultdict(int)
    
    def validate_name(self, name: str, declared_type: str) -> Dict:
        """Validate a school name against its declared type with priority matching"""
        name_lower = name.lower()
        detected_types = set()
        
        # Track type statistics
        self.type_stats[declared_type] += 1
        
        # Check name against patterns in specific order
        for type_ in ['ROC', 'PROS', 'SPEC', 'SBAS', 'BAS', 'VOS', 'UNIV', 'HBOS']:
            if type_ in self.patterns:
                if any(pattern.search(name_lower) for pattern in self.patterns[type_]):
                    detected_types.add(type_)
                    if type_ in ['ROC', 'PROS', 'SPEC', 'SBAS']:
                        break
        
        # Special case handling
        if 'college' in name_lower and not detected_types:
            if 'praktijk' in name_lower:
                detected_types.add('PROS')
            elif 'mbo' in name_lower or 'beroeps' in name_lower:
                detected_types.add('ROC')
            else:
                detected_types.add('VOS')
        
        # Record mismatch if found
        if detected_types and declared_type not in detected_types:
            self.mismatches[declared_type].append({
                'name': name,
                'detected_types': list(detected_types)
            })
        
        return {
            'matches_declared': declared_type in detected_types,
            'detected_types': list(detected_types),
            'needs_review': bool(detected_types and declared_type not in detected_types)
        }
    
    def get_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        return {
            'type_distribution': dict(self.type_stats),
            'mismatches_by_type': {
                type_: len(mismatches)
                for type_, mismatches in self.mismatches.items()
            },
            'mismatch_examples': {
                type_: mismatches[:5]
                for type_, mismatches in self.mismatches.items()
            }
        }

class EducationDataPreparator:
    """Prepare education data for analysis by creating analysis files"""
    
    def __init__(self, input_geojson: str):
        """Initialize with input GeoJSON path"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load data
        self.logger.info(f"Loading data from {input_geojson}")
        with open(input_geojson, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.features = self.data['features']
        self.logger.info(f"Loaded {len(self.features)} features")
        
        # Initialize components
        self.location_stats = defaultdict(int)
        self.transition_types = defaultdict(int)
        self.name_validator = NameValidator()
    
    def _get_decade_activity(self, start_date: str, end_date: str) -> Dict[str, bool]:
        """Determine which decades a school was active in"""
        try:
            start_year = int(start_date.split('-')[2]) if start_date else 1900
            end_year = int(end_date.split('-')[2]) if end_date else 2025
            
            return {
                f"active_{decade}_{decade+9}": (
                    start_year <= decade+9 and 
                    (end_year >= decade or not end_date)
                )
                for decade in range(1950, 2030, 10)
            }
        except (ValueError, IndexError):
            return {f"active_{d}_{d+9}": False for d in range(1950, 2030, 10)}
    
    def process_data(self):
        """Process the education data and track statistics"""
        self.logger.info("Processing education data...")
        
        # Step 1: Collect all branch IDs and their parents
        branch_ids = set()
        parent_to_branches = defaultdict(list)
        
        for feature in self.features:
            props = feature['properties']
            if 'branches' in props:
                for branch in props['branches']:
                    branch_ids.add(branch['branch_id'])
                    parent_to_branches[props['institution_id']].append(branch)
        
        # Step 2: Process main locations and validate names
        main_locations = []
        total_true_main = 0
        
        for feature in self.features:
            props = feature['properties']
            inst_id = props['institution_id']
            
            # Skip if this location is itself a branch
            if inst_id in branch_ids:
                continue
            
            # Track statistics
            if props.get('end_date'):
                self.location_stats['historical'] += 1
            if 'transitions' in props:
                self.location_stats['with_transitions'] += 1
                for trans in props['transitions']:
                    self.transition_types[trans['type']] += 1
            
            # Process main locations
            if inst_id in parent_to_branches:
                total_true_main += 1
                
                # Validate school name
                validation_result = self.name_validator.validate_name(
                    props['name'],
                    props['education_type']
                )
                
                decade_activity = self._get_decade_activity(
                    props.get('start_date'),
                    props.get('end_date')
                )
                
                processed_location = {
                    "type": "Feature",
                    "geometry": feature['geometry'],
                    "properties": {
                        "id": inst_id,
                        "name": props['name'],
                        "type": props['education_type'],
                        "validation_status": not validation_result['needs_review'],
                        "start_date": props.get('start_date'),
                        "end_date": props.get('end_date'),
                        "municipality": props['main_address']['city'],
                        "province": props['main_address']['province'],
                        "num_branches": len(parent_to_branches[inst_id]),
                        "temporal_activity": decade_activity
                    }
                }
                main_locations.append(processed_location)
        
        # Log analysis results
        self.logger.info(f"\nDetailed Location Analysis:")
        self.logger.info(f"Total locations in dataset: {len(self.features)}")
        self.logger.info(f"Branch locations: {len(branch_ids)}")
        self.logger.info(f"True main locations: {total_true_main}")
        
        # Log name validation results
        validation_report = self.name_validator.get_validation_report()
        self.logger.info("\nName Validation Results:")
        for type_, count in validation_report['mismatches_by_type'].items():
            if count > 0:
                self.logger.info(f"{type_}: {count} potential mismatches")
                examples = validation_report['mismatch_examples'][type_]
                self.logger.info("Example mismatches:")
                for ex in examples:
                    self.logger.info(f"  - {ex['name']} (detected as: {', '.join(ex['detected_types'])})")
        
        self.location_stats['main'] = total_true_main
        self.location_stats['branches'] = len(branch_ids)
        
        return main_locations
    
    def export_data(self, output_dir: Path):
        """Export processed data to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process data
        main_locations = self.process_data()
        
        # Save GeoJSON
        self.logger.info("Saving GeoJSON...")
        geojson_output = {
            "type": "FeatureCollection",
            "metadata": {
                "description": "Main education locations with type validation",
                "processing_date": datetime.now().isoformat(),
                "statistics": {
                    "total_main_locations": self.location_stats['main'],
                    "historical_locations": self.location_stats['historical'],
                    "locations_with_transitions": self.location_stats['with_transitions']
                },
                "name_validation": self.name_validator.get_validation_report()
            },
            "features": main_locations
        }
        
        with open(output_dir / 'education_locations_analysis.geojson', 'w', encoding='utf-8') as f:
            json.dump(geojson_output, f, ensure_ascii=False, indent=2)
        
        # Save CSV
        self.logger.info("Saving CSV...")
        if main_locations:
            csv_rows = []
            for loc in main_locations:
                props = loc['properties']
                coords = loc['geometry']['coordinates']
                
                csv_row = {
                    "id": props['id'],
                    "name": props['name'],
                    "geom": f"POINT({coords[0]} {coords[1]})",
                    "type": props['type'],
                    "type_validated": props['validation_status'],
                    "start_date": props['start_date'],
                    "end_date": props['end_date'],
                    "municipality": props['municipality'],
                    "province": props['province'],
                    "num_branches": props['num_branches']
                }
                
                # Add decade activity fields
                csv_row.update(props['temporal_activity'])
                csv_rows.append(csv_row)
            
            with open(output_dir / 'education_locations.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)
        
       # Save validation report
        with open(output_dir / 'name_validation_report.json', 'w', encoding='utf-8') as f:
            json.dump(self.name_validator.get_validation_report(), f, ensure_ascii=False, indent=2)
        
        # Print summary
        self.logger.info("\n=== Processing Summary ===")
        self.logger.info(f"Main locations processed: {self.location_stats['main']}")
        self.logger.info(f"Historical locations: {self.location_stats['historical']}")
        self.logger.info(f"Locations with transitions: {self.location_stats['with_transitions']}")
        
        self.logger.info("\nTransition Types:")
        for t_type, count in self.transition_types.items():
            self.logger.info(f"  {t_type}: {count}")

def main():
    # Configure paths
    project_root = Path(__file__).parent.parent
    input_geojson = project_root / 'data' / 'processed' / 'education_data.geojson'
    output_dir = project_root / 'data' / 'analysis'
    
    # Process data
    preparator = EducationDataPreparator(str(input_geojson))
    preparator.export_data(output_dir)

if __name__ == "__main__":
    main()