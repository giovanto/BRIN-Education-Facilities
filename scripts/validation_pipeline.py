#!/usr/bin/env python3
"""
validation_pipeline.py
Unified pipeline for BRIN location data validation and analysis
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

@dataclass
class LocationMatch:
    """Data class for location matching results"""
    brin_id: str
    reference_id: str
    type: str
    distance: float
    geocode_confidence: float
    geocode_method: str
    location_type: str  # 'main' or 'branch'

class ValidationPipeline:
    """Unified pipeline for validating and analyzing BRIN location data"""
    
    def __init__(
        self,
        brin_path: Path,
        reference_path: Path,
        output_dir: Path,
        max_distance: float = 100.0  # meters
    ):
        self.brin_path = brin_path
        self.reference_path = reference_path
        self.output_dir = output_dir
        self.max_distance = max_distance
        self.logger = self._setup_logging()
        
        # Initialize results storage
        self.validation_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'brin_file': str(brin_path),
                'reference_file': str(reference_path),
                'max_distance': max_distance
            },
            'location_validation': {
                'matches': [],
                'unmatched_reference': [],
                'unmatched_brin': []
            },
            'type_distribution': {}
        }
        
        # Load and validate input data
        self.brin_data = self._load_brin_data()
        self.reference_data = self._load_reference_data()
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging with file and console output"""
        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'validation_{datetime.now():%Y%m%d_%H%M%S}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def _load_brin_data(self) -> Dict:
        """Load and validate BRIN data"""
        self.logger.info(f"Loading BRIN data from {self.brin_path}")
        try:
            with open(self.brin_path) as f:
                data = json.load(f)
            
            # Validate required fields
            self._validate_brin_structure(data)
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading BRIN data: {e}")
            raise

    def _load_reference_data(self) -> Dict:
        """Load and validate reference dataset"""
        self.logger.info(f"Loading reference data from {self.reference_path}")
        try:
            with open(self.reference_path) as f:
                data = json.load(f)
            
            # Validate GeoJSON structure
            if data.get('type') != 'FeatureCollection':
                raise ValueError("Reference data must be GeoJSON FeatureCollection")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading reference data: {e}")
            raise

    def _validate_brin_structure(self, data: Dict) -> None:
        """Validate BRIN data structure"""
        required_fields = {
            'institutions': ['institution_id', 'type', 'geocode'],
            'geocode': ['lat', 'lon', 'confidence', 'method']
        }
        
        for edu_type, edu_data in data.items():
            if not isinstance(edu_data, dict):
                raise ValueError(f"Invalid structure for education type {edu_type}")
            
            for inst in edu_data.get('institutions', []):
                for field_group, fields in required_fields.items():
                    if field_group == 'institutions':
                        missing = [f for f in fields if f not in inst]
                        if missing:
                            raise ValueError(f"Missing required fields {missing} in institution {inst.get('institution_id')}")
                    elif field_group == 'geocode' and 'geocode' in inst:
                        missing = [f for f in fields if f not in inst['geocode']]
                        if missing:
                            raise ValueError(f"Missing required geocode fields {missing} in institution {inst.get('institution_id')}")

    def calculate_location_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> float:
        """Calculate distance between two points in meters using Haversine formula"""
        from math import radians, sin, cos, sqrt, atan2
        
        R = 6371000  # Earth radius in meters
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        return R * c

    def match_locations(self) -> None:
        """Match BRIN locations with reference dataset"""
        self.logger.info("Starting location matching")
        
        # Create spatial index for reference locations
        reference_locations = []
        for feature in self.reference_data['features']:
            coords = feature['geometry']['coordinates']
            properties = feature['properties']
            reference_locations.append({
                'id': properties['id'],
                'type': properties['type'],
                'subtype': properties.get('subtype'),
                'lat': coords[1],
                'lon': coords[0],
                'matched': False
            })
        
        # Match each BRIN location
        for edu_type, edu_data in self.brin_data.items():
            type_stats = {
                'brin_total': 0,
                'reference_total': sum(1 for loc in reference_locations if loc['type'] == edu_type),
                'matched': 0
            }
            
            for inst in edu_data['institutions']:
                if 'geocode' not in inst:
                    continue
                    
                type_stats['brin_total'] += 1
                best_match = None
                min_distance = float('inf')
                
                # Check against all reference locations
                for ref_loc in reference_locations:
                    if ref_loc['matched']:
                        continue
                        
                    distance = self.calculate_location_distance(
                        inst['geocode']['lat'],
                        inst['geocode']['lon'],
                        ref_loc['lat'],
                        ref_loc['lon']
                    )
                    
                    if distance <= self.max_distance and distance < min_distance:
                        min_distance = distance
                        best_match = ref_loc
                
                if best_match:
                    match = LocationMatch(
                        brin_id=inst['institution_id'],
                        reference_id=best_match['id'],
                        type=edu_type,
                        distance=min_distance,
                        geocode_confidence=inst['geocode']['confidence'],
                        geocode_method=inst['geocode']['method'],
                        location_type='main'
                    )
                    
                    self.validation_results['location_validation']['matches'].append(
                        match.__dict__
                    )
                    best_match['matched'] = True
                    type_stats['matched'] += 1
                else:
                    self.validation_results['location_validation']['unmatched_brin'].append({
                        'id': inst['institution_id'],
                        'type': edu_type,
                        'original_type': inst.get('original_type'),
                        'geocode': inst['geocode']
                    })
            
            # Record unmatched reference locations
            self.validation_results['location_validation']['unmatched_reference'].extend([
                {
                    'id': loc['id'],
                    'type': loc['type'],
                    'subtype': loc['subtype'],
                    'coordinates': [loc['lon'], loc['lat']]
                }
                for loc in reference_locations
                if not loc['matched'] and loc['type'] == edu_type
            ])
            
            self.validation_results['type_distribution'][edu_type] = type_stats
        
        self.logger.info("Location matching completed")

    def analyze_results(self) -> None:
        """Run complete analysis on validation results"""
        self.logger.info("Starting validation analysis")
        
        # Calculate match rates and statistics
        self._analyze_match_rates()
        self._analyze_geocoding_patterns()
        self._analyze_unmatched_locations()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Export results
        self._export_results()
        
        self.logger.info("Validation analysis completed")

    def _analyze_geocoding_patterns(self) -> None:
        """Analyze patterns in geocoding methods"""
        geocoding_stats = defaultdict(lambda: {
            'total_uses': 0,
            'successful_matches': 0,
            'average_confidence': 0.0,
            'by_type': defaultdict(int)
        })
        
        # Analyze matched locations
        for match in self.validation_results['location_validation']['matches']:
            method = match['geocode_method']
            edu_type = match['type']
            
            stats = geocoding_stats[method]
            stats['successful_matches'] += 1
            stats['by_type'][edu_type] += 1
            stats['total_confidence'] = stats.get('total_confidence', 0) + match['geocode_confidence']
        
        # Calculate averages and clean up
        for method, stats in geocoding_stats.items():
            if stats['successful_matches'] > 0:
                stats['average_confidence'] = stats['total_confidence'] / stats['successful_matches']
            stats.pop('total_confidence', None)
        
        self.validation_results['analysis']['geocoding_patterns'] = dict(geocoding_stats)

    def _analyze_unmatched_locations(self) -> None:
        """Analyze patterns in unmatched locations"""
        unmatched_stats = {
            'reference': {
                'total': 0,
                'by_type': defaultdict(int),
                'by_subtype': defaultdict(int)
            },
            'brin': {
                'total': 0,
                'by_type': defaultdict(int),
                'by_method': defaultdict(int),
                'common_patterns': []
            }
        }
        
        # Analyze unmatched reference locations 
        for loc in self.validation_results['location_validation']['unmatched_reference']:
            unmatched_stats['reference']['total'] += 1
            unmatched_stats['reference']['by_type'][loc['type']] += 1
            if 'subtype' in loc:
                unmatched_stats['reference']['by_subtype'][loc['subtype']] += 1
        
        # Analyze unmatched BRIN locations
        for loc in self.validation_results['location_validation']['unmatched_brin']:
            unmatched_stats['brin']['total'] += 1
            unmatched_stats['brin']['by_type'][loc['type']] += 1
            if 'geocode' in loc:
                unmatched_stats['brin']['by_method'][loc['geocode']['method']] += 1
        
        # Find common patterns (e.g., specific areas or institution types)
        type_combinations = defaultdict(int)
        for ref in self.validation_results['location_validation']['unmatched_reference']:
            for brin in self.validation_results['location_validation']['unmatched_brin']:
                if ref['type'] == brin['type']:
                    pattern = f"{ref['type']}"
                    if 'subtype' in ref:
                        pattern += f"_{ref['subtype']}"
                    type_combinations[pattern] += 1
        
        # Add significant patterns
        unmatched_stats['brin']['common_patterns'] = [
            {'pattern': pattern, 'count': count}
            for pattern, count in type_combinations.items()
            if count >= 5  # Threshold for significance
        ]
        
        self.validation_results['analysis']['unmatched_patterns'] = unmatched_stats

    def _analyze_match_rates(self) -> None:
        """Analyze match rates by education type"""
        match_stats = defaultdict(lambda: {
            'matched': 0,
            'unmatched': 0,
            'total': 0,
            'match_rate': 0.0,
            'average_distance': 0.0,
            'distance_stats': []
        })
        
        for match in self.validation_results['location_validation']['matches']:
            edu_type = match['type']
            match_stats[edu_type]['matched'] += 1
            match_stats[edu_type]['distance_stats'].append(match['distance'])
        
        for edu_type, stats in match_stats.items():
            total = stats['matched'] + len([
                u for u in self.validation_results['location_validation']['unmatched_brin']
                if u['type'] == edu_type
            ])
            stats['total'] = total
            stats['match_rate'] = stats['matched'] / total if total > 0 else 0
            stats['average_distance'] = np.mean(stats['distance_stats']) if stats['distance_stats'] else 0
            stats['distance_quartiles'] = (
                np.percentile(stats['distance_stats'], [25, 50, 75]).tolist()
                if stats['distance_stats'] else []
            )
        
        self.validation_results['analysis'] = {'match_rates': dict(match_stats)}

    def _generate_visualizations(self) -> None:
        """Generate analysis visualizations"""
        plots_dir = self.output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Match rate by type
        plt.figure(figsize=(10, 6))
        match_rates = self.validation_results['analysis']['match_rates']
        types = list(match_rates.keys())
        rates = [stats['match_rate'] * 100 for stats in match_rates.values()]
        
        plt.bar(types, rates)
        plt.title('Match Rates by Education Type')
        plt.ylabel('Match Rate (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'match_rates.png')
        plt.close()
        
        # Distance distribution
        plt.figure(figsize=(10, 6))
        for edu_type, stats in match_rates.items():
            if stats['distance_stats']:
                plt.hist(
                    stats['distance_stats'],
                    bins=30,
                    alpha=0.5,
                    label=edu_type
                )
        plt.title('Match Distance Distribution')
        plt.xlabel('Distance (meters)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / 'distance_distribution.png')
        plt.close()

    def generate_recommendations(self) -> List[Dict]:
        """Generate recommendations for pipeline improvement"""
        recommendations = []
        
        # Analyze match rates for potential issues
        match_rates = self.validation_results['analysis']['match_rates']
        for edu_type, stats in match_rates.items():
            if stats['match_rate'] < 0.8:  # Less than 80% match rate
                recommendations.append({
                    'type': 'match_rate',
                    'education_type': edu_type,
                    'severity': 'high' if stats['match_rate'] < 0.6 else 'medium',
                    'description': f"Low match rate ({stats['match_rate']*100:.1f}%) for {edu_type}",
                    'suggestions': [
                        "Review location filtering logic for this education type",
                        "Check for systematic differences in address formatting",
                        "Verify branch location handling"
                    ]
                })
        
        # Analyze geocoding patterns
        geocoding_patterns = self.validation_results['analysis']['geocoding_patterns']
        for method, stats in geocoding_patterns.items():
            if stats['average_confidence'] < 0.7:  # Low confidence threshold
                recommendations.append({
                    'type': 'geocoding',
                    'method': method,
                    'severity': 'medium',
                    'description': f"Low confidence scores for method: {method}",
                    'suggestions': [
                        "Review geocoding parameters",
                        "Consider alternative geocoding strategies",
                        "Verify address cleaning process"
                    ]
                })
        
        # Analyze unmatched patterns
        unmatched = self.validation_results['analysis']['unmatched_patterns']
        if unmatched['reference']['total'] > 0:
            common_types = sorted(
                unmatched['reference']['by_type'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            if common_types:
                recommendations.append({
                    'type': 'unmatched',
                    'severity': 'high',
                    'description': "Systematic unmatched locations detected",
                    'details': {
                        'most_affected_types': [t[0] for t in common_types],
                        'counts': [t[1] for t in common_types]
                    },
                    'suggestions': [
                        "Review location extraction logic for affected types",
                        "Check for missing branch locations",
                        "Verify address standardization process"
                    ]
                })
        
        return sorted(recommendations, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['severity']])

    def _export_results(self) -> None:
        """Export validation results and analysis"""
        # Save validation results
        results_file = self.output_dir / 'validation_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.validation_results, f, indent=2)
        
        # Generate markdown report
        report_file = self.output_dir / 'validation_report.md'
        with open(report_file, 'w') as f:
            f.write("# BRIN Location Validation Report\n\n")
            
            f.write("## Summary\n")
            f.write(f"- Analysis timestamp: {self.validation_results['metadata']['timestamp']}\n")
            f.write(f"- Maximum match distance: {self.validation_results['metadata']['max_distance']}m\n\n")
            
            f.write("## Match Rates by Education Type\n")
            for edu_type, stats in self.validation_results['analysis']['match_rates'].items():
                f.write(f"\n### {edu_type}\n")
                f.write(f"- Matched locations: {stats['matched']}/{stats['total']}\n")
                f.write(f"- Match rate: {stats['match_rate']*100:.1f}%\n")
                f.write(f"- Average distance: {stats['average_distance']:.1f}m\n")
                if stats['distance_quartiles']:
                    f.write("- Distance quartiles (25%, 50%, 75%):")
                    f.write(f" {', '.join(f'{q:.1f}m' for q in stats['distance_quartiles'])}\n")

    def analyze_location_relationships(self) -> None:
        """Analyze relationships between main and branch locations"""
        relationship_stats = {
            'branches_per_main': defaultdict(int),
            'distance_patterns': defaultdict(list),
            'coverage_gaps': []
        }
        
        # Analyze branch distribution
        for edu_type, edu_data in self.brin_data.items():
            for inst in edu_data['institutions']:
                branch_count = len(inst.get('branches', []))
                relationship_stats['branches_per_main'][edu_type] += branch_count
                
                # Analyze branch distances if geocoded
                if 'geocode' in inst and 'branches' in inst:
                    main_coords = (inst['geocode']['lat'], inst['geocode']['lon'])
                    
                    for branch in inst['branches']:
                        if 'geocode' in branch:
                            branch_coords = (branch['geocode']['lat'], branch['geocode']['lon'])
                            distance = self.calculate_location_distance(
                                main_coords[0], main_coords[1],
                                branch_coords[0], branch_coords[1]
                            )
                            relationship_stats['distance_patterns'][edu_type].append(distance)
        
        # Analyze potential coverage gaps
        for edu_type in self.type_mapping.values():
            ref_locs = [
                f for f in self.reference_data['features']
                if f['properties']['type'] == edu_type
            ]
            
            if not ref_locs:
                continue
                
            # Create grid of reference locations
            from scipy.spatial import ConvexHull, Delaunay
            points = np.array([
                [f['geometry']['coordinates'][1], f['geometry']['coordinates'][0]]
                for f in ref_locs
            ])
            
            if len(points) > 3:  # Need at least 4 points for meaningful analysis
                try:
                    # Find areas with sparse coverage
                    hull = ConvexHull(points)
                    tri = Delaunay(points)
                    
                    # Find large triangles (potential gaps)
                    for simplex in tri.simplices:
                        triangle_pts = points[simplex]
                        # Calculate triangle area
                        area = 0.5 * abs(np.cross(
                            triangle_pts[1] - triangle_pts[0],
                            triangle_pts[2] - triangle_pts[0]
                        ))
                        
                        if area > np.mean([
                            0.5 * abs(np.cross(
                                points[s[1]] - points[s[0]],
                                points[s[2]] - points[s[0]]
                            ))
                            for s in tri.simplices
                        ]) * 2:  # Threshold for large triangles
                            relationship_stats['coverage_gaps'].append({
                                'type': edu_type,
                                'area': float(area),
                                'center': triangle_pts.mean(axis=0).tolist(),
                                'vertices': triangle_pts.tolist()
                            })
                except Exception as e:
                    self.logger.warning(f"Could not analyze coverage for {edu_type}: {e}")
        
        self.validation_results['analysis']['location_relationships'] = relationship_stats

def main():
    """Main execution"""
    try:
        # Configure paths
        project_root = Path(__file__).resolve().parent.parent
        data_dir = project_root / 'data'
        
        pipeline = ValidationPipeline(
            brin_path=data_dir / 'processed' / 'education_data_geocoded.json',
            reference_path=data_dir / 'reference' / 'onderwijs_basis.geojson',
            output_dir=data_dir / 'validation',
            max_distance=100.0  # meters
        )
        
        pipeline.match_locations()
        pipeline.analyze_results()
    except Exception as e:
        print(f"An error occurred: {e}")