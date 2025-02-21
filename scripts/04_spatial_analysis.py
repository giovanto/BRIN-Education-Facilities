import json
from pathlib import Path
import logging
from datetime import datetime
import rtree
import numpy as np
from collections import defaultdict
import pandas as pd

# Setup project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
ANALYSIS_DIR = DATA_DIR / "analysis"
LOG_DIR = PROJECT_ROOT / "logs"

# Setup logging
log_file = LOG_DIR / f"reverse_spatial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def create_spatial_index(points, is_validated=False):
    """
    Create an R-tree index for fast spatial queries
    """
    idx = rtree.index.Index()
    valid_points = {}
    
    for i, point in enumerate(points):
        if point['geometry'] is not None:
            coords = point['geometry']['coordinates'][:2]  # Ignore Z if present
            idx.insert(i, (coords[0], coords[1], coords[0], coords[1]))
            if is_validated:
                valid_points[i] = {
                    'point': point,
                    'type': point['properties'].get('subtype', 'Unknown')
                }
            else:
                valid_points[i] = point
                
    return idx, valid_points

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points in meters
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    r = 6371000  # Radius of earth in meters
    return c * r

def find_validated_matches(raw_point, validated_index, validated_points, max_distance=500):
    """
    Find all validated points within range of a raw point
    """
    if raw_point['geometry'] is None:
        return []
        
    r_lon, r_lat = raw_point['geometry']['coordinates'][:2]
    
    # Convert distance to degrees with buffer
    degree_distance = (max_distance / 111320) * 1.2
    
    search_bounds = (
        r_lon - degree_distance,
        r_lat - degree_distance,
        r_lon + degree_distance,
        r_lat + degree_distance
    )
    
    matches = []
    candidate_indices = list(validated_index.intersection(search_bounds))
    
    for idx in candidate_indices:
        validated_data = validated_points[idx]
        v_point = validated_data['point']
        v_lon, v_lat = v_point['geometry']['coordinates'][:2]
        
        distance = haversine_distance(r_lon, r_lat, v_lon, v_lat)
        if distance <= max_distance:
            matches.append({
                'validated_point': v_point,
                'distance': distance,
                'school_type': validated_data['type']
            })
    
    return matches

def analyze_point_properties(raw_point, matches):
    """
    Analyze properties of a raw point and its matches
    """
    props = raw_point['properties']
    analysis = {
        'properties': {
            'CODE_FUNCTIE': props.get('CODE_FUNCTIE', 'Unknown'),
            'CODE_SOORT': props.get('CODE_SOORT', 'Unknown'),
            'NAAM_FUNCTIE': props.get('NAAM_FUNCTIE', 'Unknown'),
            'NAAM_SOORT': props.get('NAAM_SOORT', 'Unknown'),
            'IND_OPGEHEVEN': props.get('IND_OPGEHEVEN', 'Unknown'),
            'CODE_STAND_RECORD': props.get('CODE_STAND_RECORD', 'Unknown')
        },
        'matched': len(matches) > 0,
        'num_matches': len(matches),
        'matches': [{
            'distance': m['distance'],
            'school_type': m['school_type']
        } for m in matches],
        'matched_types': list(set(m['school_type'] for m in matches))
    }
    
    if matches:
        analysis['closest_match'] = min(matches, key=lambda x: x['distance'])['distance']
        analysis['min_distance'] = min(m['distance'] for m in matches)
        analysis['max_distance'] = max(m['distance'] for m in matches)
        analysis['avg_distance'] = np.mean([m['distance'] for m in matches])
    
    return analysis

def analyze_match_distances(matching_patterns):
    """
    Analyze distance patterns in matches
    """
    distances = defaultdict(list)
    for pattern in matching_patterns:
        func_code = pattern['properties']['CODE_FUNCTIE']
        sort_code = pattern['properties']['CODE_SOORT']
        key = f"{func_code}_{sort_code}"
        
        if pattern['matches']:  # Check if there are any matches
            distances[key].extend([m['distance'] for m in pattern['matches']])
    
    distance_stats = {}
    for key, vals in distances.items():
        if vals:  # Check if we have any distances for this combination
            distance_stats[key] = {
                'mean': float(np.mean(vals)),  # Convert numpy types to native Python types
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'count': len(vals),
                'min': float(np.min(vals)),
                'max': float(np.max(vals))
            }
    
    return distance_stats

def analyze_school_type_matches(matching_patterns):
    """
    Analyze patterns in school type matches
    """
    type_matches = defaultdict(lambda: defaultdict(int))
    for pattern in matching_patterns:
        func_code = pattern['properties']['CODE_FUNCTIE']
        sort_code = pattern['properties']['CODE_SOORT']
        key = f"{func_code}_{sort_code}"
        
        for match in pattern['matches']:
            type_matches[key][match['school_type']] += 1
    
    # Convert defaultdict to regular dict for JSON serialization
    return {k: dict(v) for k, v in type_matches.items()}

def main():
    logging.info("Starting reverse spatial analysis...")
    
    # Load datasets
    try:
        with open(REFERENCE_DIR / "onderwijs_basis.geojson") as f:
            validated_data = json.load(f)
        with open(PROCESSED_DIR / "education_data_all_geocoded.geojson") as f:
            raw_data = json.load(f)
    except FileNotFoundError as e:
        logging.error(f"Could not find input files: {e}")
        return
    
    # Create spatial indices
    logging.info("Creating spatial indices...")
    validated_index, validated_points = create_spatial_index(validated_data['features'], True)
    
    # Initialize results structure
    results = {
        'by_function': defaultdict(int),
        'by_sort': defaultdict(int),
        'matched_by_function': defaultdict(lambda: {'matched': 0, 'unmatched': 0}),
        'matched_by_sort': defaultdict(lambda: {'matched': 0, 'unmatched': 0}),
        'type_combinations': defaultdict(int)
    }
    
    matching_patterns = []
    
    # Process raw points
    logging.info("Processing raw points...")
    total_points = len(raw_data['features'])
    
    for idx, raw_point in enumerate(raw_data['features']):
        if idx % 1000 == 0:
            logging.info(f"Processed {idx}/{total_points} raw points...")
            
        matches = find_validated_matches(raw_point, validated_index, validated_points)
        analysis = analyze_point_properties(raw_point, matches)
        
        # Update statistics
        func_code = analysis['properties']['CODE_FUNCTIE']
        sort_code = analysis['properties']['CODE_SOORT']
        is_matched = analysis['matched']
        
        results['by_function'][func_code] += 1
        results['by_sort'][sort_code] += 1
        
        if is_matched:
            results['matched_by_function'][func_code]['matched'] += 1
            results['matched_by_sort'][sort_code]['matched'] += 1
            matching_patterns.append(analysis)
        else:
            results['matched_by_function'][func_code]['unmatched'] += 1
            results['matched_by_sort'][sort_code]['unmatched'] += 1
        
        # Track combinations
        results['type_combinations'][f"{func_code}_{sort_code}"] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    serializable_results = {
        'by_function': dict(results['by_function']),
        'by_sort': dict(results['by_sort']),
        'matched_by_function': {k: dict(v) for k, v in results['matched_by_function'].items()},
        'matched_by_sort': {k: dict(v) for k, v in results['matched_by_sort'].items()},
        'type_combinations': dict(results['type_combinations'])
    }
    
    # Additional analysis
    distance_stats = analyze_match_distances(matching_patterns)
    type_match_stats = analyze_school_type_matches(matching_patterns)
    
    # Calculate match rates
    match_rates = {
        'function_rates': {},
        'sort_rates': {}
    }
    
    for func in serializable_results['by_function']:
        total = serializable_results['by_function'][func]
        matches = serializable_results['matched_by_function'][func]['matched']
        if total > 0:
            match_rates['function_rates'][func] = (matches / total) * 100
            
    for sort in serializable_results['by_sort']:
        total = serializable_results['by_sort'][sort]
        matches = serializable_results['matched_by_sort'][sort]['matched']
        if total > 0:
            match_rates['sort_rates'][sort] = (matches / total) * 100
    
    # Save results
    output_file = ANALYSIS_DIR / "reverse_spatial_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            'statistics': serializable_results,
            'match_rates': match_rates,
            'distance_stats': distance_stats,
            'type_match_stats': type_match_stats,
            'matching_patterns': matching_patterns[:1000]  # Limit patterns to avoid huge files
        }, f, indent=2)
    
    # Log summary statistics
    logging.info("\nAnalysis Summary:")
    
    logging.info("\nMatch rates by function code:")
    for func, rate in sorted(match_rates['function_rates'].items(), key=lambda x: x[1], reverse=True):
        total = serializable_results['by_function'][func]
        matches = serializable_results['matched_by_function'][func]['matched']
        logging.info(f"{func}: {rate:.1f}% ({matches}/{total})")
    
    logging.info("\nMatch rates by sort code:")
    for sort, rate in sorted(match_rates['sort_rates'].items(), key=lambda x: x[1], reverse=True):
        total = serializable_results['by_sort'][sort]
        matches = serializable_results['matched_by_sort'][sort]['matched']
        logging.info(f"{sort}: {rate:.1f}% ({matches}/{total})")
    
    # Log match rate summary
    logging.info("\n=== Match Rate Summary ===")
    total_matched = sum(results['matched_by_function'][f]['matched'] for f in results['by_function'])
    total_points = sum(results['by_function'].values())
    overall_match_rate = (total_matched / total_points) * 100 if total_points > 0 else 0
    logging.info(f"Overall match rate: {overall_match_rate:.1f}% ({total_matched}/{total_points})")

    # Log function code statistics
    logging.info("\n=== Function Code Statistics ===")
    for func_code in sorted(results['by_function'].keys()):
        total = results['by_function'][func_code]
        matches = results['matched_by_function'][func_code]['matched']
        rate = (matches / total) * 100 if total > 0 else 0
        logging.info(f"Function {func_code}: {rate:.1f}% matched ({matches}/{total})")

    # Log top combinations
    logging.info("\n=== Top Type Combinations ===")
    logging.info("Format: FUNCTION_SORT: count (match_rate%)")
    for combo, count in sorted(serializable_results['type_combinations'].items(), 
                             key=lambda x: x[1], reverse=True)[:10]:
        func, sort = combo.split('_')
        matched = sum(1 for p in matching_patterns if 
                     p['properties']['CODE_FUNCTIE'] == func and 
                     p['properties']['CODE_SOORT'] == sort)
        match_rate = (matched / count) * 100 if count > 0 else 0
        logging.info(f"{combo}: {count} ({match_rate:.1f}% matched)")

    # Log distance statistics for top combinations
    if distance_stats:
        logging.info("\n=== Distance Statistics for Top Combinations ===")
        for combo in sorted(distance_stats.keys(), 
                          key=lambda x: distance_stats[x]['count'], 
                          reverse=True)[:5]:
            stats = distance_stats[combo]
            logging.info(f"{combo}:")
            logging.info(f"  Count: {stats['count']}")
            logging.info(f"  Mean: {stats['mean']:.1f}m")
            logging.info(f"  Median: {stats['median']:.1f}m")
            logging.info(f"  Min: {stats['min']:.1f}m")
            logging.info(f"  Max: {stats['max']:.1f}m")
    
    logging.info(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()