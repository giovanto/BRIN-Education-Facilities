# Dutch Education Data Processing Pipeline

## Overview
This pipeline processes Dutch education institution data from raw BRIN data to analysis-ready formats with geocoding and type validation.

## Project Structure
```
project_root/
├── data/
│   ├── raw/                     # Raw BRIN CSV files
│   ├── processed/               # Intermediate processing files
│   └── analysis/               # Final analysis files
│   └── reference/               # onderwijs_basis.geojson
├── scripts/
│   ├── 01_create_JSON.py       # Initial data structuring
│   ├── 02_geocode_addresses.py # Geocoding
│   └── 03_prepare_for_analysis.py # Analysis preparation
└── logs/                       # Processing logs
```

## Data Processing Stages

### 1. Initial Processing (01_create_JSON.py)
Converts raw BRIN data to structured JSON.

**Inputs:**
- `ORGANISATIES_20250203.csv`
- `RELATIES_20250203.csv`
- `OVERGANGEN_20250203.csv`

**Output:**
- `education_data_all.json`

### 2. Geocoding (02_geocode_addresses.py)
Adds geographic coordinates using local Nominatim service.

**Input:**
- `education_data_all.json`

**Outputs:**
- `education_data_geocoded.json`
- `education_data.geojson`

### 3. Analysis Preparation (03_prepare_for_analysis.py)
Prepares final analysis files with validation.

**Input:**
- `education_data.geojson`

**Outputs:**
1. `education_locations_analysis.geojson`: Main locations with validation
2. `education_locations.csv`: Database-ready format
3. `name_validation_report.json`: Validation results

## Data Validation

### Institution Types
The pipeline validates Dutch education types using hierarchical patterns:

#### Basic Education (BAS)
- Standard patterns: `basisschool`, `obs`, `cbs`, `pcbs`, `rkbs`
- Special cases: "Pro Rege" type names (not PROS)
- Junior schools and Montessori/Dalton

#### Secondary Education (VOS)
- Main patterns: `lyceum`, `gymnasium`, `atheneum`
- VMBO/HAVO/VWO combinations
- Schools with PRO/LWOO sections remain VOS
- Regular colleges (non-ROC/PROS)

#### Regional Education (ROC)
- Specific institutions: Deltion, Vista, Graafschap College
- Regional education centers
- MBO-focused institutions

#### Special Education Types
- SPEC: Special education institutions
- SBAS: Special basic education
- PROS: Practical education
- UNIV/HBOS: Higher education

### Current Validation Results
From latest run (2025-02-15):
- Total locations: 8,711
- Main locations: 7,688
- Branch locations: 10,333
- Validation matches needed review:
  - BAS: 5 cases
  - VOS: 39 cases
  - SBAS: 4 cases
  - SPEC: 28 cases
  - ROC: 5 cases
  - UNIV: 1 case
  - PROS: 4 cases

## Output Formats

### 1. Analysis GeoJSON
```json
{
  "type": "FeatureCollection",
  "metadata": {
    "statistics": {
      "total_main_locations": 7688,
      "historical_locations": 1204,
      "locations_with_transitions": 2466
    }
  },
  "features": [...]
}
```

### 2. CSV Structure
```csv
id,name,geom,type,type_validated,start_date,end_date,municipality,province,num_branches,active_*
```

### 3. Validation Report
```json
{
  "type_distribution": {},
  "mismatches_by_type": {},
  "mismatch_examples": []
}
```

## Usage

### Prerequisites
- Python 3.8+
- Running Nominatim instance
- Required Python packages

### Running the Pipeline
```bash
# 1. Initial processing
python 01_create_JSON.py

# 2. Geocoding
python 02_geocode_addresses.py

# 3. Analysis preparation
python 03_prepare_for_analysis.py
```

### Output Files Location
```
data/analysis/
├── education_locations_analysis.geojson
├── education_locations.csv
└── name_validation_report.json
```

## Data Quality Notes

### Type Validation Challenges
1. Multi-type institutions
   - Schools offering multiple education levels
   - Combined PRO/VMBO programs
   - Special education with regular programs

2. Name Pattern Conflicts
   - "Pro" in school names
   - College type disambiguation
   - Special education institutions

### Location Data
- All coordinates in WGS84 (EPSG:4326)
- Main/branch relationship validation
- Temporal consistency checks

## Transition Analysis
Current statistics:
- Mergers (Fusie): 2,797
- Transfers (Overdracht): 159
- Splits (Splitsing): 31
- Branch Independence: 2

## Maintenance

### Regular Updates
1. Update validation patterns
2. Monitor validation results
3. Review complex cases
4. Update institution lists

### Quality Checks
1. Validate type assignments
2. Check temporal consistency
3. Verify location accuracy
4. Review transition logic