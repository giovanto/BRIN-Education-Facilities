"""
Step 1: Data Preparation & Integration
Processes Dutch education facility data from three main sources:
- ORGANISATIES_20250203.csv: Organization details and addresses
- RELATIES_20250203.csv: Inter-organizational relationships
- OVERGANGEN_20250203.csv: Organizational transitions (mergers, splits, etc.)

The script loads, processes, and integrates the data while preserving temporal
relationships and status information. Output is saved as JSON for further processing.
"""

import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from collections import defaultdict

# Handle deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Project structure setup
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DIR = DATA_DIR / "reference"
ANALYSIS_DIR = DATA_DIR / "analysis"
LOG_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [PROCESSED_DIR, ANALYSIS_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging with both file and console handlers
    
    Args:
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger instance
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOG_DIR / f'prepare_raw_data_{timestamp}.log'
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def safe_isoformat(dt: Optional[pd.Timestamp]) -> Optional[str]:
    """
    Safely convert Pandas Timestamp to ISO format string
    
    Args:
        dt: Pandas Timestamp or None
    
    Returns:
        ISO format string or None
    """
    if pd.isna(dt):
        return None
    return dt.isoformat()

def load_raw_data(logger: logging.Logger) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and validate raw CSV files
    
    Args:
        logger: Logger instance
    
    Returns:
        Tuple of (organisations, relations, transitions) DataFrames
    """
    try:
        # Load CSVs with proper settings
        orgs_raw = pd.read_csv(RAW_DIR / "ORGANISATIES_20250203.csv", low_memory=False)
        rels_raw = pd.read_csv(RAW_DIR / "RELATIES_20250203.csv", low_memory=False)
        trans_raw = pd.read_csv(RAW_DIR / "OVERGANGEN_20250203.csv", low_memory=False)
        
        # Log data structure info
        for name, df in [("ORGANISATIES", orgs_raw), ("RELATIES", rels_raw), ("OVERGANGEN", trans_raw)]:
            logger.info(f"\nAnalyzing {name} structure:")
            logger.info(f"Columns: {df.columns.tolist()}")
            
            # Sample unique values for key fields
            for col in df.columns:
                n_unique = df[col].nunique()
                n_null = df[col].isna().sum()
                logger.info(f"{col}: {n_unique} unique values, {n_null} nulls")
                
                # Show sample values if not too many
                if n_unique < 10:
                    samples = df[col].dropna().unique().tolist()
                    logger.info(f"Sample values: {samples}")
        
        # Define date columns for each dataset
        date_cols = {
            'organisations': [
                'DT_BEGIN_RECORD', 'DT_EINDE_RECORD', 'DT_STICHTING',
                'DT_IN_BEDRIJF', 'DT_UIT_BEDRIJF', 'DT_AFGEHANDELD'
            ],
            'relations': ['DT_BEGIN_RELATIE', 'DT_EINDE_RELATIE'],
            'transitions': ['DT_OVERGANG']
        }
        
        # Convert dates for each dataframe
        organisations = orgs_raw.copy()
        relations = rels_raw.copy()
        transitions = trans_raw.copy()
        
        for df, cols in [
            (organisations, date_cols['organisations']),
            (relations, date_cols['relations']),
            (transitions, date_cols['transitions'])
        ]:
            for col in cols:
                df[col] = pd.to_datetime(df[col], format='%d-%m-%Y', errors='coerce')
        
        logger.info(f"\nLoaded final data:")
        logger.info(f"Organisations: {len(organisations)} rows")
        logger.info(f"Relations: {len(relations)} rows")
        logger.info(f"Transitions: {len(transitions)} rows")
        
        return organisations, relations, transitions
        
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        raise

def analyze_data_structure(
    organisations: pd.DataFrame,
    relations: pd.DataFrame,
    transitions: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze and summarize key characteristics of the datasets
    
    Args:
        organisations: Organisations DataFrame
        relations: Relations DataFrame
        transitions: Transitions DataFrame
        logger: Logger instance
    
    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        'organisations': {
            'total_count': len(organisations),
            'function_types': organisations['CODE_FUNCTIE'].value_counts().to_dict(),
            'organisation_types': organisations['CODE_SOORT'].value_counts().to_dict(),
            'active_status': organisations['IND_OPGEHEVEN'].value_counts().to_dict(),
            'date_range': {
                'earliest_record': safe_isoformat(organisations['DT_BEGIN_RECORD'].min()),
                'latest_record': safe_isoformat(organisations['DT_EINDE_RECORD'].max()),
                'earliest_operation': safe_isoformat(organisations['DT_IN_BEDRIJF'].min()),
                'latest_closure': safe_isoformat(organisations['DT_UIT_BEDRIJF'].max()),
            }
        },
        'relations': {
            'total_count': len(relations),
            'relation_types': relations['NAAM_RELATIE_LEID'].value_counts().to_dict(),
            'leading_functions': relations['CODE_FUNCTIE_LEID'].value_counts().to_dict(),
            'following_functions': relations['CODE_FUNCTIE_VOLG'].value_counts().to_dict(),
            'date_range': {
                'earliest_relation': safe_isoformat(relations['DT_BEGIN_RELATIE'].min()),
                'latest_relation': safe_isoformat(relations['DT_EINDE_RELATIE'].max())
            }
        }
    }
    
    # Log key findings
    logger.info("\nData Structure Analysis:")
    logger.info(f"\nOrganisations ({analysis['organisations']['total_count']} total):")
    logger.info(f"Function types: {analysis['organisations']['function_types']}")
    logger.info(f"Organisation types: {dict(list(analysis['organisations']['organisation_types'].items())[:5])}...")
    logger.info(f"Active status: {analysis['organisations']['active_status']}")
    
    logger.info(f"\nRelations ({analysis['relations']['total_count']} total):")
    logger.info(f"Relation types: {analysis['relations']['relation_types']}")
    logger.info(f"Leading functions: {analysis['relations']['leading_functions']}")
    logger.info(f"Following functions: {analysis['relations']['following_functions']}")
    
    return analysis

def analyze_transitions(
    transitions: pd.DataFrame,
    organisations: pd.DataFrame,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Analyze patterns in organizational transitions
    
    Args:
        transitions: Transitions DataFrame
        organisations: Organisations DataFrame
        logger: Logger instance
    
    Returns:
        Dictionary containing transition analysis
    """
    analysis = {
        'transition_counts': {},
        'transition_patterns': [],
        'timeline': {}
    }
    
    # Track organizations involved in multiple transitions
    org_transitions = defaultdict(list)
    for _, row in transitions.iterrows():
        org_transitions[row['NR_ADMIN_VAN']].append({
            'date': safe_isoformat(row['DT_OVERGANG']),
            'type': row['CODE_OVERGANG'],
            'to': row['NR_ADMIN_NAAR']
        })
        org_transitions[row['NR_ADMIN_NAAR']].append({
            'date': safe_isoformat(row['DT_OVERGANG']),
            'type': row['CODE_OVERGANG'],
            'from': row['NR_ADMIN_VAN']
        })
    
    # Find complex transition patterns
    complex_cases = {
        org_id: transitions 
        for org_id, transitions in org_transitions.items() 
        if len(transitions) > 1
    }
    
    # Analyze transition chains
    chains = []
    for org_id, trans_list in complex_cases.items():
        sorted_trans = sorted(trans_list, key=lambda x: x['date'])
        
        # Get organization names if available
        org_info = organisations[organisations['NR_ADMINISTRATIE'] == org_id]
        org_name = org_info['NAAM_KORT'].iloc[0] if not org_info.empty else org_id
        
        chains.append({
            'organization': org_name,
            'transitions': sorted_trans
        })
    
    analysis['complex_transitions'] = chains
    
    # Quick stats
    analysis['stats'] = {
        'total_transitions': len(transitions),
        'organizations_with_multiple_transitions': len(complex_cases),
        'transition_types': transitions['CODE_OVERGANG'].value_counts().to_dict(),
        'complete_transitions': transitions['IND_VOLLEDIG'].value_counts().to_dict()
    }
    
    logger.info("\nTransition Analysis:")
    logger.info(f"Total transitions: {analysis['stats']['total_transitions']}")
    logger.info(f"Organizations with multiple transitions: {len(complex_cases)}")
    logger.info(f"Transition types: {analysis['stats']['transition_types']}")
    
    if chains:
        logger.info("\nComplex transition examples:")
        for chain in chains[:3]:  # Show first 3 examples
            logger.info(f"\nOrg {chain['organization']}:")
            for t in chain['transitions']:
                logger.info(f"  {t['date']}: {t['type']}")
    
    return analysis

def process_organisation_history(organisations: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process organisations while preserving historical records and status
    
    Args:
        organisations: Organisations DataFrame
    
    Returns:
        List of processed organization dictionaries
    """
    processed_orgs = []
    
    for _, org in organisations.iterrows():
        org_dict = {
            'id': org['NR_ADMINISTRATIE'],
            'basic_info': {
                'short_name': org['NAAM_KORT'],
                'full_name': org['NAAM_VOLLEDIG'],
                'function': {
                    'code': org['CODE_FUNCTIE'],
                    'name': org['NAAM_FUNCTIE']
                },
                'type': {
                    'code': org['CODE_SOORT'],
                    'name': org['NAAM_SOORT']
                },
                'sector': {
                    'code': org['CODE_WET'],
                    'funding_type': org['CODE_TYPE_BEKOSTIGING']
                }
            },
            'status': {
                'is_active': org['IND_OPGEHEVEN'] == 'N',
                'record_status': org['CODE_STAND_RECORD'],
                'dates': {
                    'founded': safe_isoformat(org['DT_STICHTING']),
                    'started_operation': safe_isoformat(org['DT_IN_BEDRIJF']),
                    'ended_operation': safe_isoformat(org['DT_UIT_BEDRIJF']),
                    'record_start': safe_isoformat(org['DT_BEGIN_RECORD']),
                    'record_end': safe_isoformat(org['DT_EINDE_RECORD'])
                }
            },
            'location': {
                'visiting_address': {
                    'street': org['NAAM_STRAAT_VEST'],
                    'house_number': org['NR_HUIS_VEST'],
                    'house_number_addition': org['NR_HUIS_TOEV_VEST'],
                    'postcode': org['POSTCODE_VEST'],
                    'city': org['NAAM_PLAATS_VEST'],
                    'municipality_code': org['NR_GEMEENTE_VEST'],
                    'province': org['PROVINCIE_VEST']
                },
                'mailing_address': {
                    'street': org['NAAM_STRAAT_CORR'],
                    'house_number': org['NR_HUIS_CORR'],
                    'house_number_addition': org['NR_HUIS_TOEV_CORR'],
                    'postcode': org['POSTCODE_CORR'],
                    'city': org['NAAM_PLAATS_CORR'],
                    'municipality_code': org['NR_GEMEENTE_CORR'],
                    'province': org['PROVINCIE_CORR']
                }
            },
            'contact': {
                'phone': org['NR_TELEFOON'],
                'email': org['E_MAIL'],
                'website': org['INTERNET'],
                'kvk_number': org['KVK_NR']
            }
        }
        processed_orgs.append(org_dict)
    
    return processed_orgs

def process_relations(relations: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process relations while preserving connection information
    
    Args:
        relations: Relations DataFrame
    
    Returns:
        List of processed relation dictionaries
    """
    processed_rels = []
    
    for _, rel in relations.iterrows():
        rel_dict = {
            'leading_organization': {
                'id': rel['NR_ADMIN_LEID'],
                'function': rel['CODE_FUNCTIE_LEID'],
                'type': rel['CODE_SOORT_LEID'],
                'sector': rel['CODE_WET_LEID']
            },
            'following_organization': {
                'id': rel['NR_ADMIN_VOLG'],
                'function': rel['CODE_FUNCTIE_VOLG'],
                'type': rel['CODE_SOORT_VOLG'],
                'sector': rel['CODE_WET_VOLG']
            },
            'relationship': {
                'from_leader': rel['NAAM_RELATIE_LEID'],
                'from_follower': rel['NAAM_RELATIE_VOLG']
            },
            'validity': {
                'start_date': safe_isoformat(rel['DT_BEGIN_RELATIE']),
                'end_date': safe_isoformat(rel['DT_EINDE_RELATIE']),
                'status': rel['CODE_STAND_RECORD'],
                'is_terminated': rel['IND_OPGEHEVEN'] == 'J'
            }
        }
        processed_rels.append(rel_dict)
    
    return processed_rels

def process_transitions(transitions: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Process transitions while preserving history and connections
    
    Args:
        transitions: Transitions DataFrame
    
    Returns:
        List of processed transition dictionaries
    """
    processed_trans = []
    
    for _, trans in transitions.iterrows():
        trans_dict = {
            'source': {
                'id': trans['NR_ADMIN_VAN'],
                'function': trans['CODE_FUNCTIE_VAN'],
                'type': trans['CODE_SOORT_VAN'],
                'sector': trans['CODE_WET_VAN']
            },
            'destination': {
                'id': trans['NR_ADMIN_NAAR'],
                'function': trans['CODE_FUNCTIE_NAAR'],
                'type': trans['CODE_SOORT_NAAR'],
                'sector': trans['CODE_WET_NAAR']
            },
            'transition': {
                'date': safe_isoformat(trans['DT_OVERGANG']),
                'type': trans['CODE_OVERGANG'],
                'name': trans['NAAM_OVERGANG'],
                'is_complete': trans['IND_VOLLEDIG'] == 'J',
                'status': trans['CODE_STAND_RECORD']
            }
        }
        processed_trans.append(trans_dict)
    
    return processed_trans

def save_analysis_summary(analysis: Dict[str, Any], transition_analysis: Dict[str, Any]) -> None:
    """
    Save analysis results to a separate summary file
    
    Args:
        analysis: Data structure analysis results
        transition_analysis: Transition analysis results
    """
    summary = {
        'created_at': datetime.now().isoformat(),
        'data_structure': analysis,
        'transitions': transition_analysis
    }
    
    summary_file = ANALYSIS_DIR / 'data_analysis_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

def main():
    """
    Main execution function for data preparation and integration
    """
    logger = setup_logging()
    logger.info("Starting raw data preparation...")
    
    try:
        # Load raw data
        organisations, relations, transitions = load_raw_data(logger)
        
        # Analyze data structures
        analysis = analyze_data_structure(organisations, relations, transitions, logger)
        transition_analysis = analyze_transitions(transitions, organisations, logger)
        
        # Save analysis summary
        save_analysis_summary(analysis, transition_analysis)
        logger.info("Saved analysis summary to data_analysis_summary.json")
        
        # Process data
        processed_orgs = process_organisation_history(organisations)
        processed_rels = process_relations(relations)
        processed_trans = process_transitions(transitions)
        
        # Prepare final output
        output = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'record_counts': {
                    'organisations': len(processed_orgs),
                    'relations': len(processed_rels),
                    'transitions': len(processed_trans)
                },
                'analysis_summary': {
                    'organisations': {
                        'total': analysis['organisations']['total_count'],
                        'active': analysis['organisations']['active_status'].get('N', 0),
                        'inactive': analysis['organisations']['active_status'].get('J', 0)
                    },
                    'relations': {
                        'total': analysis['relations']['total_count'],
                        'types': len(analysis['relations']['relation_types'])
                    },
                    'transitions': {
                        'total': transition_analysis['stats']['total_transitions'],
                        'complex_cases': transition_analysis['stats']['organizations_with_multiple_transitions']
                    }
                }
            },
            'organisations': processed_orgs,
            'relations': processed_rels,
            'transitions': processed_trans
        }
        
        # Save processed data
        output_file = PROCESSED_DIR / 'education_organisations_processed.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Successfully saved processed data to {output_file}")
        
    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise

if __name__ == "__main__":
    main()