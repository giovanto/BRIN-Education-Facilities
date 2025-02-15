import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Optional
import os
from pathlib import Path
from tqdm import tqdm

class EducationDataProcessor:
    """Process Dutch education institution data from BRIN database"""
    
    # Education type mappings
    EDUCATION_TYPES = {
        'BAS': 'Basic Education',
        'VOS': 'Secondary Education',
        'SBAS': 'Special Basic Education',
        'SPEC': 'Special Education',
        'HBOS': 'Higher Professional Education',
        'ROC': 'Regional Education Center',
        'UNIV': 'University',
        'MBO': 'Vocational Education',
        'PROS': 'Practical Education'
    }
    
    def __init__(self, data_dir: str):
        """Initialize with path to data directory"""
        print("Loading data files...")
        self.org_df = pd.read_csv(os.path.join(data_dir, 'ORGANISATIES_20250203.csv'), 
                                 sep=',', encoding='utf-8', quotechar='"', low_memory=False)
        self.rel_df = pd.read_csv(os.path.join(data_dir, 'RELATIES_20250203.csv'), 
                                 sep=',', encoding='utf-8', quotechar='"')
        self.ovg_df = pd.read_csv(os.path.join(data_dir, 'OVERGANGEN_20250203.csv'), 
                                 sep=',', encoding='utf-8', quotechar='"')
        
    def get_institution_stats(self, edu_type: str) -> Dict:
        """Get statistics for a specific education type"""
        institutions = self.org_df[
            (self.org_df['CODE_SOORT'] == edu_type) &
            (self.org_df['CODE_FUNCTIE'] == 'U')
        ]
        
        stats = {
            'total_count': len(institutions),
            'status_distribution': institutions['CODE_STAND_RECORD'].value_counts().to_dict(),
            'operational_status': institutions['IND_OPGEHEVEN'].value_counts().to_dict()
        }
        
        return stats
    
    def get_address_info(self, row: pd.Series) -> Dict:
        """Extract address information from a row"""
        return {
            'street': row['NAAM_STRAAT_VEST'],
            'number': row['NR_HUIS_VEST'],
            'number_addition': row['NR_HUIS_TOEV_VEST'] if pd.notna(row['NR_HUIS_TOEV_VEST']) else None,
            'postcode': row['POSTCODE_VEST'],
            'city': row['NAAM_PLAATS_VEST'],
            'province': row['PROVINCIE_VEST']
        }
    
    def get_branches(self, institution_id: str) -> List[Dict]:
        """Get branch locations for an institution"""
        branches = self.rel_df[
            (self.rel_df['NR_ADMIN_LEID'] == institution_id) &
            (self.rel_df['NAAM_RELATIE_LEID'] == 'vestigt') &
            (self.rel_df['CODE_STAND_RECORD'] == 'A')
        ]
        
        branch_list = []
        for _, branch in branches.iterrows():
            branch_org = self.org_df[self.org_df['NR_ADMINISTRATIE'] == branch['NR_ADMIN_VOLG']]
            if not branch_org.empty:
                branch_org = branch_org.iloc[0]
                branch_list.append({
                    'branch_id': branch['NR_ADMIN_VOLG'],
                    'address': self.get_address_info(branch_org),
                    'start_date': branch['DT_BEGIN_RELATIE'],
                    'end_date': branch['DT_EINDE_RELATIE'] if pd.notna(branch['DT_EINDE_RELATIE']) else None
                })
        return branch_list
    
    def get_transitions(self, institution_id: str) -> List[Dict]:
        """Get transition history for an institution"""
        transitions = self.ovg_df[
            (self.ovg_df['NR_ADMIN_NAAR'] == institution_id) |
            (self.ovg_df['NR_ADMIN_VAN'] == institution_id)
        ]
        
        transition_list = []
        for _, trans in transitions.iterrows():
            other_id = trans['NR_ADMIN_VAN'] if trans['NR_ADMIN_NAAR'] == institution_id else trans['NR_ADMIN_NAAR']
            other_org = self.org_df[self.org_df['NR_ADMINISTRATIE'] == other_id]
            if not other_org.empty:
                other_org = other_org.iloc[0]
                transition_list.append({
                    'type': trans['NAAM_OVERGANG'],
                    'date': trans['DT_OVERGANG'],
                    'direction': 'from' if trans['NR_ADMIN_NAAR'] == institution_id else 'to',
                    'other_institution': {
                        'id': other_id,
                        'name': other_org['NAAM_VOLLEDIG'],
                        'type': other_org['CODE_SOORT']
                    }
                })
        return sorted(transition_list, key=lambda x: x['date']) if transition_list else []
    
    def get_board_info(self, institution_id: str) -> Optional[Dict]:
        """Get board information for an institution"""
        board = self.rel_df[
            (self.rel_df['NR_ADMIN_VOLG'] == institution_id) &
            (self.rel_df['NAAM_RELATIE_VOLG'] == 'wordt bestuurd door') &
            (self.rel_df['CODE_STAND_RECORD'] == 'A')
        ]
        
        if not board.empty:
            board_org = self.org_df[self.org_df['NR_ADMINISTRATIE'] == board.iloc[0]['NR_ADMIN_LEID']]
            if not board_org.empty:
                board_org = board_org.iloc[0]
                return {
                    'id': board_org['NR_ADMINISTRATIE'],
                    'name': board_org['NAAM_VOLLEDIG'],
                    'start_date': board.iloc[0]['DT_BEGIN_RELATIE'],
                    'end_date': board.iloc[0]['DT_EINDE_RELATIE'] if pd.notna(board.iloc[0]['DT_EINDE_RELATIE']) else None
                }
        return None
    
    def process_education_type(self, edu_type: str) -> Dict:
        """Process all institutions of a specific education type"""
        print(f"\nProcessing {self.EDUCATION_TYPES.get(edu_type, edu_type)}...")
        
        # Get basic statistics
        stats = self.get_institution_stats(edu_type)
        print(f"Found {stats['total_count']} total institutions")
        
        # Process current institutions
        current_institutions = self.org_df[
            (self.org_df['CODE_SOORT'] == edu_type) &
            (self.org_df['CODE_FUNCTIE'] == 'U') &
            (self.org_df['CODE_STAND_RECORD'] == 'A')
        ]
        
        results = []
        total_institutions = len(current_institutions)
        
        # Reset index to get correct counting
        current_institutions = current_institutions.reset_index(drop=True)
        
        # Use tqdm for progress tracking
        for idx, inst in tqdm(current_institutions.iterrows(), 
                            total=total_institutions,
                            desc=f"Processing {edu_type} institutions",
                            ncols=100):
            
            institution_info = {
                'institution_id': inst['NR_ADMINISTRATIE'],
                'name': inst['NAAM_VOLLEDIG'],
                'type': edu_type,
                'main_address': self.get_address_info(inst),
                'status': 'Active' if inst['IND_OPGEHEVEN'] == 'N' else 'Closed',
                'start_date': inst['DT_IN_BEDRIJF'],
                'end_date': inst['DT_UIT_BEDRIJF'] if pd.notna(inst['DT_UIT_BEDRIJF']) else None
            }
            
            # Get branches
            branches = self.get_branches(inst['NR_ADMINISTRATIE'])
            if branches:
                institution_info['branches'] = branches
            
            # Get transitions
            transitions = self.get_transitions(inst['NR_ADMINISTRATIE'])
            if transitions:
                institution_info['transitions'] = transitions
            
            # Get board
            board = self.get_board_info(inst['NR_ADMINISTRATIE'])
            if board:
                institution_info['board'] = board
            
            results.append(institution_info)
        
        return {
            'statistics': stats,
            'institutions': results
        }

def main():
    # Use existing directory structure
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / 'data' / 'raw'
    processed_data_dir = project_root / 'data' / 'processed'
    
    # Initialize processor
    processor = EducationDataProcessor(str(raw_data_dir))
    
    # Process each education type
    all_results = {}
    for edu_type in processor.EDUCATION_TYPES.keys():
        results = processor.process_education_type(edu_type)
        all_results[edu_type] = results
    
    # Save combined results
    output_file = processed_data_dir / 'education_data_all.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()