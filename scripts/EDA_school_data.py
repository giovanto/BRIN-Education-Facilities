import pandas as pd
import numpy as np
from datetime import datetime

def build_vos_dataset():
    """Build comprehensive dataset for secondary schools with better progress tracking"""
    
    print("Loading data files...")
    org_df = pd.read_csv('ORGANISATIES_20250203.csv', sep=',', encoding='utf-8', quotechar='"', low_memory=False)
    rel_df = pd.read_csv('RELATIES_20250203.csv', sep=',', encoding='utf-8', quotechar='"')
    ovg_df = pd.read_csv('OVERGANGEN_20250203.csv', sep=',', encoding='utf-8', quotechar='"')
    
    # 1. Get all VOS main institutions
    vos_schools = org_df[
        (org_df['CODE_SOORT'] == 'VOS') &
        (org_df['CODE_FUNCTIE'] == 'U')
    ].copy()
    
    print("\nSchool Statistics:")
    print(f"Total secondary schools found: {len(vos_schools)}")
    print("\nBy status:")
    print(vos_schools['CODE_STAND_RECORD'].value_counts())
    print("\nBy operational status:")
    print(vos_schools['IND_OPGEHEVEN'].value_counts())
    
    # 2. Process current schools
    current_schools = vos_schools[vos_schools['CODE_STAND_RECORD'] == 'A'].copy()
    print(f"\nProcessing {len(current_schools)} current schools...")
    
    results = []
    for idx, school in current_schools.iterrows():
        if idx % 50 == 0:  # Progress update every 50 schools
            print(f"Processing school {idx}/{len(current_schools)}")
        
        # Basic school info
        school_info = {
            'school_id': school['NR_ADMINISTRATIE'],
            'name': school['NAAM_VOLLEDIG'],
            'main_address': {
                'street': school['NAAM_STRAAT_VEST'],
                'number': school['NR_HUIS_VEST'],
                'postcode': school['POSTCODE_VEST'],
                'city': school['NAAM_PLAATS_VEST'],
                'province': school['PROVINCIE_VEST']
            },
            'status': 'Active' if school['IND_OPGEHEVEN'] == 'N' else 'Closed',
            'start_date': school['DT_IN_BEDRIJF'],
            'end_date': school['DT_UIT_BEDRIJF'] if pd.notna(school['DT_UIT_BEDRIJF']) else None
        }
        
        # Get branch locations (current only)
        branches = rel_df[
            (rel_df['NR_ADMIN_LEID'] == school['NR_ADMINISTRATIE']) &
            (rel_df['NAAM_RELATIE_LEID'] == 'vestigt') &
            (rel_df['CODE_STAND_RECORD'] == 'A')
        ]
        
        branch_locations = []
        for _, branch in branches.iterrows():
            branch_org = org_df[org_df['NR_ADMINISTRATIE'] == branch['NR_ADMIN_VOLG']]
            if not branch_org.empty:
                branch_org = branch_org.iloc[0]
                branch_locations.append({
                    'branch_id': branch['NR_ADMIN_VOLG'],
                    'address': {
                        'street': branch_org['NAAM_STRAAT_VEST'],
                        'number': branch_org['NR_HUIS_VEST'],
                        'postcode': branch_org['POSTCODE_VEST'],
                        'city': branch_org['NAAM_PLAATS_VEST']
                    },
                    'start_date': branch['DT_BEGIN_RELATIE'],
                    'end_date': branch['DT_EINDE_RELATIE'] if pd.notna(branch['DT_EINDE_RELATIE']) else None
                })
        if branch_locations:
            school_info['branches'] = branch_locations
        
        # Get all transitions (historical and current)
        transitions = ovg_df[
            (ovg_df['NR_ADMIN_NAAR'] == school['NR_ADMINISTRATIE']) |
            (ovg_df['NR_ADMIN_VAN'] == school['NR_ADMINISTRATIE'])
        ]
        
        if not transitions.empty:
            transition_history = []
            for _, trans in transitions.iterrows():
                other_id = trans['NR_ADMIN_VAN'] if trans['NR_ADMIN_NAAR'] == school['NR_ADMINISTRATIE'] else trans['NR_ADMIN_NAAR']
                other_org = org_df[org_df['NR_ADMINISTRATIE'] == other_id]
                if not other_org.empty:
                    other_org = other_org.iloc[0]
                    transition_history.append({
                        'type': trans['NAAM_OVERGANG'],
                        'date': trans['DT_OVERGANG'],
                        'direction': 'from' if trans['NR_ADMIN_NAAR'] == school['NR_ADMINISTRATIE'] else 'to',
                        'other_institution': {
                            'id': other_id,
                            'name': other_org['NAAM_VOLLEDIG'],
                            'type': other_org['CODE_SOORT']
                        }
                    })
            if transition_history:
                school_info['transitions'] = sorted(transition_history, key=lambda x: x['date'])
        
        # Get current board
        board = rel_df[
            (rel_df['NR_ADMIN_VOLG'] == school['NR_ADMINISTRATIE']) &
            (rel_df['NAAM_RELATIE_VOLG'] == 'wordt bestuurd door') &
            (rel_df['CODE_STAND_RECORD'] == 'A')
        ]
        
        if not board.empty:
            board_org = org_df[org_df['NR_ADMINISTRATIE'] == board.iloc[0]['NR_ADMIN_LEID']]
            if not board_org.empty:
                board_org = board_org.iloc[0]
                school_info['board'] = {
                    'id': board_org['NR_ADMINISTRATIE'],
                    'name': board_org['NAAM_VOLLEDIG'],
                    'start_date': board.iloc[0]['DT_BEGIN_RELATIE'],
                    'end_date': board.iloc[0]['DT_EINDE_RELATIE'] if pd.notna(board.iloc[0]['DT_EINDE_RELATIE']) else None
                }
        
        results.append(school_info)
    
    print(f"\nProcessing complete. Found {len(results)} schools with complete information.")
    return results

def print_school_summary(school):
    """Print a formatted summary of a school's information"""
    print("\n" + "="*50)
    print(f"School: {school['name']}")
    print(f"ID: {school['school_id']}")
    print(f"Status: {school['status']}")
    
    print("\nMain Address:")
    addr = school['main_address']
    print(f"{addr['street']} {addr['number']}")
    print(f"{addr['postcode']} {addr['city']}")
    print(f"Province: {addr['province']}")
    
    if 'branches' in school:
        print(f"\nBranches ({len(school['branches'])}):")
        for branch in school['branches']:
            print(f"- {branch['address']['street']} {branch['address']['number']}, {branch['address']['city']}")
    
    if 'transitions' in school:
        print(f"\nTransitions ({len(school['transitions'])}):")
        for trans in school['transitions']:
            print(f"- {trans['date']}: {trans['type']} {trans['direction']} {trans['other_institution']['name']}")
    
    if 'board' in school:
        print(f"\nBoard: {school['board']['name']}")

def main():
    schools = build_vos_dataset()
    
    # Print detailed information for first 3 schools
    print("\n=== Sample Detailed School Information ===")
    for school in schools[:3]:
        print_school_summary(school)
    
    # Print some statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total schools processed: {len(schools)}")
    
    schools_with_branches = len([s for s in schools if 'branches' in s])
    print(f"Schools with branches: {schools_with_branches}")
    
    schools_with_transitions = len([s for s in schools if 'transitions' in s])
    print(f"Schools with transitions: {schools_with_transitions}")

if __name__ == "__main__":
    main()