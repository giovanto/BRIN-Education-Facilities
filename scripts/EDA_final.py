import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_education_system():
    """
    Comprehensive analysis of Dutch education system locations and relationships.
    
    Key Questions:
    1. Where do students actually go to school? (teaching vs administrative locations)
    2. How are different types of locations (VST, LOC, DIS, etc.) used across education types?
    3. What's the relationship between main institutions and their branches?
    4. How do we interpret different status codes (A/H/T and J/N)?
    5. Are we missing important locations by focusing on wrong codes?
    """
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    raw_data_dir = project_root / 'data' / 'raw'
    
    print("Loading data files...")
    # Load with low_memory=False to prevent mixed type inference
    org_df = pd.read_csv(raw_data_dir / 'ORGANISATIES_20250203.csv', 
                        sep=',', encoding='utf-8', quotechar='"', low_memory=False)
    rel_df = pd.read_csv(raw_data_dir / 'RELATIES_20250203.csv', 
                        sep=',', encoding='utf-8', quotechar='"')
    
    # Print overall dataset statistics
    print("\n=== Overall Dataset Statistics ===")
    print(f"Total organizations: {len(org_df)}")
    print(f"Total relationships: {len(rel_df)}")
    
    # Analyze function codes
    print("\n=== Function Code Distribution ===")
    print("CODE_FUNCTIE meanings:")
    print("U = uitvoering (main institution)")
    print("D = onderdeel (department/branch)")
    print("B = bestuur (board)")
    print("S = samenwerking (collaboration)")
    print("\nDistribution:")
    print(org_df['CODE_FUNCTIE'].value_counts())
    
    # Analyze record and operational status
    print("\n=== Status Code Analysis ===")
    print("CODE_STAND_RECORD (Record Status):")
    print("A = current")
    print("H = historical")
    print("T = future")
    print("\nIND_OPGEHEVEN (Operational Status):")
    print("N = operating")
    print("J = discontinued")
    
    status_cross = pd.crosstab(org_df['CODE_STAND_RECORD'], 
                              org_df['IND_OPGEHEVEN'])
    print("\nStatus Cross-tabulation:")
    print(status_cross)
    
    # Analyze relationship types
    print("\n=== Relationship Types ===")
    print(rel_df['NAAM_RELATIE_LEID'].value_counts())
    
    # Main education types analysis
    edu_types = {
        'BAS': 'Primary Education',
        'VOS': 'Secondary Education',
        'HBOS': 'Higher Professional Education',
        'ROC': 'Regional Education Center',
        'UNIV': 'University'
    }
    
    for edu_type, edu_name in edu_types.items():
        print(f"\n{'='*20} {edu_type} ({edu_name}) Analysis {'='*20}")
        
        # 1. Main Institutions Analysis
        main_insts = org_df[
            (org_df['CODE_SOORT'] == edu_type) &
            (org_df['CODE_FUNCTIE'] == 'U')
        ]
        
        print(f"\nMain Institutions Analysis ({len(main_insts)}):")
        
        # Record status distribution
        print("\nRecord Status Distribution:")
        record_status = main_insts['CODE_STAND_RECORD'].value_counts()
        print(record_status)
        
        # Operational status
        print("\nOperational Status:")
        op_status = main_insts['IND_OPGEHEVEN'].value_counts()
        print(op_status)
        
        # Currently operating institutions
        current_active = main_insts[
            (main_insts['CODE_STAND_RECORD'].isin(['A', 'T'])) &
            (main_insts['IND_OPGEHEVEN'] == 'N')
        ]
        print(f"\nCurrently operating institutions: {len(current_active)}")
        
        # 2. Branch Location Analysis
        inst_ids = main_insts['NR_ADMINISTRATIE'].unique()
        
        # Get branch relationships
        branches = rel_df[
            (rel_df['NR_ADMIN_LEID'].isin(inst_ids)) &
            (rel_df['NAAM_RELATIE_LEID'] == 'vestigt')
        ]
        
        # Get branch location details
        branch_locations = org_df[
            org_df['NR_ADMINISTRATIE'].isin(branches['NR_ADMIN_VOLG'])
        ]
        
        print(f"\nBranch Analysis ({len(branch_locations)} total):")
        
        # Location type distribution
        print("\nLocation Type Distribution:")
        print(branch_locations['CODE_SOORT'].value_counts())
        
        # Calculate branches per institution
        branches_per_inst = branches.groupby('NR_ADMIN_LEID').size()
        print("\nBranches per Institution:")
        print(f"Average: {branches_per_inst.mean():.2f}")
        print(f"Median: {branches_per_inst.median()}")
        print(f"Max: {branches_per_inst.max()}")
        
        # 3. Sample Analysis of Large Institution
        if len(current_active) > 0:
            # Get institution with most branches
            most_branches_id = branches_per_inst.idxmax()
            sample_inst = main_insts[
                main_insts['NR_ADMINISTRATIE'] == most_branches_id
            ].iloc[0]
            
            print(f"\nLargest Institution Analysis:")
            print(f"Name: {sample_inst['NAAM_VOLLEDIG']}")
            
            # Get its branches
            inst_branches = branches[branches['NR_ADMIN_LEID'] == most_branches_id]
            branch_details = org_df[
                org_df['NR_ADMINISTRATIE'].isin(inst_branches['NR_ADMIN_VOLG'])
            ]
            
            print(f"Total branches: {len(branch_details)}")
            
            # Analyze branch types
            print("\nBranch type distribution:")
            print(branch_details['CODE_SOORT'].value_counts())
            
            # Get unique cities
            print("\nCities covered:")
            print(branch_details['NAAM_PLAATS_VEST'].value_counts())
        
        # 4. Additional Location Analysis
        if edu_type in ['VOS', 'HBOS', 'ROC']:
            # Define keywords for each type
            keywords = {
                'VOS': ['vmbo', 'havo', 'vwo', 'lyceum', 'college', 'atheneum', 'gymnasium'],
                'HBOS': ['hogeschool', 'hbo', 'academie', 'conservatorium'],
                'ROC': ['roc', 'mbo', 'beroepsonderwijs', 'vakschool']
            }
            
            # Find potential locations under different codes
            other_locations = org_df[
                ~org_df['NR_ADMINISTRATIE'].isin(inst_ids) &
                org_df['NAAM_VOLLEDIG'].str.lower().str.contains('|'.join(keywords[edu_type]), na=False)
            ]
            
            print(f"\nPotential Additional Locations:")
            print("By Type:")
            print(other_locations['CODE_SOORT'].value_counts().head())
            print("\nBy Function:")
            print(other_locations['CODE_FUNCTIE'].value_counts())

if __name__ == "__main__":
    analyze_education_system()