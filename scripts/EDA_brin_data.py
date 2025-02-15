import pandas as pd
import numpy as np
from datetime import datetime

def analyze_file(file_path, file_type):
    """Analyze a single BRIN data file"""
    print(f"\n{'='*50}")
    print(f"Analyzing {file_type}")
    print(f"{'='*50}")
    
    try:
        # Read the file with explicit UTF-8 encoding and comma separator
        df = pd.read_csv(file_path, sep=',', encoding='utf-8', quotechar='"', low_memory=False)
        
        # Basic information
        print(f"\n1. Basic Information:")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        # Column analysis
        print("\n2. Columns Found:")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            print(f"\n{col}:")
            print(f"  - Null values: {null_count} ({(null_count/len(df)*100):.2f}%)")
            print(f"  - Unique values: {unique_count}")
            
            # For key classification fields, show value distribution
            if col in ['CODE_FUNCTIE', 'CODE_SOORT', 'CODE_STAND_RECORD', 'IND_OPGEHEVEN']:
                value_counts = df[col].value_counts().head(10)
                print("  - Top values:")
                for val, count in value_counts.items():
                    print(f"    {val}: {count}")
        
        # Special analysis for each file type
        if file_type == 'ORGANISATIES':
            # Analyze active vs inactive organizations
            active = df[df['IND_OPGEHEVEN'] == 'N'].shape[0]
            inactive = df[df['IND_OPGEHEVEN'] == 'J'].shape[0]
            print(f"\n3. Organization Status:")
            print(f"Active organizations: {active}")
            print(f"Inactive organizations: {inactive}")
            
        elif file_type == 'RELATIES':
            # Analyze relationship types
            if 'NAAM_RELATIE_LEID' in df.columns:
                print("\n3. Relationship Types:")
                print(df['NAAM_RELATIE_LEID'].value_counts())
                
        elif file_type == 'OVERGANGEN':
            # Analyze transition types
            if 'CODE_OVERGANG' in df.columns:
                print("\n3. Transition Types:")
                print(df['CODE_OVERGANG'].value_counts())
                
        return df
        
    except Exception as e:
        print(f"Error analyzing {file_type}: {str(e)}")
        return None

def main():
    # Analyze all three files
    try:
        org_df = analyze_file('ORGANISATIES_20250203.csv', 'ORGANISATIES')
        rel_df = analyze_file('RELATIES_20250203.csv', 'RELATIES')
        ovg_df = analyze_file('OVERGANGEN_20250203.csv', 'OVERGANGEN')
        
        # Cross-file validation if all files were loaded successfully
        if all([org_df is not None, rel_df is not None, ovg_df is not None]):
            print("\nCross-file Validation:")
            
            # Check if all organizations in relations exist in organizations file
            if 'NR_ADMIN_LEID' in rel_df.columns and 'NR_ADMINISTRATIE' in org_df.columns:
                leid_orgs = set(rel_df['NR_ADMIN_LEID'].unique())
                all_orgs = set(org_df['NR_ADMINISTRATIE'].unique())
                missing_orgs = leid_orgs - all_orgs
                if missing_orgs:
                    print(f"\nWarning: Found {len(missing_orgs)} leading organizations in RELATIES not present in ORGANISATIES")
                
    except Exception as e:
        print(f"Error in main analysis: {str(e)}")

if __name__ == "__main__":
    main()