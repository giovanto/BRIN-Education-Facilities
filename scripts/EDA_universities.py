import pandas as pd

def extract_current_universities():
    # Read all files
    print("Reading data files...")
    org_df = pd.read_csv('ORGANISATIES_20250203.csv', sep=',', encoding='utf-8', quotechar='"', low_memory=False)
    rel_df = pd.read_csv('RELATIES_20250203.csv', sep=',', encoding='utf-8', quotechar='"')
    ovg_df = pd.read_csv('OVERGANGEN_20250203.csv', sep=',', encoding='utf-8', quotechar='"')
    
    # Initial filter for universities
    universities = org_df[
        (org_df['CODE_SOORT'] == 'UNIV') &
        (org_df['CODE_FUNCTIE'] == 'U') &
        (org_df['CODE_STAND_RECORD'] == 'A') &
        (org_df['IND_OPGEHEVEN'] == 'N')
    ].copy()
    
    print(f"\nInitial university count from ORGANISATIES: {len(universities)}")
    
    # Check relationships
    print("\nAnalyzing relationships...")
    for idx, univ in universities.iterrows():
        # Check if university is involved in relationships
        related_leid = rel_df[
            (rel_df['NR_ADMIN_LEID'] == univ['NR_ADMINISTRATIE']) &
            (rel_df['CODE_STAND_RECORD'] == 'A')
        ]
        related_volg = rel_df[
            (rel_df['NR_ADMIN_VOLG'] == univ['NR_ADMINISTRATIE']) &
            (rel_df['CODE_STAND_RECORD'] == 'A')
        ]
        
        print(f"\nUniversity: {univ['NAAM_VOLLEDIG']}")
        print(f"Leading relationships: {len(related_leid)}")
        if len(related_leid) > 0:
            print("Types:", related_leid['NAAM_RELATIE_LEID'].unique())
        print(f"Following relationships: {len(related_volg)}")
        if len(related_volg) > 0:
            print("Types:", related_volg['NAAM_RELATIE_VOLG'].unique())
    
    # Check transitions
    print("\nAnalyzing transitions...")
    recent_transitions = ovg_df[
        (ovg_df['CODE_STAND_RECORD'].isin(['A', 'T'])) &
        ((ovg_df['CODE_SOORT_VAN'] == 'UNIV') | (ovg_df['CODE_SOORT_NAAR'] == 'UNIV'))
    ]
    
    if len(recent_transitions) > 0:
        print("\nFound recent transitions affecting universities:")
        for _, trans in recent_transitions.iterrows():
            print(f"Type: {trans['NAAM_OVERGANG']}")
            print(f"Date: {trans['DT_OVERGANG']}")
    else:
        print("\nNo recent transitions found affecting universities")
    
    # Select relevant columns for final output
    columns = [
        'NR_ADMINISTRATIE', 'NAAM_VOLLEDIG',
        'NAAM_STRAAT_VEST', 'NR_HUIS_VEST', 'NR_HUIS_TOEV_VEST',
        'POSTCODE_VEST', 'NAAM_PLAATS_VEST', 'PROVINCIE_VEST',
        'DT_IN_BEDRIJF'
    ]
    
    return universities[columns]

if __name__ == "__main__":
    universities = extract_current_universities()