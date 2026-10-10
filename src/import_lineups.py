"""
import_lineups.py - Optional pipeline step for Squad & Injury data.

Because historical lineups, injuries, and suspensions with STRICT pre-match 
timestamps are not available for free (and scraping violates many terms of service
or risks post-match data leakage), this script provides a template to import 
and merge a user-provided CSV dataset.

Expected CSV format (`data/raw/lineups_injuries.csv`):
- Date (YYYY-MM-DD)
- HomeTeam (String, must match football-data.co.uk names)
- AwayTeam (String)
- Home_Missing_Count (Int)
- Away_Missing_Count (Int)
- Home_Missing_Key_Players (Int - e.g. starters missing)
- Away_Missing_Key_Players (Int)
- Home_Lineup_Continuity (Float - percentage of previous match starters kept)
- Away_Lineup_Continuity (Float)

If the file is provided, this script merges it into `cleaned_matches.csv`.
"""

import os
import pandas as pd
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def merge_lineup_data(cleaned_df, lineup_df):
    """
    Merge the lineup/injury data into the cleaned match data.
    """
    # Ensure dates match format
    lineup_df['Date'] = pd.to_datetime(lineup_df['Date'])
    cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
    
    # Merge on Date, HomeTeam, AwayTeam
    merged = pd.merge(
        cleaned_df, 
        lineup_df, 
        on=['Date', 'HomeTeam', 'AwayTeam'], 
        how='left'
    )
    
    # Fill missing values with a neutral baseline (e.g., 0 missing)
    fill_cols = [
        'Home_Missing_Count', 'Away_Missing_Count', 
        'Home_Missing_Key_Players', 'Away_Missing_Key_Players'
    ]
    for col in fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
            
    if 'Home_Lineup_Continuity' in merged.columns:
        merged['Home_Lineup_Continuity'] = merged['Home_Lineup_Continuity'].fillna(1.0)
    if 'Away_Lineup_Continuity' in merged.columns:
        merged['Away_Lineup_Continuity'] = merged['Away_Lineup_Continuity'].fillna(1.0)
        
    return merged

if __name__ == "__main__":
    config = load_config()
    
    cleaned_path = "data/processed/cleaned_matches.csv"
    lineup_path = "data/raw/lineups_injuries.csv"
    
    if not os.path.exists(cleaned_path):
        print(f"[ERROR] Cleaned matches not found at {cleaned_path}. Run data_cleaning.py first.")
        exit(1)
        
    if not os.path.exists(lineup_path):
        print(f"[INFO] Optional lineup dataset not found at {lineup_path}.")
        print("       See src/import_lineups.py for required format. Skipping merge.")
        exit(0)
        
    print(f"Loading {lineup_path}...")
    cleaned_df = pd.read_csv(cleaned_path)
    lineup_df = pd.read_csv(lineup_path)
    
    merged_df = merge_lineup_data(cleaned_df, lineup_df)
    
    # Save back to processed
    merged_df.to_csv(cleaned_path, index=False)
    print(f"Lineup data successfully merged. Saved to {cleaned_path}")
