import pandas as pd
import numpy as np
import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_match_data(df):
    print("Initial shape:", df.shape)
    
    # Drop rows with missing essential data
    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
    
    # Parse Date
    # football-data.co.uk dates can be DD/MM/YYYY or DD/MM/YY
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    
    # Sort chronologically
    df = df.sort_values(by=['Date', 'Time'] if 'Time' in df.columns else ['Date']).reset_index(drop=True)
    
    # Map target
    target_map = {'H': 0, 'D': 1, 'A': 2}
    df['target'] = df['FTR'].map(target_map)
    
    # Fill missing odds with median if they exist
    for col in ['B365H', 'B365D', 'B365A']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            
    # Drop exact duplicates
    df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'])
    
    print("Cleaned shape:", df.shape)
    return df

if __name__ == "__main__":
    print("Starting data cleaning for historical matches...")
    config = load_config()
    
    raw_df = pd.read_csv(config['paths']['raw_data'])
    clean_df = clean_match_data(raw_df)
    
    os.makedirs(os.path.dirname(config['paths']['processed_matches']), exist_ok=True)
    # Save intermediate cleaned data
    clean_df.to_csv("data/processed/cleaned_matches.csv", index=False)
    print("Saved cleaned matches to data/processed/cleaned_matches.csv")
