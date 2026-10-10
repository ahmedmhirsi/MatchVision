import pandas as pd
import numpy as np
import yaml
import os

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def clean_match_data(df, df_xg=None):
    print("Initial shape:", df.shape)
    
    # Drop rows with missing essential data
    df = df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
    
    # Parse Date
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
    
    if df_xg is not None:
        df_xg['Date'] = pd.to_datetime(df_xg['Date'])
        # Map Understat team names to football-data.co.uk names
        team_mapping = {
            'FC Heidenheim': 'Heidenheim', 'Manchester United': 'Man United', 
            'Bayer Leverkusen': 'Leverkusen', 'Paris Saint Germain': 'Paris SG', 
            'Manchester City': 'Man City', 'Real Oviedo': 'Oviedo', 
            'RasenBallsport Leipzig': 'RB Leipzig', 'Newcastle United': 'Newcastle', 
            'Eintracht Frankfurt': 'Ein Frankfurt', 'Hamburger SV': 'Hamburg', 
            'Real Valladolid': 'Valladolid', 'VfB Stuttgart': 'Stuttgart', 
            'Mainz 05': 'Mainz', 'Celta Vigo': 'Celta', 
            'Nottingham Forest': "Nott'm Forest", 'Parma Calcio 1913': 'Parma', 
            'Athletic Club': 'Ath Bilbao', 'Rayo Vallecano': 'Vallecano', 
            'Saint-Etienne': 'St Etienne', 'AC Milan': 'Milan', 
            'FC Cologne': 'FC Koln', 'Wolverhampton Wanderers': 'Wolves', 
            'Espanyol': 'Espanol', 'Atletico Madrid': 'Ath Madrid', 
            'Borussia Dortmund': 'Dortmund', 'Borussia M.Gladbach': "M'gladbach", 
            'Real Betis': 'Betis', 'St. Pauli': 'St Pauli', 'Real Sociedad': 'Sociedad'
        }
        df_xg['HomeTeam_mapped'] = df_xg['HomeTeam_xG'].replace(team_mapping)
        df_xg['AwayTeam_mapped'] = df_xg['AwayTeam_xG'].replace(team_mapping)
        
        # Merge by Date and mapped HomeTeam
        df = pd.merge(
            df, 
            df_xg[['Date', 'HomeTeam_mapped', 'AwayTeam_mapped', 'xG_h', 'xG_a']], 
            left_on=['Date', 'HomeTeam', 'AwayTeam'], 
            right_on=['Date', 'HomeTeam_mapped', 'AwayTeam_mapped'], 
            how='left'
        )
        df = df.drop(columns=['HomeTeam_mapped', 'AwayTeam_mapped'])
        
        # Fill missing xG with actual goals as fallback
        missing_xg = df['xG_h'].isna().sum()
        if missing_xg > 0:
            print(f"Warning: {missing_xg} matches missing xG data. Filling with actual goals.")
            df['xG_h'] = df['xG_h'].fillna(df['FTHG'])
            df['xG_a'] = df['xG_a'].fillna(df['FTAG'])
    
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
    
    # Load xG data if available
    xg_path = "data/raw/understat_matches.csv"
    if os.path.exists(xg_path):
        df_xg = pd.read_csv(xg_path)
    else:
        df_xg = None
        
    clean_df = clean_match_data(raw_df, df_xg)
    
    os.makedirs(os.path.dirname(config['paths']['processed_matches']), exist_ok=True)
    # Save intermediate cleaned data
    clean_df.to_csv("data/processed/cleaned_matches.csv", index=False)
    print("Saved cleaned matches to data/processed/cleaned_matches.csv")
