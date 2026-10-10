import pandas as pd
import numpy as np
import yaml

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_team_past_stats(df, team, date, n_matches=5):
    """
    Get the statistics of a team for the last `n_matches` strictly BEFORE `date`.
    """
    past_matches = df[((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & (df['Date'] < date)].tail(n_matches)
    
    if len(past_matches) == 0:
        return {
            'goals_scored_avg': 0,
            'goals_conceded_avg': 0,
            'points_avg': 0,
            'xG_avg': 0,
            'xGA_avg': 0,
            'xG_diff_avg': 0
        }
        
    goals_scored = 0
    goals_conceded = 0
    points = 0
    xg = 0
    xga = 0
    
    for _, match in past_matches.iterrows():
        if match['HomeTeam'] == team:
            goals_scored += match['FTHG']
            goals_conceded += match['FTAG']
            xg += match.get('xG_h', match['FTHG'])
            xga += match.get('xG_a', match['FTAG'])
            if match['target'] == 0: points += 3
            elif match['target'] == 1: points += 1
        else:
            goals_scored += match['FTAG']
            goals_conceded += match['FTHG']
            xg += match.get('xG_a', match['FTAG'])
            xga += match.get('xG_h', match['FTHG'])
            if match['target'] == 2: points += 3
            elif match['target'] == 1: points += 1
            
    num = len(past_matches)
    xg_avg = xg / num
    xga_avg = xga / num
    return {
        'goals_scored_avg': goals_scored / num,
        'goals_conceded_avg': goals_conceded / num,
        'points_avg': points / num,
        'xG_avg': xg_avg,
        'xGA_avg': xga_avg,
        'xG_diff_avg': xg_avg - xga_avg
    }

def create_match_features(df):
    print("Generating rolling features... This may take a moment.")
    features_list = []
    
    # We must sort chronologically to ensure rolling features are correct
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    for idx, row in df.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = row['Date']
        
        home_stats = get_team_past_stats(df, home_team, match_date, n_matches=5)
        away_stats = get_team_past_stats(df, away_team, match_date, n_matches=5)
        
        feature_row = {
            'Date': row['Date'],
            'League': row.get('League', 'Unknown'),
            'Season': row.get('Season', 'Unknown'),
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'target': row['target'],
            
            'Home_Gls_Scored_avg': home_stats['goals_scored_avg'],
            'Home_Gls_Conceded_avg': home_stats['goals_conceded_avg'],
            'Home_Pts_avg': home_stats['points_avg'],
            'Home_xG_avg': home_stats['xG_avg'],
            'Home_xGA_avg': home_stats['xGA_avg'],
            'Home_xG_diff_avg': home_stats['xG_diff_avg'],
            
            'Away_Gls_Scored_avg': away_stats['goals_scored_avg'],
            'Away_Gls_Conceded_avg': away_stats['goals_conceded_avg'],
            'Away_Pts_avg': away_stats['points_avg'],
            'Away_xG_avg': away_stats['xG_avg'],
            'Away_xGA_avg': away_stats['xGA_avg'],
            'Away_xG_diff_avg': away_stats['xG_diff_avg'],
            
            'Form_Advantage': home_stats['points_avg'] - away_stats['points_avg'],
            'xG_Advantage': home_stats['xG_diff_avg'] - away_stats['xG_diff_avg'],
        }
        
        # Add optional lineup features if present
        lineup_cols = [
            'Home_Missing_Count', 'Away_Missing_Count',
            'Home_Missing_Key_Players', 'Away_Missing_Key_Players',
            'Home_Lineup_Continuity', 'Away_Lineup_Continuity'
        ]
        for col in lineup_cols:
            if col in df.columns:
                feature_row[col] = row[col]
        
        # Add odds if they exist
        if 'B365H' in df.columns:
            feature_row['B365H'] = row['B365H']
            feature_row['B365D'] = row['B365D']
            feature_row['B365A'] = row['B365A']
            
        features_list.append(feature_row)
        
    features_df = pd.DataFrame(features_list)
    
    # Drop rows where teams have no past history (e.g. first match of the dataset)
    # Actually, we can keep them with 0s, but they might add noise. Let's keep them for now.
    
    return features_df

if __name__ == "__main__":
    print("Starting feature engineering on historical data...")
    config = load_config()
    
    df = pd.read_csv("data/processed/cleaned_matches.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    
    features_df = create_match_features(df)
    
    output_path = config['paths']['processed_matches']
    features_df.to_csv(output_path, index=False)
    print(f"Feature engineering complete. Saved to {output_path}. Shape: {features_df.shape}")
