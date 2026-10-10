import os
import requests
import pandas as pd

def download_football_data(seasons, leagues, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = []
    
    base_url = "https://www.football-data.co.uk/mmz4281"
    
    for season in seasons:
        for league, league_code in leagues.items():
            url = f"{base_url}/{season}/{league_code}.csv"
            print(f"Downloading {league} for season {season} from {url}...")
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    filename = os.path.join(output_dir, f"{league_code}_{season}.csv")
                    with open(filename, 'wb') as f:
                        f.write(response.content)
                    
                    df = pd.read_csv(filename, encoding='latin1')
                    df['Season'] = season
                    df['League'] = league
                    all_data.append(df)
                    print(f"Successfully loaded {len(df)} matches.")
                else:
                    print(f"Failed to download {url}. Status code: {response.status_code}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")
                
    if all_data:
        # Common columns to keep
        cols = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'Season', 'League']
        # Odds columns if they exist
        odds_cols = ['B365H', 'B365D', 'B365A']
        
        final_dfs = []
        for df in all_data:
            keep_cols = [c for c in cols if c in df.columns]
            for o in odds_cols:
                if o in df.columns:
                    keep_cols.append(o)
            final_dfs.append(df[keep_cols])
            
        combined_df = pd.concat(final_dfs, ignore_index=True)
        combined_df.to_csv(os.path.join(output_dir, "historical_matches.csv"), index=False)
        print(f"Combined data saved to historical_matches.csv with {len(combined_df)} matches.")
    else:
        print("No data was downloaded.")

if __name__ == "__main__":
    # Seasons: 2425 (2024/2025), 2526 (2025/2026)
    seasons = ["2425", "2526"]
    leagues = {
        "Premier League": "E0",
        "Ligue 1": "F1",
        "Bundesliga": "D1",
        "Serie A": "I1",
        "La Liga": "SP1"
    }
    download_football_data(seasons, leagues)
