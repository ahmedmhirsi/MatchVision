import os
import pandas as pd
from understatapi import UnderstatClient

def download_understat_data(seasons, leagues, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    all_matches = []
    
    with UnderstatClient() as client:
        for season in seasons:
            for league_name, league_code in leagues.items():
                print(f"Downloading xG for {league_name} season {season}...")
                try:
                    data = client.league(league=league_name).get_match_data(season=season)
                    for match in data:
                        if match['isResult']:
                            all_matches.append({
                                'League': league_name.replace('_', ' ') if league_name != "EPL" else "Premier League",
                                'Season': str(int(season) - 2000) + str(int(season) - 2000 + 1), # e.g. 2024 -> 2425
                                'Date': match['datetime'].split(' ')[0], # YYYY-MM-DD
                                'HomeTeam_xG': match['h']['title'],
                                'AwayTeam_xG': match['a']['title'],
                                'FTHG': int(match['goals']['h']),
                                'FTAG': int(match['goals']['a']),
                                'xG_h': float(match['xG']['h']),
                                'xG_a': float(match['xG']['a'])
                            })
                except Exception as e:
                    print(f"Error fetching {league_name} {season}: {e}")
                    
    df = pd.DataFrame(all_matches)
    df.to_csv(os.path.join(output_dir, "understat_matches.csv"), index=False)
    print(f"Saved {len(df)} xG matches to {output_dir}/understat_matches.csv")

if __name__ == "__main__":
    # Understat seasons use the starting year, e.g. 2024 for 2425
    seasons = ["2024", "2025"]
    leagues = {
        "EPL": "E0",
        "Ligue_1": "F1",
        "Bundesliga": "D1",
        "Serie_A": "I1",
        "La_Liga": "SP1"
    }
    download_understat_data(seasons, leagues)
