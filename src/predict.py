import pandas as pd
import numpy as np
import yaml
import joblib
import sys

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def predict_match(home_team, away_team, config):
    try:
        model = joblib.load(config['paths']['models_dir'] + "best_model.pkl")
        squads_df = pd.read_csv(config['paths']['processed_squads'])
    except Exception as e:
        print(f"Error loading models or data: {e}")
        return
        
    home_stats = squads_df[squads_df['Squad'] == home_team]
    away_stats = squads_df[squads_df['Squad'] == away_team]
    
    if home_stats.empty or away_stats.empty:
        print("Data is insufficient for these teams.")
        print(f"Available teams: {squads_df['Squad'].unique()[:5]}...")
        return
        
    features = {
        'Home_Off_Str': home_stats['Offensive_Strength'].values[0],
        'Home_Def_Str': home_stats['Defensive_Strength'].values[0],
        'Home_Gls_avg': home_stats['Gls_per_match'].values[0],
        'Home_SoT_avg': home_stats['SoT_per_match'].values[0],
        'Away_Off_Str': away_stats['Offensive_Strength'].values[0],
        'Away_Def_Str': away_stats['Defensive_Strength'].values[0],
        'Away_Gls_avg': away_stats['Gls_per_match'].values[0],
        'Away_SoT_avg': away_stats['SoT_per_match'].values[0],
    }
    
    features['Off_Advantage'] = features['Home_Off_Str'] - features['Away_Off_Str']
    features['Def_Advantage'] = features['Home_Def_Str'] - features['Away_Def_Str']
    
    X = pd.DataFrame([features])
    
    probs = model.predict_proba(X)[0]
    
    print("\n--- Match Prediction ---")
    print(f"Home: {home_team} vs Away: {away_team}")
    print(f"Prob. Home Win (1): {probs[0]*100:.2f}%")
    print(f"Prob. Draw (X):     {probs[1]*100:.2f}%")
    print(f"Prob. Away Win (2): {probs[2]*100:.2f}%")
    
    predictions = ["Home Win", "Draw", "Away Win"]
    main_pred = predictions[np.argmax(probs)]
    max_prob = np.max(probs)
    
    confidence = "Low"
    if max_prob > 0.6: confidence = "High"
    elif max_prob > 0.45: confidence = "Medium"
    
    print(f"Main Prediction: {main_pred}")
    print(f"Statistical Confidence: {confidence} (based on maximum probability margin)")
    
    print("\nNote: Models currently trained on simulated mock schedule due to missing historical match data.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/predict.py 'Home Team' 'Away Team'")
        sys.exit(1)
        
    config = load_config()
    predict_match(sys.argv[1], sys.argv[2], config)
