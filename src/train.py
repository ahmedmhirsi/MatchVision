import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from evaluate import evaluate_model
import json

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class BaselineModel:
    """Simple statistical baseline predicting the historical marginal probabilities."""
    def __init__(self):
        self.probs = None
        self.classes_ = np.array([0, 1, 2])
        
    def fit(self, X, y):
        counts = y.value_counts(normalize=True).sort_index()
        self.probs = np.array([counts.get(0, 0.33), counts.get(1, 0.33), counts.get(2, 0.33)])
        
    def predict_proba(self, X):
        return np.tile(self.probs, (X.shape[0], 1))
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class MajorityClassBaseline:
    """Baseline that always predicts the majority class."""
    def __init__(self):
        self.majority_class = None
        self.classes_ = np.array([0, 1, 2])
        self.probs = np.zeros(3)
        
    def fit(self, X, y):
        counts = y.value_counts(normalize=True).sort_index()
        self.majority_class = counts.idxmax()
        self.probs[self.majority_class] = 1.0
        
    def predict_proba(self, X):
        return np.tile(self.probs, (X.shape[0], 1))
        
    def predict(self, X):
        return np.full(X.shape[0], self.majority_class)

if __name__ == "__main__":
    print("Starting training pipeline with real data...")
    config = load_config()
    
    df = pd.read_csv(config['paths']['processed_matches'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Optional: use odds to generate implied probabilities as features
    if 'B365H' in df.columns:
        df['Prob_H_implied'] = 1 / df['B365H']
        df['Prob_D_implied'] = 1 / df['B365D']
        df['Prob_A_implied'] = 1 / df['B365A']
        
        # Normalize to remove bookmaker margin
        margin = df['Prob_H_implied'] + df['Prob_D_implied'] + df['Prob_A_implied']
        df['Prob_H_implied'] /= margin
        df['Prob_D_implied'] /= margin
        df['Prob_A_implied'] /= margin

    # Drop early matches (where past stats are 0 because no history yet)
    df = df[df['Home_Pts_avg'] > 0].copy()
    
    # Chronological Split (Train 70%, Val 15%, Test 15%)
    df = df.sort_values('Date')
    n_samples = len(df)
    train_size = int(n_samples * 0.70)
    val_size = int(n_samples * 0.15)
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size+val_size]
    test_df = df.iloc[train_size+val_size:]
    
    feature_cols = [
        'Home_Gls_Scored_avg', 'Home_Gls_Conceded_avg', 'Home_Pts_avg',
        'Away_Gls_Scored_avg', 'Away_Gls_Conceded_avg', 'Away_Pts_avg',
        'Form_Advantage'
    ]
    
    if 'Prob_H_implied' in df.columns:
        feature_cols.extend(['Prob_H_implied', 'Prob_D_implied', 'Prob_A_implied'])
        
    target_col = 'target'
    
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]
    
    print(f"Features used: {feature_cols}")
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples (Out of sample): {len(X_test)}")
    
    models = {
        "Majority_Baseline": MajorityClassBaseline(),
        "Statistical_Baseline": BaselineModel(),
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=config['model']['seed']),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=config['model']['seed']),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=config['model']['seed'])
    }
    
    results = {}
    best_model = None
    best_loss = float('inf')
    best_model_name = ""
    
    for name, model in models.items():
        print(f"\n{'#'*50}\nTraining {name}...\n{'#'*50}")
        model.fit(X_train, y_train)
        
        print("\n--- TRAIN PERFORMANCE (In-Sample) ---")
        metrics_train = evaluate_model(y_train, model.predict_proba(X_train), model.predict(X_train), model_name=f"{name}_Train")
        
        print("\n--- VALIDATION PERFORMANCE (Chronological) ---")
        metrics_val = evaluate_model(y_val, model.predict_proba(X_val), model.predict(X_val), model_name=f"{name}_Validation")
        
        results[name] = metrics_val
        
        if metrics_val['log_loss'] < best_loss:
            best_loss = metrics_val['log_loss']
            best_model = model
            best_model_name = name
            
    print(f"\n\n{'*'*50}")
    print(f"Best model based on Validation Log Loss: {best_model_name}")
    print(f"{'*'*50}")
    
    print("\n--- FINAL TEST PERFORMANCE (Strictly Out-of-Sample) ---")
    metrics_test = evaluate_model(y_test, best_model.predict_proba(X_test), best_model.predict(X_test), model_name=f"{best_model_name}_Test")
    
    print("\n--- PERFORMANCE BY LEAGUE (Test Set) ---")
    for comp in test_df['League'].unique():
        comp_df = test_df[test_df['League'] == comp]
        if len(comp_df) > 5:
            X_comp = comp_df[feature_cols]
            y_comp = comp_df[target_col]
            y_prob_comp = best_model.predict_proba(X_comp)
            ll = log_loss(y_comp, y_prob_comp, labels=[0, 1, 2])
            acc = accuracy_score(y_comp, best_model.predict(X_comp))
            print(f"League: {comp} | Matches: {len(comp_df)} | Log Loss: {ll:.4f} | Accuracy: {acc:.4f}")
            
    print("\n--- TEST 5: PREDICTION ON HISTORICAL TEST MATCHES ---")
    sample_test = test_df.sample(n=min(5, len(test_df)), random_state=42)
    classes = ["Home Win", "Draw", "Away Win"]
    for idx, row in sample_test.iterrows():
        print(f"\nMatch: {row['HomeTeam']} vs {row['AwayTeam']} ({row['League']}) - Date: {row['Date'].date()}")
        X_sample = row[feature_cols].to_frame().T
        prob = best_model.predict_proba(X_sample)[0]
        pred_idx = best_model.predict(X_sample)[0]
        true_idx = int(row[target_col])
        
        print(f"  Probs  - 1: {prob[0]*100:.1f}%, X: {prob[1]*100:.1f}%, 2: {prob[2]*100:.1f}%")
        if 'B365H' in row:
            print(f"  Bookmaker Odds - 1: {row['B365H']}, X: {row['B365D']}, 2: {row['B365A']}")
        print(f"  Pred   - {classes[pred_idx]}")
        print(f"  Real   - {classes[true_idx]}")
        print(f"  Status - {'CORRECT' if pred_idx == true_idx else 'INCORRECT'}")
    
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    joblib.dump(best_model, os.path.join(config['paths']['models_dir'], "best_model_real.pkl"))
    
    # Save results
    with open(os.path.join(config['paths']['reports_dir'], "model_comparison_real.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nTraining pipeline complete.")
