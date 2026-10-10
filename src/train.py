import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss, confusion_matrix, precision_score, recall_score, f1_score
from evaluate import evaluate_model
import json

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class BaselineModel:
    def __init__(self):
        self.probs = None
        self.classes_ = np.array([0, 1, 2])
        
    def fit(self, X, y):
        counts = pd.Series(y).value_counts(normalize=True).sort_index()
        self.probs = np.array([counts.get(0, 0.33), counts.get(1, 0.33), counts.get(2, 0.33)])
        
    def predict_proba(self, X):
        return np.tile(self.probs, (X.shape[0], 1))
        
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def evaluate_fold(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    ll = log_loss(y_true, y_prob, labels=[0, 1, 2])
    acc = accuracy_score(y_true, y_pred)
    bs = np.mean([brier_score_loss((y_true == c).astype(int), y_prob[:, c]) for c in range(3)])
    
    # Balanced Accuracy manually for 3 classes
    recalls = []
    for c in range(3):
        mask = (y_true == c)
        if np.sum(mask) > 0:
            recalls.append(np.sum((y_pred == c) & mask) / np.sum(mask))
    bacc = np.mean(recalls) if recalls else 0
    
    return {'log_loss': ll, 'accuracy': acc, 'brier': bs, 'bacc': bacc}

def find_best_draw_threshold(y_true, draw_probs):
    thresholds = np.linspace(0.20, 0.40, 21)
    best_t = 0.33
    best_f1 = -1
    y_draw = (y_true == 1).astype(int)
    
    for t in thresholds:
        pred_draw = (draw_probs >= t).astype(int)
        precision = precision_score(y_draw, pred_draw, zero_division=0)
        recall = recall_score(y_draw, pred_draw, zero_division=0)
        f1 = f1_score(y_draw, pred_draw, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t

def simulate_betting(df_test, probs_model):
    """
    Simulate a simple Value Betting strategy.
    Bet 1 unit if Model Prob > Bookmaker Implied Prob * 1.05 (5% value threshold).
    Returns ROI, number of bets, net profit.
    """
    net_profit = 0
    bets = 0
    
    for idx, (i, row) in enumerate(df_test.iterrows()):
        # Real outcomes
        true_res = int(row['target'])
        
        # Bookmaker odds
        odds = [row['B365H'], row['B365D'], row['B365A']]
        book_probs = [row['Prob_H_implied'], row['Prob_D_implied'], row['Prob_A_implied']]
        mod_probs = probs_model[idx]
        
        for c in range(3):
            # If model sees > 5% edge over bookmaker implied probability
            if mod_probs[c] > book_probs[c] * 1.05:
                bets += 1
                if true_res == c:
                    net_profit += (odds[c] - 1) # Win: gain odds-1
                else:
                    net_profit -= 1 # Lose 1 unit
                    
    roi = (net_profit / bets * 100) if bets > 0 else 0
    return bets, net_profit, roi

if __name__ == "__main__":
    print("Starting Walk-Forward Backtesting Pipeline...")
    config = load_config()
    
    df = pd.read_csv(config['paths']['processed_matches'])
    df['Date'] = pd.to_datetime(df['Date'])
    
    if 'B365H' in df.columns:
        df['Prob_H_implied'] = 1 / df['B365H']
        df['Prob_D_implied'] = 1 / df['B365D']
        df['Prob_A_implied'] = 1 / df['B365A']
        margin = df['Prob_H_implied'] + df['Prob_D_implied'] + df['Prob_A_implied']
        df['Prob_H_implied'] /= margin
        df['Prob_D_implied'] /= margin
        df['Prob_A_implied'] /= margin

    df = df[df['Home_Pts_avg'] > 0].copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    feature_cols_no_xg = [
        'Home_Gls_Scored_avg', 'Home_Gls_Conceded_avg', 'Home_Pts_avg',
        'Away_Gls_Scored_avg', 'Away_Gls_Conceded_avg', 'Away_Pts_avg',
        'Form_Advantage', 'Prob_H_implied', 'Prob_D_implied', 'Prob_A_implied'
    ]
    feature_cols_xg = feature_cols_no_xg + [
        'Home_xG_avg', 'Home_xGA_avg', 'Home_xG_diff_avg',
        'Away_xG_avg', 'Away_xGA_avg', 'Away_xG_diff_avg',
        'xG_Advantage'
    ]
    
    lineup_cols = [
        'Home_Missing_Count', 'Away_Missing_Count',
        'Home_Missing_Key_Players', 'Away_Missing_Key_Players',
        'Home_Lineup_Continuity', 'Away_Lineup_Continuity'
    ]
    has_lineups = all(c in df.columns for c in lineup_cols)
    if has_lineups:
        feature_cols_lineups = feature_cols_xg + lineup_cols
        
    target_col = 'target'
    
    # Define Walk-Forward Splits (Quarterly)
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    print(f"Dataset span: {start_date.date()} to {end_date.date()} ({len(df)} matches)")
    
    # Let's create folds by time. E.g. every 6 months.
    # Actually, let's use fixed train size and sliding window, or expanding window.
    # Expanding window: Fold 1: train on first 40%, test on next 20%
    # Fold 2: train on first 60%, test on next 20%
    # Fold 3: train on first 80%, test on last 20%
    n_samples = len(df)
    splits = [
        (int(n_samples * 0.4), int(n_samples * 0.6)),
        (int(n_samples * 0.6), int(n_samples * 0.8)),
        (int(n_samples * 0.8), n_samples)
    ]
    
    log_grid = {'C': [0.1, 1.0, 10.0]}
    rf_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}
    
    models = {
        "Baseline": BaselineModel(),
        "Logistic_Regression_No_xG": GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_grid, cv=3),
        "Logistic_Regression_With_xG": GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_grid, cv=3),
        "Random_Forest": GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=3)
    }
    if has_lineups:
        models["Logistic_Regression_With_Lineups"] = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_grid, cv=3)
    
    fold_results = {name: [] for name in models.keys()}
    fold_results["Bookmaker"] = []
    
    betting_results = []
    draw_results = []
    
    fold_idx = 1
    for train_end, test_end in splits:
        print(f"\n{'='*50}\nEvaluating Fold {fold_idx}\n{'='*50}")
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        
        X_train_no_xg, y_train = train_df[feature_cols_no_xg], train_df[target_col].values
        X_test_no_xg, y_test = test_df[feature_cols_no_xg], test_df[target_col].values
        
        X_train_xg = train_df[feature_cols_xg]
        X_test_xg = test_df[feature_cols_xg]
        
        if has_lineups:
            X_train_lineups = train_df[feature_cols_lineups]
            X_test_lineups = test_df[feature_cols_lineups]
        
        print(f"Train window: {train_df['Date'].min().date()} to {train_df['Date'].max().date()} ({len(train_df)} matches)")
        print(f"Test window:  {test_df['Date'].min().date()} to {test_df['Date'].max().date()} ({len(test_df)} matches)")
        
        # 1. Bookmaker Baseline
        book_probs_test = test_df[['Prob_H_implied', 'Prob_D_implied', 'Prob_A_implied']].values
        res_book = evaluate_fold(y_test, book_probs_test)
        fold_results["Bookmaker"].append(res_book)
        
        best_fold_model = None
        best_fold_ll = float('inf')
        
        for name, base_model in models.items():
            if name == "Baseline":
                model = base_model
                model.fit(X_train_no_xg, y_train)
                y_prob = model.predict_proba(X_test_no_xg)
            elif name == "Logistic_Regression_With_xG":
                model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
                model.fit(X_train_xg, y_train)
                y_prob = model.predict_proba(X_test_xg)
            elif name == "Logistic_Regression_With_Lineups":
                model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
                model.fit(X_train_lineups, y_train)
                y_prob = model.predict_proba(X_test_lineups)
            else:
                # Calibrate strictly on training set internally
                model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
                model.fit(X_train_no_xg, y_train)
                y_prob = model.predict_proba(X_test_no_xg)
                
            res = evaluate_fold(y_test, y_prob)
            fold_results[name].append(res)
            print(f"{name:30s} - LL: {res['log_loss']:.4f} | Acc: {res['accuracy']:.4f} | Brier: {res['brier']:.4f} | BAcc: {res['bacc']:.4f}")
            
            # Sub-analysis by League (only for No_xG as representative)
            if name == "Logistic_Regression_No_xG":
                for league in test_df['League'].unique():
                    league_mask = test_df['League'] == league
                    if league_mask.sum() > 10:
                        ll_league = log_loss(y_test[league_mask], y_prob[league_mask], labels=[0, 1, 2])
                        print(f"    -> {league:15s}: LL {ll_league:.4f} ({league_mask.sum()} matchs)")
                        
            if name == "Logistic_Regression_No_xG":
                # Tune threshold for draws using historical training data predictions
                train_probs = model.predict_proba(X_train_no_xg)
                best_t = find_best_draw_threshold(y_train, train_probs[:, 1])
                
                # Apply on Test
                draw_pred_test = (y_prob[:, 1] >= best_t).astype(int)
                y_test_draw = (y_test == 1).astype(int)
                tp = np.sum((draw_pred_test == 1) & (y_test_draw == 1))
                fp = np.sum((draw_pred_test == 1) & (y_test_draw == 0))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / np.sum(y_test_draw) if np.sum(y_test_draw) > 0 else 0
                draw_results.append({'threshold': best_t, 'precision': precision, 'recall': recall})
                
                # Betting Simulation
                bets, profit, roi = simulate_betting(test_df, y_prob)
                betting_results.append({'bets': bets, 'profit': profit, 'roi': roi})
                
        fold_idx += 1
        
    print(f"\n\n{'*'*50}")
    print("WALK-FORWARD BACKTESTING RESULTS (Aggregated over 3 folds)")
    print(f"{'*'*50}")
    
    for name in fold_results.keys():
        lls = [r['log_loss'] for r in fold_results[name]]
        accs = [r['accuracy'] for r in fold_results[name]]
        briers = [r['brier'] for r in fold_results[name]]
        baccs = [r['bacc'] for r in fold_results[name]]
        print(f"{name:30s} - LL: {np.mean(lls):.4f} (±{np.std(lls):.4f}) | Acc: {np.mean(accs):.4f} | Brier: {np.mean(briers):.4f} | BAcc: {np.mean(baccs):.4f}")
        
    print("\n--- ANALYSE DES NULS (Régression Logistique) ---")
    for i, d in enumerate(draw_results):
        print(f"Fold {i+1}: Seuil Appris = {d['threshold']:.2f} -> Test Precision: {d['precision']:.2f}, Recall: {d['recall']:.2f}")
        
    print("\n--- SIMULATION DE PARIS (Value Betting 5% Edge - LogReg) ---")
    total_bets = sum([b['bets'] for b in betting_results])
    total_profit = sum([b['profit'] for b in betting_results])
    overall_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    for i, b in enumerate(betting_results):
        print(f"Fold {i+1}: Paris: {b['bets']}, Profit: {b['profit']:.2f} U, ROI: {b['roi']:.2f}%")
    print(f"=> TOTAL: Paris: {total_bets}, Profit Net: {total_profit:.2f} Unités, ROI Global: {overall_roi:.2f}%")
    
    print("\nWalk-Forward Pipeline complete.")
