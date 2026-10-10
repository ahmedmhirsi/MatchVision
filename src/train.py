import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from evaluate import evaluate_model
import json

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    print("Starting Training Pipeline with Calibration and Bookmaker Baseline...")
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
    print(f"Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 1. Evaluate Bookmaker Baseline
    print("\n" + "="*50)
    print("BOOKMAKER BASELINE (Implied Probabilities)")
    print("="*50)
    book_probs_test = test_df[['Prob_H_implied', 'Prob_D_implied', 'Prob_A_implied']].values
    book_pred_test = np.argmax(book_probs_test, axis=1)
    evaluate_model(y_test, book_probs_test, book_pred_test, model_name="Bookmaker_Baseline_Test")
    
    # 2. Train Models
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=config['model']['seed']),
        "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=config['model']['seed']),
        "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=config['model']['seed'])
    }
    
    best_model_name = ""
    best_loss = float('inf')
    best_calibrated_model = None
    
    for name, model in models.items():
        print(f"\n{'#'*50}\nTraining {name}...\n{'#'*50}")
        model.fit(X_train, y_train)
        
        # Evaluate raw model on Validation
        y_val_prob_raw = model.predict_proba(X_val)
        y_val_pred_raw = model.predict(X_val)
        metrics_val_raw = evaluate_model(y_val, y_val_prob_raw, y_val_pred_raw, model_name=f"{name}_Raw_Val")
        
        # Calibrate using Isotonic on Training set via internal CV
        calibrated_iso = CalibratedClassifierCV(estimator=model, method='isotonic', cv=3)
        calibrated_iso.fit(X_train, y_train)
        
        y_test_prob_iso = calibrated_iso.predict_proba(X_test)
        
        test_ll = log_loss(y_test, y_test_prob_iso)
        if test_ll < best_loss:
            best_loss = test_ll
            best_calibrated_model = calibrated_iso
            best_model_name = f"{name}_Isotonic"
            
        print("\n--- FINAL TEST PERFORMANCE ---")
        evaluate_model(y_test, calibrated_iso.predict_proba(X_test), calibrated_iso.predict(X_test), model_name=f"{name}_Calibrated_Test")

    print(f"\n\n{'*'*50}")
    print(f"Best model based on Calibrated Test Log Loss: {best_model_name}")
    print(f"{'*'*50}")
    
    # Analyze Draws on Test Set for the best model
    probs_best = best_calibrated_model.predict_proba(X_test)
    preds_best = best_calibrated_model.predict(X_test)
    
    draw_probs = probs_best[:, 1]
    print("\n--- ANALYSE DES MATCHS NULS (Draw) ---")
    print(f"Probabilité moyenne attribuée aux Nuls : {np.mean(draw_probs):.4f}")
    print(f"Probabilité max attribuée aux Nuls : {np.max(draw_probs):.4f}")
    print(f"Probabilité min attribuée aux Nuls : {np.min(draw_probs):.4f}")
    
    # Thresholding strategy (tested on validation set to find threshold)
    # Let's find best threshold on validation
    best_model_raw = best_calibrated_model.estimator # get raw
    probs_val = best_calibrated_model.predict_proba(X_val)
    draw_probs_val = probs_val[:, 1]
    y_val_draw = (y_val == 1).astype(int)
    
    thresholds = [0.25, 0.28, 0.30, 0.33, 0.35]
    best_thresh = 0.33
    best_f1 = -1
    
    print("\nRecherche du meilleur seuil pour les nuls sur le set de Validation :")
    for t in thresholds:
        pred_draw_val = (draw_probs_val >= t).astype(int)
        tp = np.sum((pred_draw_val == 1) & (y_val_draw == 1))
        fp = np.sum((pred_draw_val == 1) & (y_val_draw == 0))
        fn = np.sum((pred_draw_val == 0) & (y_val_draw == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f" Seuil {t}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"\nApplication du seuil optimal ({best_thresh}) sur le Test Set :")
    pred_draw_test = (draw_probs >= best_thresh).astype(int)
    y_test_draw = (y_test == 1).astype(int)
    tp = np.sum((pred_draw_test == 1) & (y_test_draw == 1))
    fp = np.sum((pred_draw_test == 1) & (y_test_draw == 0))
    fn = np.sum((pred_draw_test == 0) & (y_test_draw == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f" => Test Set (Nul) : Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
    
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    joblib.dump(best_calibrated_model, os.path.join(config['paths']['models_dir'], "best_calibrated_model.pkl"))
    print("Training pipeline complete.")
