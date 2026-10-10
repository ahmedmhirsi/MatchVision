"""
predict.py – Match prediction using the Walk-Forward trained pipeline.

Uses the same rolling features as feature_engineering.py:
  - 5-match rolling averages (goals scored/conceded, points) per team
  - Bookmaker implied probabilities (optional)

⚠️  This model is NOT profitable for betting.  See README for details.
"""

import pandas as pd
import numpy as np
import yaml
import sys
import os

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _team_rolling_stats(df, team, n_matches=5):
    """Compute rolling stats for *team* using its last *n_matches* in *df*.

    ``df`` must be sorted by Date ascending and contain only matches that
    occurred **before** the target match (i.e. history only).
    """
    mask = (df["HomeTeam"] == team) | (df["AwayTeam"] == team)
    past = df.loc[mask].tail(n_matches)

    if past.empty:
        return {"goals_scored_avg": 0, "goals_conceded_avg": 0, "points_avg": 0}

    gs, gc, pts = 0, 0, 0
    for _, row in past.iterrows():
        if row["HomeTeam"] == team:
            gs += row["FTHG"]
            gc += row["FTAG"]
            if row["target"] == 0:
                pts += 3
            elif row["target"] == 1:
                pts += 1
        else:
            gs += row["FTAG"]
            gc += row["FTHG"]
            if row["target"] == 2:
                pts += 3
            elif row["target"] == 1:
                pts += 1

    n = len(past)
    return {
        "goals_scored_avg": gs / n,
        "goals_conceded_avg": gc / n,
        "points_avg": pts / n,
    }


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

def predict_match(home_team: str, away_team: str, config: dict,
                  odds_h: float = None, odds_d: float = None, odds_a: float = None):
    """Predict the outcome of a match.

    Parameters
    ----------
    home_team, away_team : str
        Exact team names as they appear in the dataset (e.g. 'Arsenal', 'Paris SG').
    config : dict
        Loaded config.yaml.
    odds_h, odds_d, odds_a : float, optional
        Bet365-style decimal odds. If all three are provided they are used as
        implied-probability features; otherwise they are set to neutral 1/3.
    """
    from sklearn.linear_model import LogisticRegression  # lazy import – only for fallback

    # ── Load historical data ────────────────────────────────────────────
    features_path = config["paths"]["processed_matches"]
    if not os.path.exists(features_path):
        print(f"[ERROR] Feature file not found: {features_path}")
        print("        Run the pipeline first:  py src/download_data.py && py src/data_cleaning.py && py src/feature_engineering.py && py src/train.py")
        return

    df = pd.read_csv(features_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # We also need the raw cleaned data for rolling stats
    cleaned_path = "data/processed/cleaned_matches.csv"
    if not os.path.exists(cleaned_path):
        print(f"[ERROR] Cleaned data not found: {cleaned_path}")
        return

    raw = pd.read_csv(cleaned_path)
    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.sort_values("Date").reset_index(drop=True)

    # ── Check teams exist ───────────────────────────────────────────────
    all_teams = set(raw["HomeTeam"].unique()) | set(raw["AwayTeam"].unique())
    for name, label in [(home_team, "Home"), (away_team, "Away")]:
        if name not in all_teams:
            print(f"[ERROR] {label} team '{name}' not found in dataset.")
            close = [t for t in sorted(all_teams) if name.lower() in t.lower()]
            if close:
                print(f"        Did you mean: {close[:5]}")
            return

    # ── Build feature vector ────────────────────────────────────────────
    home_stats = _team_rolling_stats(raw, home_team, n_matches=5)
    away_stats = _team_rolling_stats(raw, away_team, n_matches=5)

    # Implied probabilities
    if odds_h and odds_d and odds_a:
        raw_probs = [1 / odds_h, 1 / odds_d, 1 / odds_a]
        margin = sum(raw_probs)
        prob_h = raw_probs[0] / margin
        prob_d = raw_probs[1] / margin
        prob_a = raw_probs[2] / margin
    else:
        prob_h, prob_d, prob_a = 1 / 3, 1 / 3, 1 / 3

    feature_names = [
        "Home_Gls_Scored_avg", "Home_Gls_Conceded_avg", "Home_Pts_avg",
        "Away_Gls_Scored_avg", "Away_Gls_Conceded_avg", "Away_Pts_avg",
        "Form_Advantage", "Prob_H_implied", "Prob_D_implied", "Prob_A_implied",
    ]

    X = pd.DataFrame([{
        "Home_Gls_Scored_avg":    home_stats["goals_scored_avg"],
        "Home_Gls_Conceded_avg":  home_stats["goals_conceded_avg"],
        "Home_Pts_avg":           home_stats["points_avg"],
        "Away_Gls_Scored_avg":    away_stats["goals_scored_avg"],
        "Away_Gls_Conceded_avg":  away_stats["goals_conceded_avg"],
        "Away_Pts_avg":           away_stats["points_avg"],
        "Form_Advantage":         home_stats["points_avg"] - away_stats["points_avg"],
        "Prob_H_implied":         prob_h,
        "Prob_D_implied":         prob_d,
        "Prob_A_implied":         prob_a,
    }])

    # ── Try loading saved model, else retrain a simple one on the fly ──
    model = None
    model_path = os.path.join(config["paths"]["models_dir"], "best_calibrated_model.pkl")
    if os.path.exists(model_path):
        import joblib
        try:
            model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"[WARN] Could not load model ({e}), retraining on the fly.")

    if model is None:
        # Quick retrain (Logistic Regression) on all available data
        print("Retraining a Logistic Regression on full history (may take a moment)...")
        from sklearn.calibration import CalibratedClassifierCV

        # Prepare training data from features file
        train_df = df.copy()
        if "B365H" in train_df.columns:
            train_df["Prob_H_implied"] = 1 / train_df["B365H"]
            train_df["Prob_D_implied"] = 1 / train_df["B365D"]
            train_df["Prob_A_implied"] = 1 / train_df["B365A"]
            m = train_df["Prob_H_implied"] + train_df["Prob_D_implied"] + train_df["Prob_A_implied"]
            train_df["Prob_H_implied"] /= m
            train_df["Prob_D_implied"] /= m
            train_df["Prob_A_implied"] /= m

        # Keep only rows with history
        train_df = train_df[train_df["Home_Pts_avg"] > 0].copy()

        available = [c for c in feature_names if c in train_df.columns]
        X_train = train_df[available]
        y_train = train_df["target"].values

        base = LogisticRegression(max_iter=1000, random_state=42)
        model = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
        model.fit(X_train, y_train)
        print("Model retrained.")

    # ── Predict ─────────────────────────────────────────────────────────
    probs = model.predict_proba(X)[0]

    labels = ["Home Win", "Draw", "Away Win"]
    pred_idx = int(np.argmax(probs))
    max_prob = probs[pred_idx]

    confidence = "Low"
    if max_prob > 0.60:
        confidence = "High"
    elif max_prob > 0.45:
        confidence = "Medium"

    # ── Display ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  PREDICTION:  {home_team}  vs  {away_team}")
    print(f"{'='*50}")
    print(f"  Home Win (1) : {probs[0]*100:5.1f}%")
    print(f"  Draw     (X) : {probs[1]*100:5.1f}%")
    print(f"  Away Win (2) : {probs[2]*100:5.1f}%")
    print(f"  ──────────────────────────────")
    print(f"  Prediction   : {labels[pred_idx]}")
    print(f"  Confidence   : {confidence}")

    if odds_h and odds_d and odds_a:
        print(f"\n  Odds supplied : H={odds_h}  D={odds_d}  A={odds_a}")
        edge = probs[pred_idx] - [prob_h, prob_d, prob_a][pred_idx]
        print(f"  Edge vs market: {edge*100:+.1f}%")

    print(f"\n  ⚠️  DISCLAIMER: This model has a negative ROI (-5.98%)")
    print(f"     in walk-forward backtesting.  Do NOT use for real betting.")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:  py src/predict.py 'HomeTeam' 'AwayTeam' [odds_H odds_D odds_A]")
        print("Example: py src/predict.py 'Arsenal' 'Chelsea' 1.90 3.40 4.20")
        sys.exit(1)

    config = load_config()

    odds = {}
    if len(sys.argv) >= 6:
        odds = {"odds_h": float(sys.argv[3]), "odds_d": float(sys.argv[4]), "odds_a": float(sys.argv[5])}

    predict_match(sys.argv[1], sys.argv[2], config, **odds)
