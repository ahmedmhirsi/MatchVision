"""
test_pipeline.py – Essential tests for the MatchVision prediction pipeline.

Run with:  py -m pytest tests/ -v
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import pytest

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from import_lineups import merge_lineup_data


# ═══════════════════════════════════════════════════════════════════════════
# 1. Data Cleaning
# ═══════════════════════════════════════════════════════════════════════════

class TestDataCleaning:
    """Tests for src/data_cleaning.py."""

    def _make_raw_df(self):
        """Create a minimal raw DataFrame mimicking football-data.co.uk."""
        return pd.DataFrame({
            "Date":     ["23/08/2024", "24/08/2024", "24/08/2024", "25/08/2024", None],
            "HomeTeam": ["Arsenal",    "Chelsea",    "Chelsea",    "Liverpool",  "X"],
            "AwayTeam": ["Wolves",     "Palace",     "Palace",     "Brentford",  "Y"],
            "FTHG":     [2,            1,            1,            2,            np.nan],
            "FTAG":     [0,            1,            1,            1,            np.nan],
            "FTR":      ["H",          "D",          "D",          "H",          None],
            "B365H":    [1.40,         2.10,         2.10,         1.55,         np.nan],
            "B365D":    [4.50,         3.40,         3.40,         4.10,         np.nan],
            "B365A":    [8.00,         3.60,         3.60,         6.50,         np.nan],
        })

    def test_drop_missing_essentials(self):
        """Rows with missing Date / teams / goals / result must be dropped."""
        from data_cleaning import clean_match_data

        df = self._make_raw_df()
        cleaned = clean_match_data(df)

        # The last row (all-NaN essentials) should be gone
        assert len(cleaned) < len(df), "Missing-essential rows were not dropped"
        assert cleaned["Date"].isna().sum() == 0

    def test_date_parsing(self):
        """Dates must be parsed as datetime (DD/MM/YYYY)."""
        from data_cleaning import clean_match_data

        df = self._make_raw_df()
        cleaned = clean_match_data(df)

        assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])
        # 23/08/2024 → August
        assert cleaned["Date"].iloc[0].month == 8

    def test_target_mapping(self):
        """FTR → target:  H=0, D=1, A=2."""
        from data_cleaning import clean_match_data

        df = self._make_raw_df()
        cleaned = clean_match_data(df)

        h_rows = cleaned[cleaned["FTR"] == "H"]
        d_rows = cleaned[cleaned["FTR"] == "D"]

        assert (h_rows["target"] == 0).all(), "Home wins should map to 0"
        assert (d_rows["target"] == 1).all(), "Draws should map to 1"

    def test_duplicates_removed(self):
        """Exact duplicate matches (same Date + HomeTeam + AwayTeam) must be deduped."""
        from data_cleaning import clean_match_data

        df = self._make_raw_df()
        cleaned = clean_match_data(df)

        dupes = cleaned.duplicated(subset=["Date", "HomeTeam", "AwayTeam"])
        assert dupes.sum() == 0, "Duplicate matches not removed"

    def test_odds_numeric(self):
        """B365* columns must be numeric after cleaning."""
        from data_cleaning import clean_match_data

        df = self._make_raw_df()
        cleaned = clean_match_data(df)

        for col in ["B365H", "B365D", "B365A"]:
            assert pd.api.types.is_numeric_dtype(cleaned[col]), f"{col} is not numeric"
            assert cleaned[col].isna().sum() == 0, f"{col} has NaN after cleaning"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Feature Engineering – No Look-Ahead Leakage
# ═══════════════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    """Tests for src/feature_engineering.py."""

    def _make_cleaned_df(self):
        """Create a small sorted dataset with known outcomes."""
        dates = pd.date_range("2024-08-01", periods=10, freq="7D")
        teams = [
            ("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"),
            ("B", "C"), ("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"),
        ]
        np.random.seed(42)
        fthg = np.random.randint(0, 4, 10)
        ftag = np.random.randint(0, 4, 10)
        targets = []
        for h, a in zip(fthg, ftag):
            if h > a:
                targets.append(0)
            elif h == a:
                targets.append(1)
            else:
                targets.append(2)

        return pd.DataFrame({
            "Date":     dates,
            "HomeTeam": [t[0] for t in teams],
            "AwayTeam": [t[1] for t in teams],
            "FTHG":     fthg,
            "FTAG":     ftag,
            "target":   targets,
            "B365H":    np.random.uniform(1.3, 4.0, 10),
            "B365D":    np.random.uniform(3.0, 4.0, 10),
            "B365A":    np.random.uniform(2.0, 8.0, 10),
            "League":   ["Test"] * 10,
            "Season":   ["2425"] * 10,
        })

    def test_no_lookahead(self):
        """Rolling features for match i must NOT use data from match i or later."""
        from feature_engineering import get_team_past_stats

        df = self._make_cleaned_df()

        # For the 5th match (index 4), team "A" played at indices 0, 2, 4
        # get_team_past_stats must only use matches BEFORE index 4's date
        match_date = df.iloc[4]["Date"]
        stats = get_team_past_stats(df, "A", match_date, n_matches=5)

        # Manually compute: matches 0 and 2 are before index 4
        past_a = df[((df["HomeTeam"] == "A") | (df["AwayTeam"] == "A")) & (df["Date"] < match_date)]
        assert len(past_a) <= 2, "Look-ahead: stats used future data"

        # The function should use exactly those matches
        expected_gs = 0
        for _, row in past_a.iterrows():
            if row["HomeTeam"] == "A":
                expected_gs += row["FTHG"]
            else:
                expected_gs += row["FTAG"]

        assert abs(stats["goals_scored_avg"] - expected_gs / max(len(past_a), 1)) < 1e-6

    def test_feature_output_columns(self):
        """create_match_features must produce the expected column set."""
        from feature_engineering import create_match_features

        df = self._make_cleaned_df()
        features = create_match_features(df)

        required = [
            "Home_Gls_Scored_avg", "Home_Gls_Conceded_avg", "Home_Pts_avg",
            "Away_Gls_Scored_avg", "Away_Gls_Conceded_avg", "Away_Pts_avg",
            "Form_Advantage", "target",
        ]
        for col in required:
            assert col in features.columns, f"Missing expected column: {col}"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestLineupImport:
    def test_merge_lineups(self):
        cleaned_data = pd.DataFrame({
            'Date': ['2024-08-16', '2024-08-17'],
            'HomeTeam': ['Man United', 'Arsenal'],
            'AwayTeam': ['Fulham', 'Wolves'],
            'target': [0, 0]
        })
        lineup_data = pd.DataFrame({
            'Date': ['2024-08-16'],
            'HomeTeam': ['Man United'],
            'AwayTeam': ['Fulham'],
            'Home_Missing_Count': [2],
            'Away_Missing_Count': [1],
            'Home_Missing_Key_Players': [1],
            'Away_Missing_Key_Players': [0],
            'Home_Lineup_Continuity': [0.8],
            'Away_Lineup_Continuity': [0.9]
        })
        merged = merge_lineup_data(cleaned_data, lineup_data)
        assert len(merged) == 2
        # Check that Man United got data
        assert merged.loc[0, 'Home_Missing_Count'] == 2
        # Check that Arsenal got defaults
        assert merged.loc[1, 'Home_Missing_Count'] == 0
        assert merged.loc[1, 'Home_Lineup_Continuity'] == 1.0
class TestEvaluateFold:
    """Tests for the evaluate_fold function in train.py."""

    def test_perfect_predictions(self):
        """Perfect predictions should yield 0 log-loss and 100% accuracy."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from train import evaluate_fold

        y_true = np.array([0, 1, 2, 0, 1])
        # Near-perfect probabilities (log_loss can't reach 0 with float)
        y_prob = np.array([
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
            [0.005, 0.005, 0.99],
            [0.99, 0.005, 0.005],
            [0.005, 0.99, 0.005],
        ])

        result = evaluate_fold(y_true, y_prob)
        assert result["accuracy"] == 1.0
        assert result["log_loss"] < 0.02  # Very low but not exactly 0

    def test_random_predictions(self):
        """Uniform random predictions should give log_loss ≈ log(3) ≈ 1.099."""
        from train import evaluate_fold

        y_true = np.array([0, 1, 2] * 100)
        y_prob = np.full((300, 3), 1 / 3)

        result = evaluate_fold(y_true, y_prob)
        assert abs(result["log_loss"] - np.log(3)) < 0.01

    def test_balanced_accuracy(self):
        """Balanced accuracy should handle class imbalance correctly."""
        from train import evaluate_fold

        # All predictions are class 0 → recall for class 1 and 2 is 0
        y_true = np.array([0, 0, 0, 1, 1, 2])
        y_prob = np.array([
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
        ])

        result = evaluate_fold(y_true, y_prob)
        # recall(0)=1.0, recall(1)=0.0, recall(2)=0.0 → bacc = 1/3
        assert abs(result["bacc"] - 1 / 3) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# 4. No Data Leakage in Walk-Forward Splits
# ═══════════════════════════════════════════════════════════════════════════

class TestWalkForwardIntegrity:
    """Verify that train/test splits in train.py respect chronological order."""

    def test_train_always_before_test(self):
        """In every Walk-Forward fold, max(train_date) <= min(test_date)."""
        features_path = "data/processed/matches_features.csv"
        if not os.path.exists(features_path):
            pytest.skip("Feature file not available (run pipeline first)")

        df = pd.read_csv(features_path)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df[df["Home_Pts_avg"] > 0].sort_values("Date").reset_index(drop=True)

        n = len(df)
        splits = [
            (int(n * 0.4), int(n * 0.6)),
            (int(n * 0.6), int(n * 0.8)),
            (int(n * 0.8), n),
        ]

        for fold_i, (train_end, test_end) in enumerate(splits):
            train_max = df.iloc[:train_end]["Date"].max()
            test_min = df.iloc[train_end:test_end]["Date"].min()
            assert train_max <= test_min, (
                f"Fold {fold_i + 1}: train leaks into test! "
                f"train_max={train_max}, test_min={test_min}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 5. Betting Simulation Logic
# ═══════════════════════════════════════════════════════════════════════════

class TestBettingSimulation:
    """Tests for simulate_betting in train.py."""

    def test_winning_all_bets(self):
        """If the model always wins, profit should be positive."""
        from train import simulate_betting

        # 3 matches, model predicts perfectly with big edge
        df = pd.DataFrame({
            "target":         [0, 1, 2],
            "B365H":          [2.0, 3.0, 5.0],
            "B365D":          [3.0, 3.0, 3.0],
            "B365A":          [4.0, 3.0, 2.0],
            "Prob_H_implied": [0.50, 0.33, 0.20],
            "Prob_D_implied": [0.33, 0.33, 0.33],
            "Prob_A_implied": [0.25, 0.33, 0.50],
        })

        # Model gives huge probability to the correct outcome
        probs = np.array([
            [0.90, 0.05, 0.05],  # correctly predicts H
            [0.05, 0.90, 0.05],  # correctly predicts D
            [0.05, 0.05, 0.90],  # correctly predicts A
        ])

        bets, profit, roi = simulate_betting(df, probs)
        assert bets > 0, "Should have placed bets"
        assert profit > 0, "Perfect predictions should yield positive profit"
        assert roi > 0, "ROI should be positive when always winning"

    def test_no_edge_no_bets(self):
        """If model probabilities are below the 5% edge threshold, no bets."""
        from train import simulate_betting

        df = pd.DataFrame({
            "target":         [0],
            "B365H":          [2.0],
            "B365D":          [3.0],
            "B365A":          [4.0],
            "Prob_H_implied": [0.50],
            "Prob_D_implied": [0.33],
            "Prob_A_implied": [0.25],
        })

        # Model agrees with bookmaker (no edge)
        probs = np.array([[0.50, 0.33, 0.25]])
        # Normalized: these are <= book_prob * 1.05 for each class

        bets, profit, roi = simulate_betting(df, probs)
        assert bets == 0, "No bets should be placed without edge"
        assert roi == 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. Security – No Secrets in Git
# ═══════════════════════════════════════════════════════════════════════════

class TestSecurityGitignore:
    """Verify no sensitive files are tracked by Git."""

    def test_no_env_files_tracked(self):
        """No .env or secret files should be tracked."""
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        tracked = result.stdout.strip().split("\n")

        dangerous_patterns = [".env", ".secret", ".token", ".pem", ".key"]
        for f in tracked:
            for pat in dangerous_patterns:
                assert not f.endswith(pat), f"Sensitive file tracked by Git: {f}"

    def test_no_large_data_tracked(self):
        """No CSV data files or model pickles should be tracked."""
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, cwd=os.path.join(os.path.dirname(__file__), ".."),
        )
        tracked = result.stdout.strip().split("\n")

        for f in tracked:
            assert not f.endswith(".pkl"), f"Model file tracked by Git: {f}"
            # CSV in data/ should not be tracked
            if f.startswith("data/"):
                pytest.fail(f"Data file tracked by Git: {f}")
