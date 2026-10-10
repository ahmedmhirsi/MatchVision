# MatchVision — FootPredict AI

Système de prédiction de résultats de matchs de football (1X2) basé sur le Machine Learning.

## Objectif

MatchVision télécharge des historiques de matchs réels, extrait des features temporelles (formes des 5 derniers matchs, Expected Goals), et entraîne un modèle calibré (Régression Logistique + Isotonic) pour prédire la probabilité de chaque issue : Victoire Domicile (1), Nul (X), Victoire Extérieur (2).

**Le projet est conçu comme un outil d'apprentissage en Data Science / ML appliqué au sport.
Il n'est PAS rentable pour les paris sportifs.**

---

## Sources de données

| Source | Type | URL | Licence / Conditions |
|--------|------|-----|----------------------|
| football-data.co.uk | Résultats + cotes Bet365 | <https://www.football-data.co.uk/> | Gratuit, usage personnel et recherche. Pas de redistribution commerciale. |
| Understat (via `understatapi`) | Expected Goals (xG) | <https://understat.com/> | Données publiques ; l'API wrapper `understatapi` est un projet communautaire. Ne pas scraper abusivement. |
| *(Optionnel)* CSV utilisateur | Effectifs / blessures | Fourni par l'utilisateur | Cf. `src/import_lineups.py` pour le format. |

> **Note** : Les données brutes (`data/`) et les modèles entraînés (`models/`) sont exclus du dépôt via `.gitignore`. Vous devez les re-télécharger / re-entraîner.

---

## Architecture du projet

```
MatchVision/
├── config/
│   └── config.yaml              # Chemins et paramètres centralisés
├── data/                         # (gitignored) Données brutes et traitées
│   ├── raw/                      #   CSV téléchargés (football-data, Understat)
│   └── processed/                #   Données nettoyées + features
├── models/                       # (gitignored) Modèles entraînés (.pkl)
├── reports/
│   ├── model_comparison.json     # Métriques train/test (80/20)
│   └── model_comparison_real.json# Métriques walk-forward réelles
├── src/
│   ├── download_data.py          # Téléchargement football-data.co.uk
│   ├── download_xg.py            # Téléchargement historique xG (Understat)
│   ├── data_cleaning.py          # Nettoyage, parsing dates, merge xG
│   ├── feature_engineering.py    # Features temporelles (rolling 5 matchs)
│   ├── import_lineups.py         # (Optionnel) Merge effectifs/blessures
│   ├── train.py                  # Walk-Forward backtesting + simulation paris
│   ├── evaluate.py               # Métriques (Log Loss, Brier, Accuracy)
│   └── predict.py                # Prédiction d'un match (CLI)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # 16 tests automatisés
├── requirements.txt              # Dépendances Python
└── README.md
```

---

## Installation (Windows)

### Prérequis

- **Python 3.10+** (testé avec 3.14) — [python.org](https://www.python.org/downloads/)
- **pip** (inclus avec Python)
- **Git** (pour cloner le dépôt) — [git-scm.com](https://git-scm.com/)

> Sur Windows, utiliser `py` au lieu de `python` si l'alias n'est pas configuré.

### Étapes

```powershell
# 1. Cloner le dépôt
git clone https://github.com/ahmedmhirsi/MatchVision.git
cd MatchVision

# 2. (Recommandé) Créer un environnement virtuel
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Commandes du pipeline

Chaque étape est un script indépendant, exécutable en séquence :

```powershell
# ─── Télécharger ───
py src/download_data.py            # Résultats + cotes depuis football-data.co.uk
py src/download_xg.py             # (Optionnel) xG depuis Understat

# ─── Nettoyer ───
py src/data_cleaning.py            # Parse les dates, merge xG, crée target
py src/import_lineups.py           # (Optionnel) Merge effectifs/blessures

# ─── Feature Engineering ───
py src/feature_engineering.py      # Rolling 5 matchs → features temporelles

# ─── Entraîner & Évaluer (Walk-Forward) ───
py src/train.py                    # 3 folds expanding-window + simulation paris

# ─── Prédire ───
py src/predict.py "Arsenal" "Chelsea"
py src/predict.py "Arsenal" "Chelsea" 1.90 3.40 4.20   # avec cotes B365
```

### Pipeline complet en une ligne

```powershell
py src/download_data.py && py src/data_cleaning.py && py src/feature_engineering.py && py src/train.py
```

---

## Variables utilisées (Features)

### Features temporelles (rolling 5 derniers matchs)

| Variable | Description |
|----------|-------------|
| `Home_Gls_Scored_avg` | Moyenne de buts marqués (domicile) |
| `Home_Gls_Conceded_avg` | Moyenne de buts encaissés (domicile) |
| `Home_Pts_avg` | Moyenne de points (domicile) |
| `Away_Gls_Scored_avg` | Moyenne de buts marqués (extérieur) |
| `Away_Gls_Conceded_avg` | Moyenne de buts encaissés (extérieur) |
| `Away_Pts_avg` | Moyenne de points (extérieur) |
| `Form_Advantage` | Différence de forme (Home_Pts_avg − Away_Pts_avg) |

### Features xG (optionnelles)

| Variable | Description |
|----------|-------------|
| `Home_xG_avg` / `Away_xG_avg` | Moyenne d'Expected Goals |
| `Home_xGA_avg` / `Away_xGA_avg` | Moyenne d'Expected Goals Against |
| `Home_xG_diff_avg` / `Away_xG_diff_avg` | Différence xG − xGA |
| `xG_Advantage` | Avantage xG global (home − away) |

### Features marché (Bet365)

| Variable | Description |
|----------|-------------|
| `Prob_H_implied` | Probabilité implicite normalisée (1/B365H) |
| `Prob_D_implied` | Probabilité implicite normalisée (1/B365D) |
| `Prob_A_implied` | Probabilité implicite normalisée (1/B365A) |

### Features effectifs (optionnelles, fournies par l'utilisateur)

| Variable | Description |
|----------|-------------|
| `Home_Missing_Count` / `Away_Missing_Count` | Joueurs absents |
| `Home_Missing_Key_Players` / `Away_Missing_Key_Players` | Titulaires absents |
| `Home_Lineup_Continuity` / `Away_Lineup_Continuity` | % du XI précédent conservé |

### Variable cible

- `target` : 0 = Victoire Domicile, 1 = Nul, 2 = Victoire Extérieur

---

## Protocole Walk-Forward

Le backtesting suit un protocole **expanding-window Walk-Forward** pour simuler des conditions réelles de prédiction :

```
Données triées chronologiquement
├── Fold 1 : Train = [0%, 40%]  → Test = [40%, 60%]
├── Fold 2 : Train = [0%, 60%]  → Test = [60%, 80%]
└── Fold 3 : Train = [0%, 80%]  → Test = [80%, 100%]
```

**Principes clés** :
1. **Aucune fuite temporelle** : les features rolling n'utilisent que des matchs AVANT la date cible.
2. **Expanding window** : la taille d'entraînement croît à chaque fold.
3. **Calibration Isotonic** interne (CV=3) sur le train uniquement.
4. **Hyperparameter tuning** via GridSearchCV (C ∈ {0.1, 1, 10}) sur le train uniquement.
5. **Test vérifié** : un test automatisé (`test_train_always_before_test`) valide `max(train_date) ≤ min(test_date)`.

---

## Métriques réelles (Walk-Forward, 3 folds, ~3368 matchs)

### Comparaison des modèles

| Modèle | Mean Log Loss | Mean Brier | Mean Accuracy | Balanced Acc |
|--------|:---:|:---:|:---:|:---:|
| **Bookmaker (baseline)** | **0.9739** | **0.1931** | **53.9%** | **46.3%** |
| Logistic Regression (sans xG) | 0.9845 | 0.1952 | 53.1% | 46.1% |
| Random Forest (sans xG) | 1.0018 | 0.1951 | 52.9% | 46.3% |
| Logistic Regression (avec xG) | 1.0124 | 0.1976 | 51.8% | 45.3% |
| Baseline (fréquentiel) | 1.0711 | 0.2160 | 44.1% | 33.3% |

### Simulation de paris (Value Betting, seuil 5% d'Edge)

| Fold | Paris placés | Profit | ROI |
|------|:---:|:---:|:---:|
| 1 | 818 | −113.02 U | −13.82% |
| 2 | 703 | −52.86 U | −7.52% |
| 3 | 644 | −4.69 U | −0.73% |
| **Total** | **2165** | **−170.57 U** | **−7.88%** |

---

## Limites du modèle et résultats négatifs

> **⚠️ Ce modèle n'est PAS rentable pour les paris sportifs.**

### Résultats négatifs constatés

1. **Les bookmakers sont supérieurs** : Le Log Loss du modèle (0.9845) est systématiquement inférieur à celui du marché (0.9739). Le modèle ne possède aucun *edge* statistique.
2. **ROI négatif** : −7.88% sur 2165 paris simulés. Cela revient à payer la marge du bookmaker sans avantage.
3. **L'ajout naïf des xG dégrade les performances** : Les moyennes roulantes de xG n'apportent pas de signal supplémentaire par rapport aux buts réels + points. Log Loss passe de 0.9845 à 1.0124.
4. **Prédiction des nuls faible** : Le Balanced Accuracy (~46%) montre que les nuls restent difficiles à prédire, même avec un seuil optimisé.

### Limites structurelles

- **Features limitées** : Pas de données d'effectifs, de forme individuelle des joueurs, de contexte tactique, ou de données météo.
- **Cross-ligue** : Le modèle s'entraîne sur 5 championnats simultanément, ce qui peut diluer les patterns ligue-spécifiques.
- **Pas de gestion de la saisonnalité** : Début et fin de saison, trêve hivernale, etc.
- **Modèle linéaire** : La Régression Logistique a une capacité limitée à capturer des interactions non-linéaires complexes.

---

## Exemples de prédictions

> **⚠️ Les exemples ci-dessous sont des illustrations du fonctionnement du modèle.
> Ils NE constituent PAS des conseils de paris ni des garanties de performance.**

```
$ py src/predict.py "Arsenal" "Chelsea"

==================================================
  PREDICTION:  Arsenal  vs  Chelsea
==================================================
  Home Win (1) :  52.3%
  Draw     (X) :  25.1%
  Away Win (2) :  22.6%
  ──────────────────────────────
  Prediction   : Home Win
  Confidence   : Medium

  ⚠️  DISCLAIMER: This model has a negative ROI (-5.98%)
     in walk-forward backtesting.  Do NOT use for real betting.
==================================================
```

```
$ py src/predict.py "Arsenal" "Chelsea" 1.90 3.40 4.20

==================================================
  PREDICTION:  Arsenal  vs  Chelsea
==================================================
  Home Win (1) :  54.1%
  Draw     (X) :  24.8%
  Away Win (2) :  21.1%
  ──────────────────────────────
  Prediction   : Home Win
  Confidence   : Medium

  Odds supplied : H=1.9  D=3.4  A=4.2
  Edge vs market: +1.2%

  ⚠️  DISCLAIMER: This model has a negative ROI (-5.98%)
     in walk-forward backtesting.  Do NOT use for real betting.
==================================================
```

> **Rappel** : Les probabilités affichées proviennent d'un modèle dont le Log Loss est 0.9845 vs 0.9739 pour le marché. Le modèle est en moyenne *moins précis* que les cotes Bet365.

---

## Tests

```powershell
py -m pytest tests/ -v
```

Les 16 tests couvrent :
- **Data Cleaning** (5 tests) : suppression NaN, parsing dates, mapping target, déduplication, odds numériques
- **Feature Engineering** (2 tests) : absence de look-ahead leakage, colonnes attendues
- **Lineup Import** (1 test) : merge correct avec valeurs par défaut
- **Évaluation** (3 tests) : prédictions parfaites, aléatoires, balanced accuracy
- **Walk-Forward** (1 test) : `max(train_date) ≤ min(test_date)` sur données réelles
- **Simulation de Paris** (2 tests) : scénario gagnant, aucun pari sans edge
- **Sécurité** (2 tests) : aucun secret ni données dans git

---

## Contribuer

1. Fork le dépôt
2. Créer une branche feature (`git checkout -b feat/ma-feature`)
3. Commiter (`git commit -m "feat: description"`)
4. Pousser (`git push origin feat/ma-feature`)
5. Ouvrir une Pull Request

---

## Licence

Ce projet est un outil de recherche et d'apprentissage. Les données proviennent de sources tierces avec leurs propres conditions d'utilisation (cf. [Sources de données](#sources-de-données)).
