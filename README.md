# MatchVision

Système de prédiction de matchs de football basé sur du Machine Learning.
Le pipeline télécharge des historiques de matchs réels (via football-data.co.uk), extrait des features temporelles (formes des 5 derniers matchs), et entraîne un modèle pour prédire le résultat (1X2).

## Prérequis

- **Python 3.10+** (testé avec 3.14)
- **pip** (inclus avec Python)
- **Windows** : utiliser `py` au lieu de `python` si l'alias n'est pas configuré
- **Git** (pour cloner et contribuer)

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation (Pipeline Complet)

```bash
# 1. Télécharger les données réelles
py src/download_data.py

# 2. Nettoyer les données
py src/data_cleaning.py

# 3. Générer les features temporelles
py src/feature_engineering.py

# 4. Entraîner et backtester (Walk-Forward)
py src/train.py

# 5. Prédire un match (optionnel : ajouter les cotes B365)
py src/predict.py "Arsenal" "Chelsea"
py src/predict.py "Arsenal" "Chelsea" 1.90 3.40 4.20
```

## Tests

```bash
py -m pytest tests/ -v
```

## Architecture du Projet

```
MatchVision/
├── config/
│   └── config.yaml          # Chemins et paramètres
├── data/                     # (gitignored) Données brutes et traitées
├── models/                   # (gitignored) Modèles entraînés (.pkl)
├── reports/                  # Rapports JSON + logs
├── src/
│   ├── download_data.py      # Téléchargement football-data.co.uk
│   ├── data_cleaning.py      # Nettoyage et parsing des dates
│   ├── feature_engineering.py # Features temporelles (rolling 5 matchs)
│   ├── train.py              # Walk-Forward backtesting + simulation paris
│   ├── evaluate.py           # Métriques (Log Loss, Brier, Accuracy)
│   └── predict.py            # Prédiction d'un match
├── tests/
│   └── test_pipeline.py      # Tests automatisés
├── requirements.txt
└── README.md
```

## Performances et Limites Statistiques

⚠️ **Avertissement de Rentabilité** : 
Le système a été rigoureusement audité via un backtesting *Walk-Forward* sur plus de 2000 matchs réels. 
Les résultats ont démontré que :
1. **Les Bookmakers restent supérieurs** : L'algorithme (Régression Logistique calibrée) atteint une précision globale de ~52.8%, mais son *Log Loss* est légèrement en deçà de la performance des probabilités implicites du marché (Bookmakers).
2. **ROI Négatif** : Une simulation stricte de Value Betting (miser sur un avantage modèle > 5% face au marché) débouche sur un **ROI de -5.98%**, ce qui équivaut mathématiquement à payer la marge bénéficiaire du bookmaker sans posséder d'avantage statistique (Edge). 
3. **Usage** : Ce projet constitue une excellente base d'apprentissage en Data Science et Machine Learning (ingénierie des caractéristiques temporelles, calibration Isotonic, backtesting glissant), mais il **n'est pas rentable en l'état pour les paris sportifs**. Il nécessite l'intégration de métriques avancées (xG, effectifs) pour espérer battre le marché.

### Résultats Walk-Forward (3 folds, 3368 matchs)

| Modèle | Mean Log Loss | Mean Accuracy |
|--------|:---:|:---:|
| Bookmaker (baseline) | **0.9739** | **53.9%** |
| Random Forest | 0.9815 | 52.4% |
| Logistic Regression | 0.9861 | 52.9% |
| Baseline (fréquentiel) | 1.0711 | 44.1% |

### Simulation de Paris (Value Betting 5% Edge)

| Fold | Paris | Profit | ROI |
|------|:---:|:---:|:---:|
| 1 | 804 | -92.78 U | -11.54% |
| 2 | 725 | -31.28 U | -4.31% |
| 3 | 639 | -5.64 U | -0.88% |
| **Total** | **2168** | **-129.70 U** | **-5.98%** |
