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
# 1. Télécharger les données réelles et les xG
py src/download_data.py
py src/download_xg.py

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
│   ├── download_xg.py        # Téléchargement historique xG (Understat)
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
| Logistic Regression (Sans xG) | 0.9861 | 52.9% |
| Logistic Regression (Avec xG) | 1.0091 | 52.0% |
| Baseline (fréquentiel) | 1.0711 | 44.1% |

**Note sur les Expected Goals (xG)** : Les données xG historiques proviennent de [Understat](https://understat.com). Les tests indépendants menés montrent que l'intégration naïve des xG (moyennes roulantes) **n'améliore pas** les performances par rapport au modèle basé uniquement sur les buts réels et les points. L'Edge reste négatif.

### Simulation de Paris (Value Betting 5% Edge - Modèle sans xG)

| Fold | Paris | Profit | ROI |
|------|:---:|:---:|:---:|
| 1 | 820 | -89.93 U | -10.97% |
| 2 | 713 | -36.00 U | -5.05% |
| 3 | 693 | -22.09 U | -3.19% |
| **Total** | **2226** | **-148.02 U** | **-6.65%** |
