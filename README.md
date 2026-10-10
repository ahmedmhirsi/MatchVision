# MatchVision

Système de prédiction de matchs de football basé sur du Machine Learning.
Le pipeline télécharge des historiques de matchs réels (via football-data.co.uk), extrait des features temporelles (formes des 5 derniers matchs), et entraîne un modèle pour prédire le résultat (1X2).

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation (Pipeline Complet)

```bash
python src/download_data.py
python src/data_cleaning.py
python src/feature_engineering.py
python src/train.py
```

## Performances et Limites Statistiques

⚠️ **Avertissement de Rentabilité** : 
Le système a été rigoureusement audité via un backtesting *Walk-Forward* sur plus de 2000 matchs réels. 
Les résultats ont démontré que :
1. **Les Bookmakers restent supérieurs** : L'algorithme (Régression Logistique calibrée) atteint une précision globale de ~52.8%, mais son *Log Loss* est légèrement en deçà de la performance des probabilités implicites du marché (Bookmakers).
2. **ROI Négatif** : Une simulation stricte de Value Betting (miser sur un avantage modèle > 5% face au marché) débouche sur un **ROI de -5.98%**, ce qui équivaut mathématiquement à payer la marge bénéficiaire du bookmaker sans posséder d'avantage statistique (Edge). 
3. **Usage** : Ce projet constitue une excellente base d'apprentissage en Data Science et Machine Learning (ingénierie des caractéristiques temporelles, calibration Isotonic, backtesting glissant), mais il **n'est pas rentable en l'état pour les paris sportifs**. Il nécessite l'intégration de métriques avancées (xG, effectifs) pour espérer battre le marché.
