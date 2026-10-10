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
