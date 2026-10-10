import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, classification_report, balanced_accuracy_score, confusion_matrix
import json
import os

def evaluate_model(y_true, y_prob, y_pred, model_name="Model"):
    """
    Evaluates a model using Log Loss, Brier Score (approximated for multi-class or one-vs-all),
    Accuracy, Balanced Accuracy, and Confusion Matrix.
    """
    metrics = {}
    
    # Log loss
    try:
        metrics['log_loss'] = log_loss(y_true, y_prob)
    except Exception as e:
        metrics['log_loss'] = np.nan
        
    # Brier Score (average across classes)
    brier_scores = []
    for i in range(y_prob.shape[1]):
        y_true_binary = (y_true == i).astype(int)
        brier_scores.append(brier_score_loss(y_true_binary, y_prob[:, i]))
    metrics['brier_score'] = np.mean(brier_scores)
    
    # Accuracy & Balanced Accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    
    print(f"\n{'='*40}")
    print(f"--- Evaluation for {model_name} ---")
    print(f"{'='*40}")
    print(f"Log Loss:          {metrics['log_loss']:.4f}")
    print(f"Brier Score:       {metrics['brier_score']:.4f} (overall average)")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    
    print("\nCalibration / Brier Score per class:")
    class_names = ["Home", "Draw", "Away"]
    for i, b_score in enumerate(brier_scores):
        print(f" - {class_names[i]}: {b_score:.4f}")
        
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    print("\nConfusion Matrix (Rows: True, Cols: Predicted - [Home, Draw, Away]):")
    print(confusion_matrix(y_true, y_pred))
    
    return metrics
