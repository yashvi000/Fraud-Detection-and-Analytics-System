import pandas as pd
import numpy as np
import logging

from sklearn.metrics import (
    average_precision_score, roc_auc_score, roc_curve, 
    f1_score, recall_score, precision_score
)

logger = logging.getLogger(__name__)

def evaluate_model (model, x, y, threshold=0.5, split_name="Val"):

    prob = model.predict_proba(x)[:, 1]
    y_pred = (prob >= threshold).astype(int)

    pr_auc = average_precision_score(y, prob)
    roc_auc = roc_auc_score(y, prob)
    
    f1 = f1_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    precision = precision_score(y, y_pred, zero_division=0)

    tn = ((y_pred == 0) & (y == 0)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.00

    # recall @ fpr(fixed)
    fpr_arr, tpr_arr, _ = roc_curve(y, prob)
    
    def recall_at_fpr(target_fpr):
        mask = fpr_arr <= target_fpr
        return float(tpr_arr[mask].max()) if mask.sum() > 0 else 0.00

    # precision top k
    sorted_idx = np.argsort(prob)[::-1]
    y_sorted = np.array(y)[sorted_idx]
    
    def precision_at_k(k):
        return float(y_sorted[:k].mean()) if k <= len(y_sorted) else 0.00

    metrics = {
        "pr_auc" : round(pr_auc, 5),
        "roc_auc" : round(roc_auc, 5),
        "f1" : round(f1, 5),
        "recall" : round(recall, 5),
        "precision" : round(precision, 5),
        "fpr" : round(fpr, 5),
        "recall_at_0.1_fpr" : round(recall_at_fpr(0.001), 5),
        "recall_at_1_fpr" : round(recall_at_fpr(0.01), 5),
        "precision_top_1k" : round(precision_at_k(1000), 5),
        "precision_top_5k" : round(precision_at_k(5000), 5),
        "precision_top_10k" : round(precision_at_k(10000), 5),
    }

    logger.info(f"{split_name} Evaluation (threshold = {threshold}):")
    for metric, value in metrics.items():
        logger.info(f"{metric:<18} : {value:.5f}")
    
    return prob, metrics
