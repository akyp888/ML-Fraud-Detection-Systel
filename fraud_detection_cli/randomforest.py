#!/usr/bin/env python3
"""
randomforest.py
================

RandomForest-specific training logic and hyperparameter configuration.
"""

import time
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from preprocess import Config, logger

# ------------------------------------------------------------------------------
# RandomForest hyperparameters
# ------------------------------------------------------------------------------

# You can tune these directly in this file.
RF_N_ESTIMATORS: int = 400
RF_WARM_START_CHUNK: int = 400
RF_N_JOBS: int = -1
RF_MAX_DEPTH: int = 8
RF_MIN_SAMPLES_SPLIT: int = 10
RF_MIN_SAMPLES_LEAF: int = 5
RF_MAX_FEATURES: str = "sqrt"
RF_MAX_POS_WEIGHT: float = 1000.0


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
) -> RandomForestClassifier:
    """
    Train a regularized RandomForestClassifier with class weighting and warm-starting.
    """
    logger.info("=" * 70)
    logger.info("Training RandomForestClassifier (regularized + class weighting)")

    total_trees = max(1, int(RF_N_ESTIMATORS))
    chunk = max(1, min(int(RF_WARM_START_CHUNK), total_trees))
    warm = total_trees > chunk

    n_fraud = int((y_train == 1).sum())
    n_non_fraud = int((y_train == 0).sum())
    raw_ratio = n_non_fraud / (n_fraud + 1e-8) if n_fraud > 0 else 1.0
    fraud_weight = min(raw_ratio, RF_MAX_POS_WEIGHT)
    class_weight = {0: 1.0, 1: fraud_weight}
    logger.info(
        "  Class weights (capped): fraud=%.2f, non-fraud=1.0 | raw_ratio=%.2f",
        fraud_weight,
        raw_ratio,
    )

    rf = RandomForestClassifier(
        n_estimators=chunk if warm else total_trees,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        n_jobs=RF_N_JOBS,
        class_weight=class_weight,
        random_state=cfg.random_state,
        warm_start=warm,
    )

    trees_built = 0
    overall_start = time.time()
    while trees_built < total_trees:
        target_trees = min(total_trees, trees_built + chunk)
        rf.set_params(n_estimators=target_trees)
        start = time.time()
        rf.fit(X_train, y_train)
        trees_built = target_trees
        duration = time.time() - start

        if hasattr(rf, "predict_proba"):
            from sklearn.metrics import roc_auc_score

            val_proba = rf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            logger.info(
                "  -> RandomForest progress: %d/%d trees (val AUC=%.4f, %.1fs)",
                trees_built,
                total_trees,
                val_auc,
                duration,
            )
        else:
            logger.info(
                "  -> RandomForest progress: %d/%d trees (%.1fs)",
                trees_built,
                total_trees,
                duration,
            )

    logger.info("RandomForest training complete in %.1fs", time.time() - overall_start)
    return rf
