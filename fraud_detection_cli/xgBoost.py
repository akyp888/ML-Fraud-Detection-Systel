#!/usr/bin/env python3
"""
xgboost.py
===========

XGBoost-specific training logic and hyperparameter configuration.
"""

from typing import Optional

import numpy as np

from preprocess import Config, logger

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None

# ------------------------------------------------------------------------------
# XGBoost hyperparameters
# ------------------------------------------------------------------------------

# You can tune these directly in this file.
XGB_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=-1,
)


# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
) -> Optional[XGBClassifier]:
    """
    Train an XGBoost classifier with a sane imbalance strategy.

    With our training sampling (train_target_fraud_rate ~ 2%), we usually do NOT
    need an additional scale_pos_weight.
    """
    if XGBClassifier is None:
        logger.warning("XGBoost not installed; skipping XGBoost model.")
        return None

    logger.info("=" * 70)
    logger.info("Training XGBoostClassifier with sane imbalance strategy")

    n_fraud = int((y_train == 1).sum())
    n_non_fraud = int((y_train == 0).sum())
    raw_ratio = float(n_non_fraud) / float(n_fraud + 1e-8) if n_fraud > 0 else 1.0
    logger.info("  Observed train class ratio (non-fraud / fraud): %.2f", raw_ratio)

    use_full = getattr(cfg, "use_full_train_for_xgb", False)
    if use_full:
        scale_pos_weight = raw_ratio
        logger.info(
            "  Using full training distribution for XGBoost; scale_pos_weight=%.2f",
            scale_pos_weight,
        )
    elif getattr(cfg, "train_target_fraud_rate", None) is not None:
        scale_pos_weight = 1.0
        logger.info(
            "  train_target_fraud_rate is set; disabling scale_pos_weight to avoid double compensation"
        )
    else:
        scale_pos_weight = raw_ratio
        logger.info("  No train_target_fraud_rate; using scale_pos_weight=%.2f", scale_pos_weight)

    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.random_state,
    )

    logger.info("Training XGBoost (no early stopping; version limitation)")
    xgb.fit(X_train, y_train)
    return xgb
