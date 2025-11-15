#!/usr/bin/env python3
"""
Hybrid Resampling + Multi-Model Fraud Detection Pipeline
========================================================

Assumptions:
  - Input file: transactions.jsonl with schema based on T_RT_FRD_JUMP_TRX.
  - Label column: FRAUD_IND (string/number: "1" = fraud, else non-fraud).
  - Extreme imbalance (e.g., ~0.0037% fraud).

Pipeline:
  1. Load JSONL and create binary label column.
  2. Stratified train / validation / test split.
  3. Preprocessing:
       - Numeric: median imputation + StandardScaler.
       - Categorical: most_frequent imputation + OneHotEncoder.
  4. Hybrid resampling on TRAIN ONLY:
       - SMOTETomek (SMOTE + Tomek Links).
       - SMOTEENN  (SMOTE + Edited Nearest Neighbors).
       - Each with configurable oversampling factor.
  5. Train models on resampled data:
       - RandomForestClassifier
       - LGBMClassifier (LightGBM)
       - XGBClassifier  (XGBoost)
  6. Evaluate:
       - ROC-AUC, PR-AUC (Average Precision).
       - Threshold tuned by F_beta on validation.
       - Classification report on val & test.
       - Precision@top-K (top fraction of highest scores).

Note: This script focuses on the *tabular ML / resampling* side and assumes
the data schema is already correct as per your data dictionary. It only
relies on the presence of FRAUD_IND and treats all other columns as features.
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.over_sampling import SMOTE

import lightgbm as lgb
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(msg, file=sys.stdout, flush=True)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    transactions_path: Path
    label_col: str = "label"
    raw_label_col: str = "FRAUD_IND"
    random_state: int = 42

    # Splits
    test_size: float = 0.2
    val_size: float = 0.1  # relative to full dataset

    # Resampling
    smotetomek_factor: float = 10.0   # oversample factor for SMOTETomek
    smoteenn_factor: float = 10.0     # oversample factor for SMOTEENN

    # Evaluation
    f_beta: float = 1.0               # F1 by default; >1 favors recall
    top_k_frac: float = 0.01          # Precision@top 1% of scores
    verbose_reports: bool = True

    # Which strategies/models to run
    resamplers: Tuple[str, ...] = ("smotetomek", "smoteenn")
    models: Tuple[str, ...] = ("rf", "lgbm", "xgb")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_transactions(cfg: Config) -> pd.DataFrame:
    log(f"[INFO] Loading transactions from {cfg.transactions_path}")
    df = pd.read_json(cfg.transactions_path, lines=True)

    if cfg.raw_label_col not in df.columns:
        raise KeyError(
            f"Expected label column '{cfg.raw_label_col}' in transactions file."
        )

    # Normalize FRAUD_IND -> 0/1
    df[cfg.label_col] = (
        df[cfg.raw_label_col].astype(str).str.strip().eq("1").astype(int)
    )

    log(
        f"[INFO] Loaded {len(df)} rows; "
        f"fraud count={int(df[cfg.label_col].sum())}"
    )
    return df


def stratified_split(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split on the label column.
    """
    y = df[cfg.label_col].values

    # 1) Train+Val vs Test
    train_val_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )

    # 2) Train vs Val (relative)
    y_train_val = train_val_df[cfg.label_col].values
    val_rel = cfg.val_size / (1.0 - cfg.test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel,
        stratify=y_train_val,
        random_state=cfg.random_state,
    )

    log(
        "[INFO] Split label=1 counts: "
        f"train={int(train_df[cfg.label_col].sum())}/{len(train_df)}, "
        f"val={int(val_df[cfg.label_col].sum())}/{len(val_df)}, "
        f"test={int(test_df[cfg.label_col].sum())}/{len(test_df)}"
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Preprocessing (numeric + categorical)
# ---------------------------------------------------------------------------

def build_preprocessor(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[ColumnTransformer, List[str]]:
    """
    Detect numeric vs categorical columns, build ColumnTransformer:

      - Numeric: median imputation + StandardScaler
      - Categorical: most_frequent imputation + OneHotEncoder(handle_unknown='ignore')
    """
    exclude_cols = {cfg.label_col, cfg.raw_label_col}

    feature_cols = [c for c in df.columns if c not in exclude_cols]

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    log(f"[INFO] # numeric features    : {len(numeric_cols)}")
    log(f"[INFO] # categorical features: {len(categorical_cols)}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Configure OneHotEncoder with dense output and min_frequency if available
    ohe_kwargs: Dict[str, Any] = {"handle_unknown": "ignore"}
    from inspect import signature
    sig = signature(OneHotEncoder.__init__).parameters
    if "min_frequency" in sig:
        # avoid exploding dimension with ultra-rare categories
        ohe_kwargs["min_frequency"] = 10
    if "sparse_output" in sig:
        ohe_kwargs["sparse_output"] = False
    elif "sparse" in sig:
        ohe_kwargs["sparse"] = False

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(**ohe_kwargs)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, feature_cols


# ---------------------------------------------------------------------------
# Hybrid resampling builders
# ---------------------------------------------------------------------------

def build_smote_base(
    y_train: np.ndarray,
    oversample_factor: float,
    random_state: int,
) -> SMOTE:
    """
    Construct a SMOTE object with a custom sampling_strategy based on
    an oversampling factor:

      - Let n_min = minority count, n_maj = majority count.
      - Target minority count n_min_target = min(n_maj, n_min * oversample_factor).
      - sampling_strategy = n_min_target / n_maj

    This allows you to oversample *partially* instead of fully balancing.
    """
    counter = Counter(y_train)
    n_min = int(counter[1])
    n_maj = int(counter[0])

    if n_min == 0:
        raise ValueError("No minority (fraud) samples in y_train; cannot apply SMOTE.")

    n_min_target = int(min(n_maj, n_min * oversample_factor))
    sampling_strategy = n_min_target / n_maj

    log(f"[INFO] Original train class counts: {counter}")
    log(
        f"[INFO] SMOTE target minority count={n_min_target} "
        f"(factor={oversample_factor}), sampling_strategy={sampling_strategy:.6f}"
    )

    k_neighbors = min(5, max(1, n_min - 1))  # guard for tiny minority

    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
        n_jobs=-1,
    )
    return smote


def resample_with_smotetomek(
    X_train,
    y_train: np.ndarray,
    cfg: Config,
):
    """
    Apply SMOTETomek (SMOTE + Tomek Links) on train set.
    """
    log("[INFO] Applying SMOTETomek (SMOTE + Tomek Links) on train...")
    smote = build_smote_base(
        y_train,
        oversample_factor=cfg.smotetomek_factor,
        random_state=cfg.random_state,
    )

    smt = SMOTETomek(
        smote=smote,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    X_res, y_res = smt.fit_resample(X_train, y_train)
    log(
        f"[INFO] After SMOTETomek: class counts={Counter(y_res)}, "
        f"shape={X_res.shape}"
    )
    return X_res, y_res


def resample_with_smoteenn(
    X_train,
    y_train: np.ndarray,
    cfg: Config,
):
    """
    Apply SMOTEENN (SMOTE + Edited Nearest Neighbors) on train set.
    """
    log("[INFO] Applying SMOTEENN (SMOTE + ENN) on train...")
    smote = build_smote_base(
        y_train,
        oversample_factor=cfg.smoteenn_factor,
        random_state=cfg.random_state,
    )

    sme = SMOTEENN(
        smote=smote,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    X_res, y_res = sme.fit_resample(X_train, y_train)
    log(
        f"[INFO] After SMOTEENN: class counts={Counter(y_res)}, "
        f"shape={X_res.shape}"
    )
    return X_res, y_res


# ---------------------------------------------------------------------------
# Threshold tuning & evaluation utilities
# ---------------------------------------------------------------------------

def find_best_threshold_fbeta(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 1.0,
) -> Tuple[float, float]:
    """
    Find probability threshold that maximizes F_beta via the PR curve.
    Returns (best_threshold, best_f_beta).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)  # align lengths

    best_t = 0.5
    best_score = -1.0
    eps = 1e-9

    for p, r, t in zip(precision, recall, thresholds):
        if p + r == 0:
            continue
        f_beta = (1 + beta**2) * (p * r) / (beta**2 * p + r + eps)
        if f_beta > best_score:
            best_score = f_beta
            best_t = t

    return float(best_t), float(best_score)


def precision_recall_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    frac: float = 0.01,
) -> Tuple[int, float, float]:
    """
    Precision@K and Recall@K for top frac of highest scores.
    """
    n = len(y_true)
    k = max(1, int(n * frac))
    idx = np.argsort(y_proba)[::-1][:k]
    y_top = y_true[idx]
    prec_k = float(y_top.mean())
    rec_k = float(y_top.sum() / max(1, y_true.sum()))
    return k, prec_k, rec_k


def evaluate_model(
    name: str,
    model,
    X_val,
    y_val: np.ndarray,
    X_test,
    y_test: np.ndarray,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Evaluate model on validation and test sets:
      - ROC-AUC, PR-AUC
      - threshold tuned by F_beta on val
      - classification_report (val & test)
      - precision@top-K
    """
    log(f"[INFO] Evaluating model: {name}")

    # Probabilities
    if hasattr(model, "predict_proba"):
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        val_scores = model.decision_function(X_val)
        test_scores = model.decision_function(X_test)
        v_min, v_max = val_scores.min(), val_scores.max()
        t_min, t_max = test_scores.min(), test_scores.max()
        val_proba = (val_scores - v_min) / (v_max - v_min + 1e-9)
        test_proba = (test_scores - t_min) / (t_max - t_min + 1e-9)
    else:
        raise ValueError(f"Model {name} does not support predict_proba or decision_function")

    # ROC-AUC & PR-AUC
    val_auc = roc_auc_score(y_val, val_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    val_ap = average_precision_score(y_val, val_proba)
    test_ap = average_precision_score(y_test, test_proba)

    log(
        f"[{name}] Val ROC-AUC={val_auc:.4f}, PR-AUC={val_ap:.4f} | "
        f"Test ROC-AUC={test_auc:.4f}, PR-AUC={test_ap:.4f}"
    )

    # Threshold tuning (F_beta on validation PR curve)
    best_t, best_fbeta = find_best_threshold_fbeta(y_val, val_proba, beta=cfg.f_beta)
    log(
        f"[{name}] Best threshold (F_{cfg.f_beta:.1f} on val)={best_t:.4f}, "
        f"F_{cfg.f_beta:.1f}={best_fbeta:.4f}"
    )

    val_pred = (val_proba >= best_t).astype(int)
    test_pred = (test_proba >= best_t).astype(int)

    val_f1 = f1_score(y_val, val_pred)
    test_f1 = f1_score(y_test, test_pred)

    log(f"[{name}] Validation F1 at best_t={val_f1:.4f}, Test F1={test_f1:.4f}")

    if cfg.verbose_reports:
        log(f"[{name}] Classification report (Validation):")
        log(classification_report(y_val, val_pred, digits=4))
        log(f"[{name}] Classification report (Test):")
        log(classification_report(y_test, test_pred, digits=4))

    # Precision@top-K on test
    k, prec_k, rec_k = precision_recall_at_k(
        y_test,
        test_proba,
        frac=cfg.top_k_frac,
    )
    log(
        f"[{name}] Precision@top-{cfg.top_k_frac*100:.1f}% (K={k}): "
        f"precision={prec_k:.4f}, recall={rec_k:.4f}"
    )

    return {
        "val_auc": val_auc,
        "test_auc": test_auc,
        "val_ap": val_ap,
        "test_ap": test_ap,
        "best_threshold": best_t,
        "best_fbeta": best_fbeta,
        "val_f1": val_f1,
        "test_f1": test_f1,
        "precision_at_k": prec_k,
        "recall_at_k": rec_k,
        "k": k,
    }


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train,
    y_train: np.ndarray,
    cfg: Config,
):
    log("[INFO] Training RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=cfg.random_state,
        class_weight=None,  # imbalance handled via resampling
    )
    rf.fit(X_train, y_train)
    return rf


def train_lightgbm(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    cfg: Config,
):
    log("[INFO] Training LightGBM (LGBMClassifier) with early stopping...")

    # On resampled data, we typically don't need big scale_pos_weight.
    lgbm = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        min_child_samples=50,
        random_state=cfg.random_state,
        n_jobs=-1,
        is_unbalance=False,
        scale_pos_weight=1.0,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=callbacks,
    )
    return lgbm


def train_xgboost(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    cfg: Config,
):
    log("[INFO] Training XGBoostClassifier with early stopping...")

    xgb = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=-1,
        eval_metric="auc",
        random_state=cfg.random_state,
        scale_pos_weight=1.0,  # resampled, so keep this neutral
        use_label_encoder=False,
    )

    xgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True,
        early_stopping_rounds=50,
    )
    return xgb


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_for_resampler(
    resampler_name: str,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    X_test,
    y_test: np.ndarray,
    cfg: Config,
) -> Dict[str, Dict[str, Any]]:
    """
    Apply given resampler (smotetomek or smoteenn), then train & evaluate models.
    Returns a dict mapping model_name -> metrics.
    """
    if resampler_name == "smotetomek":
        X_train_res, y_train_res = resample_with_smotetomek(X_train, y_train, cfg)
    elif resampler_name == "smoteenn":
        X_train_res, y_train_res = resample_with_smoteenn(X_train, y_train, cfg)
    else:
        raise ValueError(f"Unknown resampler: {resampler_name}")

    metrics_by_model: Dict[str, Dict[str, Any]] = {}

    if "rf" in cfg.models:
        rf = train_random_forest(X_train_res, y_train_res, cfg)
        metrics_by_model["RandomForest"] = evaluate_model(
            f"{resampler_name.upper()} + RandomForest",
            rf,
            X_val,
            y_val,
            X_test,
            y_test,
            cfg,
        )

    if "lgbm" in cfg.models:
        lgbm = train_lightgbm(X_train_res, y_train_res, X_val, y_val, cfg)
        metrics_by_model["LightGBM"] = evaluate_model(
            f"{resampler_name.upper()} + LightGBM",
            lgbm,
            X_val,
            y_val,
            X_test,
            y_test,
            cfg,
        )

    if "xgb" in cfg.models:
        xgb = train_xgboost(X_train_res, y_train_res, X_val, y_val, cfg)
        metrics_by_model["XGBoost"] = evaluate_model(
            f"{resampler_name.upper()} + XGBoost",
            xgb,
            X_val,
            y_val,
            X_test,
            y_test,
            cfg,
        )

    return metrics_by_model


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid (SMOTETomek / SMOTEENN) resampling + multi-model training."
    )
    parser.add_argument(
        "--transactions-path",
        type=str,
        default="transactions.jsonl",
        help="Path to transactions.jsonl (with FRAUD_IND).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test fraction (default=0.2).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation fraction relative to full dataset (default=0.1).",
    )
    parser.add_argument(
        "--smotetomek-factor",
        type=float,
        default=10.0,
        help="Oversample factor for SMOTETomek (default=10.0).",
    )
    parser.add_argument(
        "--smoteenn-factor",
        type=float,
        default=10.0,
        help="Oversample factor for SMOTEENN (default=10.0).",
    )
    parser.add_argument(
        "--f-beta",
        type=float,
        default=1.0,
        help="F_beta used for threshold tuning (default=1.0).",
    )
    parser.add_argument(
        "--top-k-frac",
        type=float,
        default=0.01,
        help="Top fraction for Precision@K (default=0.01 = top 1%%).",
    )
    parser.add_argument(
        "--resamplers",
        type=str,
        nargs="+",
        default=["smotetomek", "smoteenn"],
        help="Which resamplers to run: smotetomek, smoteenn.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["rf", "lgbm", "xgb"],
        help="Which models to run: rf, lgbm, xgb.",
    )

    args = parser.parse_args()

    cfg = Config(
        transactions_path=Path(args.transactions_path),
        test_size=args.test_size,
        val_size=args.val_size,
        smotetomek_factor=args.smotetomek_factor,
        smoteenn_factor=args.smoteenn_factor,
        f_beta=args.f_beta,
        top_k_frac=args.top_k_frac,
        resamplers=tuple(args.resamplers),
        models=tuple(args.models),
    )

    seed_everything(cfg.random_state)

    log("[INFO] Config:")
    log(json.dumps(asdict(cfg), indent=2, default=str))

    # Load & split
    df = load_transactions(cfg)
    train_df, val_df, test_df = stratified_split(df, cfg)

    # Preprocessing
    preprocessor, feature_cols = build_preprocessor(train_df, cfg)

    X_train_raw = train_df[feature_cols]
    X_val_raw = val_df[feature_cols]
    X_test_raw = test_df[feature_cols]

    y_train = train_df[cfg.label_col].values
    y_val = val_df[cfg.label_col].values
    y_test = test_df[cfg.label_col].values

    log("[INFO] Fitting preprocessing pipeline on train...")
    preprocessor.fit(X_train_raw)

    log("[INFO] Transforming train/val/test...")
    X_train = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    log(
        f"[INFO] Feature shapes: "
        f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    # Run for each resampler
    all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for res_name in cfg.resamplers:
        res_name_lower = res_name.lower()
        if res_name_lower not in {"smotetomek", "smoteenn"}:
            log(f"[WARN] Skipping unknown resampler: {res_name}")
            continue

        log("=" * 80)
        log(f"[INFO] >>> Resampler: {res_name_lower.upper()} <<<")
        res_metrics = run_for_resampler(
            res_name_lower,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            cfg,
        )
        all_results[res_name_lower] = res_metrics

    log("=" * 80)
    log("[INFO] SUMMARY (val/test metrics):")
    for res_name, model_metrics in all_results.items():
        for model_name, metrics in model_metrics.items():
            log(
                f"{res_name.upper()} + {model_name}: "
                f"Val AUC={metrics['val_auc']:.4f}, "
                f"Test AUC={metrics['test_auc']:.4f}, "
                f"Val F1={metrics['val_f1']:.4f}, "
                f"Test F1={metrics['test_f1']:.4f}, "
                f"P@K={metrics['precision_at_k']:.4f}, "
                f"R@K={metrics['recall_at_k']:.4f}"
            )


if __name__ == "__main__":
    main()
