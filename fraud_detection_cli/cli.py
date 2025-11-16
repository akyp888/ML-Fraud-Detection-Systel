#!/usr/bin/env python3
"""
cli.py
=======

Command-line entrypoint that wires together preprocessing, model training,
and evaluation.

Usage (from project root):

    python cli.py run fraud_pipeline

All configuration is edited directly in the Python modules (preprocess.py,
randomforest.py, xgboost.py); the CLI stays intentionally simple.
"""

import sys
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
)

from preprocess import (
    Config,
    apply_env_overrides,
    seed_everything,
    load_sample_jsonl,
    clean_and_normalize_raw,
    merge_trx_and_ecm,
    split_train_val_test,
    engineer_features,
    make_training_sample,
    build_feature_matrix,
    resample_training_data,
    logger,
)
from randomforest import train_random_forest
from xgboost import train_xgboost

# Optional SHAP for explainability
try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ------------------------------------------------------------------------------
# Threshold tuning helpers
# ------------------------------------------------------------------------------

def precision_recall_at_k(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    frac: float = 0.01,
) -> Tuple[int, float, float]:
    """
    Compute precision@K and recall@K where K is a fraction of the dataset (e.g., top 1%).
    """
    n = len(y_true)
    k = max(1, int(n * frac))
    idx = np.argsort(y_proba)[::-1][:k]
    y_top = y_true[idx]
    prec_k = float(y_top.mean())
    recall_k = float(y_top.sum() / max(1, y_true.sum()))
    return k, prec_k, recall_k


def find_best_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float = 1.0,
    metric: str = "f_beta",
) -> Tuple[float, float, Dict[str, float]]:
    """
    Find the probability threshold that optimizes a specified metric.

    Supports optimization by F-beta, PR-AUC, or balanced accuracy.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = np.append(thresholds, 1.0)  # align lengths

    best_t = 0.5
    best_score = -1.0
    best_metrics: Dict[str, float] = {}
    eps = 1e-9

    for p, r, t in zip(precision, recall, thresholds):
        if p + r == 0:
            continue

        if metric == "f_beta":
            score = (1 + beta**2) * (p * r) / (beta**2 * p + r + eps)
        elif metric == "pr_auc":
            score = average_precision_score(y_true, y_proba)
        elif metric == "balanced_accuracy":
            pred = (y_proba >= t).astype(int)
            tn = ((y_true == 0) & (pred == 0)).sum()
            fn = ((y_true == 1) & (pred == 0)).sum()
            tp = ((y_true == 1) & (pred == 1)).sum()
            fp = ((y_true == 0) & (pred == 1)).sum()
            sensitivity = tp / (tp + fn + eps) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp + eps) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        else:
            score = (1 + beta**2) * (p * r) / (beta**2 * p + r + eps)

        if score > best_score:
            best_score = float(score)
            best_t = float(t)
            best_metrics = {
                "precision": float(p),
                "recall": float(r),
                "f_beta": float((1 + beta**2) * (p * r) / (beta**2 * p + r + eps)),
            }

    return best_t, best_score, best_metrics


def find_cost_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> Tuple[float, float, Dict[str, float]]:
    """
    Choose threshold that minimizes expected cost:
        cost = fn_cost * P(fraud) * FNR + fp_cost * P(non-fraud) * FPR
    """
    eps = 1e-9
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    frac_pos = float(y_true.mean())
    frac_neg = 1.0 - frac_pos

    fnr = 1.0 - tpr
    costs = fn_cost * frac_pos * fnr + fp_cost * frac_neg * fpr

    best_idx = int(np.argmin(costs))
    best_t = float(thresholds[best_idx])
    best_cost = float(costs[best_idx])

    best_metrics = {
        "fpr": float(fpr[best_idx]),
        "tpr": float(tpr[best_idx]),
        "fnr": float(fnr[best_idx]),
        "cost": best_cost,
        "prevalence": frac_pos,
    }

    return best_t, best_cost, best_metrics


def find_threshold_for_target_alert_rate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_alert_rate: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Choose threshold so that approximately target_alert_rate of cases are flagged.
    """
    eps = 1e-9
    n = len(y_proba)
    k = max(1, int(n * target_alert_rate))
    sorted_scores = np.sort(y_proba)[::-1]
    t = float(sorted_scores[min(k - 1, n - 1)])

    pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    precision = tp / (tp + fp + eps) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + eps) if (tp + fn) > 0 else 0.0

    f_beta = (
        (1 + 1.0**2) * precision * recall / (1.0**2 * precision + recall + eps)
        if (precision + recall) > 0
        else 0.0
    )
    alert_rate = float(pred.mean())

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f_beta": float(f_beta),
        "alert_rate": alert_rate,
        "k": int(k),
        "threshold": t,
    }
    return t, metrics


# ------------------------------------------------------------------------------
# Calibration, feature importance, SHAP
# ------------------------------------------------------------------------------

def maybe_calibrate_model(
    model,
    X_calib: np.ndarray,
    y_calib: np.ndarray,
    cfg: Config,
    model_name: str,
):
    """
    Optionally wrap a classifier with CalibratedClassifierCV using the validation set.
    """
    if not getattr(cfg, "use_probability_calibration", False):
        return model, None

    method = getattr(cfg, "calibration_method", "sigmoid")
    logger.info(
        "[CALIBRATION] Calibrating %s probabilities using method='%s' on validation set",
        model_name,
        method,
    )
    try:
        calib = CalibratedClassifierCV(model, method=method, cv="prefit")
        calib.fit(X_calib, y_calib)
        return calib, {"method": method}
    except Exception as e:
        logger.warning(
            "Calibration for %s failed (%s); using raw model probabilities",
            model_name,
            e,
        )
        return model, None


def compute_feature_importance(
    model,
    feature_names: List[str],
    importance_type: str = "weight",
    top_k: int = 20,
):
    """
    Compute and log feature importance from a trained model.
    """
    logger.info("[STEP] Computing feature importance (%s)", importance_type)

    # Unwrap calibrated models
    inner_model = getattr(model, "base_estimator", model)

    importances = None

    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
    elif hasattr(inner_model, "get_booster"):
        booster = inner_model.get_booster()
        importance_dict = booster.get_score(importance_type=importance_type)
        importances = np.zeros(len(feature_names))
        for feat_name, score in importance_dict.items():
            if feat_name.startswith("f"):
                try:
                    idx = int(feat_name[1:])
                    if idx < len(importances):
                        importances[idx] = score
                except (ValueError, IndexError):
                    pass

    if importances is None:
        logger.warning("Could not compute feature importance; model type not supported")
        return None

    importance_pairs = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )

    logger.info("  Top %d features:", top_k)
    for feat, imp in importance_pairs[:top_k]:
        logger.info("    %s: %.6f", feat, imp)

    return importance_pairs


def explain_model_with_shap(
    model,
    X_sample: np.ndarray,
    model_name: str = "Model",
    max_samples: int = 100,
):
    """
    Generate SHAP explanations for model predictions (if SHAP is installed).
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed; skipping model explainability analysis")
        return None

    logger.info("[STEP] Computing SHAP explanations for %s", model_name)

    inner_model = getattr(model, "base_estimator", model)

    if len(X_sample) > max_samples:
        indices = np.random.choice(len(X_sample), max_samples, replace=False)
        X_explain = X_sample[indices]
        logger.info(
            "  Using %d samples for explanation (out of %d)", max_samples, len(X_sample)
        )
    else:
        X_explain = X_sample

    try:
        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer.shap_values(X_explain)
        logger.info("  SHAP TreeExplainer created; computed %d samples", len(shap_values))
        return explainer, shap_values
    except Exception as e:
        logger.warning("SHAP explanation failed (%s); skipping", e)
        return None


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def evaluate_model(
    name: str,
    model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: Config,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with multiple metrics, threshold tuning, and explainability.
    """
    logger.info("=" * 70)
    logger.info("Evaluating model: %s", name)

    # Extract probabilities
    if hasattr(model, "predict_proba"):
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        val_scores = model.decision_function(X_val)
        test_scores = model.decision_function(X_test)
        val_min, val_max = val_scores.min(), val_scores.max()
        test_min, test_max = test_scores.min(), test_scores.max()
        val_proba = (val_scores - val_min) / (val_max - val_min + 1e-9)
        test_proba = (test_scores - test_min) / (test_max - test_min + 1e-9)
    else:
        raise ValueError(f"Model {name} has neither predict_proba nor decision_function")

    # Core metrics: AUC, PR-AUC
    val_auc = roc_auc_score(y_val, val_proba)
    val_ap = average_precision_score(y_val, val_proba)
    test_auc = roc_auc_score(y_test, test_proba)
    test_ap = average_precision_score(y_test, test_proba)

    logger.info(
        "[%s] Validation AUC=%.4f, PR-AUC=%.4f | Test AUC=%.4f, PR-AUC=%.4f",
        name,
        val_auc,
        val_ap,
        test_auc,
        test_ap,
    )
    val_percentiles = np.percentile(val_proba, [50, 75, 90, 95, 99, 99.5, 99.9])
    test_percentiles = np.percentile(test_proba, [50, 75, 90, 95, 99, 99.5, 99.9])
    logger.info("[%s] Val proba percentiles P50..P99.9: %s", name, val_percentiles.tolist())
    logger.info("[%s] Test proba percentiles P50..P99.9: %s", name, test_percentiles.tolist())

    threshold_strategy = getattr(cfg, "threshold_strategy", "metric").lower()
    threshold_metric = getattr(cfg, "threshold_metric", "f_beta")

    if threshold_strategy == "cost":
        best_t, best_cost, threshold_metrics = find_cost_optimal_threshold(
            y_val,
            val_proba,
            fn_cost=getattr(cfg, "fn_cost", 10.0),
            fp_cost=getattr(cfg, "fp_cost", 1.0),
        )
        best_score = -best_cost
        logger.info(
            "[%s] Best threshold (cost-based on val) = %.4f, cost=%.6f, FPR=%.4f, TPR=%.4f",
            name,
            best_t,
            best_cost,
            threshold_metrics.get("fpr", 0.0),
            threshold_metrics.get("tpr", 0.0),
        )
    elif threshold_strategy in {"alert_rate", "volume"} and getattr(cfg, "target_alert_rate", None) is not None:
        best_t, threshold_metrics = find_threshold_for_target_alert_rate(
            y_val,
            val_proba,
            cfg.target_alert_rate,
        )
        best_score = threshold_metrics.get("f_beta", 0.0)
        logger.info(
            "[%s] Best threshold (alert-rate on val) = %.4f, alert_rate=%.4f, precision=%.4f, recall=%.4f",
            name,
            best_t,
            threshold_metrics.get("alert_rate", 0.0),
            threshold_metrics.get("precision", 0.0),
            threshold_metrics.get("recall", 0.0),
        )
    else:
        best_t, best_score, threshold_metrics = find_best_threshold(
            y_val,
            val_proba,
            beta=cfg.f_beta,
            metric=threshold_metric,
        )
        logger.info(
            "[%s] Best threshold (%s on val) = %.4f, %s=%.4f",
            name,
            threshold_metric,
            best_t,
            threshold_metric,
            best_score,
        )
        logger.info(
            "  └─ Precision=%.4f, Recall=%.4f",
            threshold_metrics.get("precision", 0.0),
            threshold_metrics.get("recall", 0.0),
        )

    # Apply threshold and compute detailed metrics
    val_pred = (val_proba >= best_t).astype(int)
    test_pred = (test_proba >= best_t).astype(int)

    val_f1 = f1_score(y_val, val_pred)
    test_f1 = f1_score(y_test, test_pred)
    val_precision = precision_score(y_val, val_pred, zero_division=0)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    val_recall = recall_score(y_val, val_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)

    logger.info(
        "[%s] Test Metrics: F1=%.4f, Precision=%.4f, Recall=%.4f",
        name,
        test_f1,
        test_precision,
        test_recall,
    )

    val_tn, val_fp, val_fn, val_tp = confusion_matrix(y_val, val_pred).ravel()
    test_tn, test_fp, test_fn, test_tp = confusion_matrix(y_test, test_pred).ravel()
    eps = 1e-9
    val_fpr = val_fp / (val_fp + val_tn + eps)
    test_fpr = test_fp / (test_fp + test_tn + eps)
    val_trr = (val_tp + val_fp) / max(1, len(y_val))
    test_trr = (test_tp + test_fp) / max(1, len(y_test))
    specificity = test_tn / (test_tn + test_fp + eps)

    logger.info(
        "[%s] Validation Confusion: TP=%d, FP=%d, FN=%d, TN=%d | VFPR=%.6f, TRR=%.6f",
        name,
        val_tp,
        val_fp,
        val_fn,
        val_tn,
        val_fpr,
        val_trr,
    )
    logger.info(
        "[%s] Test Confusion: TP=%d, FP=%d, FN=%d, TN=%d | TFPR=%.6f, TRR=%.6f, Specificity=%.4f",
        name,
        test_tp,
        test_fp,
        test_fn,
        test_tn,
        test_fpr,
        test_trr,
        specificity,
    )

    if cfg.verbose_reports:
        logger.info("[%s] Classification report (Test):", name)
        logger.info("\n%s", classification_report(y_test, test_pred, digits=3))

    k, prec_k, rec_k = precision_recall_at_k(y_test, test_proba, frac=cfg.top_k_frac)
    logger.info(
        "[%s] Precision@top-%.1f%% (K=%d): precision=%.4f, recall=%.4f",
        name,
        cfg.top_k_frac * 100,
        k,
        prec_k,
        rec_k,
    )

    if feature_names:
        compute_feature_importance(model, feature_names)

    shap_result = None
    if feature_names:
        shap_result = explain_model_with_shap(model, X_test[:100], model_name=name)

    metrics: Dict[str, Any] = {
        "name": name,
        "val_auc": float(val_auc),
        "val_ap": float(val_ap),
        "val_f1": float(val_f1),
        "val_precision": float(val_precision),
        "val_recall": float(val_recall),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "test_f1": float(test_f1),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_specificity": float(specificity),
        "val_fpr": float(val_fpr),
        "test_fpr": float(test_fpr),
        "val_trr": float(val_trr),
        "test_trr": float(test_trr),
        "best_threshold": float(best_t),
        "best_threshold_metric_val": float(best_score),
        "precision_at_k": float(prec_k),
        "recall_at_k": float(rec_k),
        "k": int(k),
        "confusion_matrix": {
            "TP": int(test_tp),
            "FP": int(test_fp),
            "FN": int(test_fn),
            "TN": int(test_tn),
        },
        "threshold_strategy": threshold_strategy,
        "threshold_metrics_val": threshold_metrics,
        "shap_result": shap_result,
    }

    return metrics


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------

def run_fraud_pipeline() -> Dict[str, Dict[str, Any]]:
    """
    Orchestrate the full pipeline end-to-end.

    Returns a dict: model_name -> metrics dict.
    """
    seed_everything(42)
    cfg = Config()
    cfg = apply_env_overrides(cfg)

    logger.info("=" * 70)
    logger.info("TECHM FRAUD DETECTION PIPELINE - GOD TIER LOCAL VERSION (modular)")
    logger.info("=" * 70)
    logger.info("Config:\n%s", json.dumps({k: str(v) for k, v in cfg.__dict__.items()}, indent=2))

    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load JSONL sample data
    df_trx_raw, df_ecm_raw = load_sample_jsonl(cfg)

    # 2. Normalize types
    df_trx_clean, df_ecm_clean = clean_and_normalize_raw(df_trx_raw, df_ecm_raw, cfg)

    # 3. Merge and label
    merged = merge_trx_and_ecm(df_trx_clean, df_ecm_clean, cfg)

    # 4. Split train/val/test
    train_df, val_df, test_df = split_train_val_test(merged, cfg)

    # 5. Feature engineering (stats computed from FULL train_df)
    train_fe_full, val_fe, test_fe = engineer_features(train_df, val_df, test_df, cfg)

    # 6. For model training, build a TRAIN SAMPLE with higher fraud rate
    train_fe = make_training_sample(train_fe_full, cfg)

    # 7. Build feature matrices
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocess_pipeline,
    ) = build_feature_matrix(train_fe, val_fe, test_fe, cfg)

    # 8. Imbalance handling
    X_train_rf, y_train_rf = resample_training_data(X_train, y_train, cfg)
    X_train_boost, y_train_boost = X_train, y_train

    # 9. Extract feature names for importance analysis and explainability
    preprocessor = preprocess_pipeline.named_steps["preprocessor"]
    feature_names: List[str] = []
    if hasattr(preprocessor, "transformers_"):
        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(columns)
            elif name == "cat":
                ohe = transformer.named_steps.get("ohe")
                if ohe is not None and hasattr(ohe, "get_feature_names_out"):
                    try:
                        ohe_names = ohe.get_feature_names_out(columns)
                        feature_names.extend(ohe_names)
                    except Exception:
                        feature_names.extend(columns)
                else:
                    feature_names.extend(columns)

    logger.info("[STEP] Final feature matrix has %d features", len(feature_names))

    results: Dict[str, Dict[str, Any]] = {}

    # 10. Train & evaluate RandomForest
    logger.info("\n[TRAIN] RandomForest Model")
    rf = train_random_forest(X_train_rf, y_train_rf, X_val, y_val, cfg)
    rf_calibrated, _ = maybe_calibrate_model(rf, X_val, y_val, cfg, "RandomForest")
    rf_metrics = evaluate_model(
        "RandomForest",
        rf_calibrated,
        X_val,
        y_val,
        X_test,
        y_test,
        cfg,
        feature_names=feature_names,
    )
    results["RandomForest"] = rf_metrics

    # 11. Train & evaluate XGBoost
    logger.info("\n[TRAIN] XGBoost Model")
    xgb = train_xgboost(X_train_boost, y_train_boost, X_val, y_val, cfg)
    if xgb is not None:
        xgb_calibrated, _ = maybe_calibrate_model(xgb, X_val, y_val, cfg, "XGBoost")
        xgb_metrics = evaluate_model(
            "XGBoost",
            xgb_calibrated,
            X_val,
            y_val,
            X_test,
            y_test,
            cfg,
            feature_names=feature_names,
        )
        results["XGBoost"] = xgb_metrics

    # 12. Summary and best model selection
    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETE!")
    logger.info("=" * 70)
    logger.info("Models trained: %s", list(results.keys()))

    ranking_metric = "test_auc"
    if results:
        best_model_name = max(
            results.keys(),
            key=lambda name: results[name].get(ranking_metric, 0.0),
        )
        best_metrics = results[best_model_name]
        logger.info("\n[BEST MODEL] Summary:")
        logger.info("  Model: %s", best_model_name)
        logger.info("  Ranking metric (%s): %.4f", ranking_metric, best_metrics.get(ranking_metric, 0.0))
        logger.info("  Test AUC: %.4f", best_metrics.get("test_auc", 0.0))
        logger.info("  Test PR-AUC: %.4f", best_metrics.get("test_ap", 0.0))
        logger.info("  Test F1: %.4f", best_metrics.get("test_f1", 0.0))
        logger.info("  Test Precision: %.4f", best_metrics.get("test_precision", 0.0))
        logger.info("  Test Recall: %.4f", best_metrics.get("test_recall", 0.0))
        logger.info("  Optimal Threshold: %.4f", best_metrics.get("best_threshold", 0.0))

    return results


# ------------------------------------------------------------------------------
# Tiny CLI
# ------------------------------------------------------------------------------

def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if len(argv) == 2 and argv[0] == "run" and argv[1] == "fraud_pipeline":
        run_fraud_pipeline()
    else:
        print("Usage: python cli.py run fraud_pipeline")
        sys.exit(1)


if __name__ == "__main__":
    main()
