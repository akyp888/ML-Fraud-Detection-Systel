#!/usr/bin/env python3
"""
TechM Fraud Detection Pipeline - God Tier Local Version
=======================================================

This script is a refactored, overfitting-resistant, feature-rich local pipeline
for fraud detection. It is designed to:

- Load sample JSONL data (transactions + ECM) from local_data/
- Intentionally "mess up" the data to simulate noisy, real-world conditions
- Merge transactions with ECM outcomes and create a clean binary label
- Engineer strong transactional + entity / graph-like features
- Handle extreme class imbalance with advanced resampling (SMOTE-Tomek, etc.)
- Train multiple tree-based models with regularization:
    - RandomForest (scikit-learn)
    - XGBoost (optional, if installed)
- Tune probability thresholds based on validation F1/F2 and Precision-Recall
- Evaluate on a held-out test set, including precision@K

It is intentionally "god tier": heavily instrumented, modular, and robust against
messy data. You can later adapt the same ideas to Spark / cluster scale.

Requirements (recommended):
    pip install pandas numpy scikit-learn imbalanced-learn xgboost

Usage:
    python techm_local_pipeline.py
"""

import os
import sys
import json
import math
import logging
import random
import inspect
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
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
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV

# Optional dependencies --------------------------------------------------------
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None

try:
    from imblearn.combine import SMOTETomek  # type: ignore
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE  # type: ignore
    from imblearn.under_sampling import TomekLinks  # type: ignore
except Exception:
    SMOTETomek = None
    BorderlineSMOTE = None
    SMOTE = None
    TomekLinks = None

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger("techm-fraud-god-tier")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
)
logger.addHandler(_handler)


# ------------------------------------------------------------------------------
# Config dataclass
# ------------------------------------------------------------------------------

@dataclass
class Config:
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = Path(__file__).resolve().parent / "local_data/sample"
    sample_dir: Path = Path(__file__).resolve().parent / "local_data/sample"
    transactions_jsonl: str = "transactions.jsonl"
    ecm_jsonl: str = "ecm.jsonl"

    # Columns (these should match your real schema if present)
    label_col: str = "label"
    ecm_label_col: str = "RESULT_TYPE_CD"   # column in ECM table to derive label from
    trx_id_col: str = "CORRELATIVE_NO"
    customer_col: str = "CUSTOMER_NO"
    account_col: str = "ACCOUNT_NO"
    dest_account_col: str = "DESTINATION_ACCOUNT_NO"
    device_col: str = "DEVICE_HASH_NO"
    ip_col: str = "CONNECTION_IP_NO"
    phone_col: str = "PHONE_CELL_NO"
    email_col: str = "EMAIL_LINE"
    trx_amount_col: str = "TRX_TOTAL_AMT"
    trx_date_col: str = "TRX_DATE"
    trx_hour_col: str = "TRX_HOUR"

    # Train/val/test split
    random_state: int = 42
    val_size: float = 0.15
    test_size: float = 0.15

    # Threshold tuning
    f_beta: float = 1.0          # F1; can set to >1 for recall-heavy F2
    top_k_frac: float = 0.01     # precision@top 1% of scores
    threshold_metric: str = "f_beta"  # which metric find_best_threshold optimizes

    # Threshold strategy / business constraints
    # "metric": optimize cfg.threshold_metric (F_beta etc.)
    # "cost":   minimize expected cost given fn_cost / fp_cost
    # "volume": fix alert volume via cfg.target_alert_rate
    threshold_strategy: str = "metric"
    fn_cost: float = 10.0
    fp_cost: float = 1.0
    target_alert_rate: Optional[float] = None  # e.g. 0.005 for 0.5% of tx flagged

    # Probability calibration
    use_probability_calibration: bool = False
    calibration_method: str = "sigmoid"  # "sigmoid" (Platt) or "isotonic"

    # Real world prevalence + training sampling for ultra-rare fraud
    # real_fraud_rate can be set from actual Scotiabank rate (~0.00037 or similar)
    real_fraud_rate: Optional[float] = None
    # We deliberately oversample fraud in TRAIN ONLY to this rate
    train_target_fraud_rate: float = 0.02  # 2% fraud in training sample
    max_nonfraud_train: Optional[int] = None  # hard cap on non-fraud rows in train

    # Imbalance handling (synthetic oversampling on top of downsampling)
    use_smote_tomek: bool = False
    smote_kind: str = "borderline"  # or "regular"

    # Misc
    verbose_reports: bool = True

    # RandomForest controls (exposed so we can tune speed quickly)
    rf_n_estimators: int = 200
    rf_warm_start_chunk: int = 50
    rf_n_jobs: int = -1

    # XGBoost controls
    xgb_use_class_weight: bool = False
    xgb_scale_pos_clip: float = 50.0

    # Encoding / feature controls
    target_encode_whitelist: Tuple[str, ...] = tuple(
        (
            "CITY_OR_LOCATION_NAME IP_COUNTRY_CD IP_REGION_NAME IP_REGION_CD "
            "PRODUCT_CD SUBPRODUCT_CD CURRENCY_CD TRX_ORIGIN_CD TRX_GROUP_CD "
            "DEVICE_OPERATING_SYSTEM_NO OS_NAME LANGUAGE_CD "
            "CUSTOMER_SEGMENT_NAME PROFESSION_NAME ECONOMIC_NAME "
            "DESTINATION_COUNTRY_CD DESTINATION_ENTITY_NAME TARGET_BANK_CD "
            "DESTINATION_PRODUCT_TYPE_CD LOCAL_OR_INTERNATIONAL_CD "
            "CLIENT_RIM_TARGET_CD"
        ).split()
    )
    id_like_cols: Tuple[str, ...] = tuple(
        (
            "SOURCE_ID CORRELATIVE_NO SESSION_ID GUID_NRO SERIAL_NRO "
            "RESULTING_DBFD_GUID_DESC REFERENCE_NO SERVICE_PAYMENT_REF_3_NO "
            "SERVICE_PAYMENT_REF_4_NO COOKIE_TEXT LOCAL_STORAGE_VALUE_TEXT "
            "RESULTING_REGISTRATION_CD HASHINTEGRITY_NO USER_DEVICE_PATTERN_DDS_NO "
            "DEVICE_NO MACADDRESS_NO IMEI_NO"
        ).split()
    )

    # Date parsing
    date_cols: Tuple[str, ...] = tuple(
        (
            "AUDIT_DT LAST_UPD_DT TRANSACTION_DT_TIME TRX_DT "
            "LAST_MOVEMENT_ACCOUNT_DATE ACCOUNT_OPENING_DATE CUSTOMER_BONDING_DATE "
            "DST_ACC_OP_DATE CLOSING_DATE"
        ).split()
    )
    int_date_cols: Tuple[str, ...] = tuple(
        (
            "TRX_DATE TRANSACTION_DT SEND_DATE RECEPTION_DATE CONS_MASIVIAN_DATE "
            "RESPONSE_MASIVIAN_DATE LAST_UP_DATE LAST_CONNECTION_JUMP_DATE "
            "LAST_REF_PAYMENT_DATE BIRTH_DATE"
        ).split()
    )

    def entity_columns(self) -> Tuple[str, ...]:
        return tuple(
            c
            for c in [
                self.customer_col,
                self.account_col,
                self.dest_account_col,
                self.device_col,
                self.ip_col,
                self.phone_col,
                self.email_col,
            ]
            if c
        )


def _parse_bool_env(value: str) -> bool:
    val = value.strip().lower()
    if val in {"1", "true", "yes", "y", "on"}:
        return True
    if val in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean env value '{value}'")


def apply_env_overrides(cfg: Config) -> Config:
    """
    Allow quick tuning without editing the script by honoring a few env vars.
    """
    env_map: Dict[str, Tuple[str, Any]] = {
        "RF_N_ESTIMATORS": ("rf_n_estimators", int),
        "RF_WARM_CHUNK": ("rf_warm_start_chunk", int),
        "RF_N_JOBS": ("rf_n_jobs", int),
        "USE_SMOTE_TOMEK": ("use_smote_tomek", _parse_bool_env),
    }

    for env_key, (field_name, caster) in env_map.items():
        raw_val = os.getenv(env_key)
        if raw_val is None:
            continue
        try:
            new_val = caster(raw_val)
            setattr(cfg, field_name, new_val)
            logger.info(f"Overriding config via ${env_key}: {field_name}={new_val}")
        except Exception as exc:  # noqa: BLE001 - log helpful context
            logger.warning(
                "Failed to apply env override %s=%s: %s", env_key, raw_val, exc
            )
    return cfg


# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed all known random generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def safe_parse_date(series: pd.Series) -> pd.Series:
    """Best-effort parse of a date-like series to pandas datetime."""
    return pd.to_datetime(series, errors="coerce")


def safe_parse_int(series: pd.Series) -> pd.Series:
    """Coerce to numeric integers; invalid -> NaN."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def safe_parse_float(series: pd.Series) -> pd.Series:
    """Coerce to floats; invalid -> NaN."""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def parse_int_yyyymmdd(series: pd.Series) -> pd.Series:
    """Parse integer-coded YYYYMMDD values into pandas datetime."""
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    string_values = numeric.astype(str).replace("<NA>", np.nan)
    return pd.to_datetime(string_values, format="%Y%m%d", errors="coerce")


def make_data_messy(
    df_trx: pd.DataFrame,
    df_ecm: pd.DataFrame,
    cfg: Config,
    rng: np.random.RandomState,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Intentionally inject noise and messiness into the data to stress-test the pipeline.

    (Currently not used in run_pipeline; kept for experimentation.)
    """

    df_trx = df_trx.copy()
    df_ecm = df_ecm.copy()

    # 1. Randomly create missing values in some columns
    noisy_cols = [
        cfg.trx_amount_col,
        cfg.trx_date_col,
        cfg.trx_hour_col,
        cfg.device_col,
        cfg.ip_col,
        cfg.phone_col,
        cfg.email_col,
    ]
    for col in noisy_cols:
        if col in df_trx.columns:
            mask = np.random.rand(len(df_trx)) < 0.03  # 3% missing
            df_trx.loc[mask, col] = np.nan

    # 2. Perturb transaction amounts
    if cfg.trx_amount_col in df_trx.columns:
        amt = df_trx[cfg.trx_amount_col].copy()
        amt = pd.to_numeric(amt, errors="coerce")
        noise_mask = np.random.rand(len(amt)) < 0.15  # 15% rows get noisy
        noise_factor = np.random.lognormal(mean=0.0, sigma=0.75, size=noise_mask.sum())
        amt.loc[noise_mask] = amt.loc[noise_mask] * noise_factor
        df_trx[cfg.trx_amount_col] = amt

    # 3. Create new weird categories in some character columns
    cat_cols = [
        cfg.device_col,
        cfg.ip_col,
        cfg.phone_col,
        cfg.email_col,
        "TRX_ORIGIN_CD",
        "CURRENCY_CD",
    ]
    for col in cat_cols:
        if col in df_trx.columns:
            weird_mask = np.random.rand(len(df_trx)) < 0.02  # 2%
            df_trx.loc[weird_mask, col] = "ZZZ_UNKNOWN_" + col

    # 4. Mutate ECM label strings (mixed case, synonyms)
    if cfg.ecm_label_col in df_ecm.columns:
        ecm = df_ecm[cfg.ecm_label_col].astype(str)
        # introduce some noise
        variants = {
            "FRAUD": ["FRAUD", "fraud", "Fraud", "FRAUDULENT"],
            "GENUINE": ["GENUINE", "genuine", "Genuine", "LEGIT"],
        }

        def randomize_state(s: str) -> str:
            s_up = s.strip().upper()
            if "FRAUD" in s_up:
                return np.random.choice(variants["FRAUD"])
            elif "GENUINE" in s_up or "LEGIT" in s_up:
                return np.random.choice(variants["GENUINE"])
            else:
                # occasionally flip or make it weird
                if np.random.rand() < 0.5:
                    return np.random.choice(variants["FRAUD"] + variants["GENUINE"])
                return s

        df_ecm[cfg.ecm_label_col] = ecm.apply(randomize_state)

    # 5. Duplicate a small % of transactions to simulate duplicates
    if len(df_trx) > 0:
        dup_mask = np.random.rand(len(df_trx)) < 0.01  # 1%
        dup_rows = df_trx[dup_mask]
        if not dup_rows.empty:
            df_trx = pd.concat([df_trx, dup_rows], ignore_index=True)

    return df_trx, df_ecm


# ------------------------------------------------------------------------------
# Data loading and merging
# ------------------------------------------------------------------------------

def load_sample_jsonl(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load transactions.jsonl and ecm.jsonl from cfg.sample_dir.

    Raises if they don't exist (we can later add synthetic generation if needed).
    """
    trx_path = cfg.sample_dir / cfg.transactions_jsonl
    ecm_path = cfg.sample_dir / cfg.ecm_jsonl

    if not trx_path.exists() or not ecm_path.exists():
        raise FileNotFoundError(
            f"Sample JSONL files not found in {cfg.sample_dir}. "
            f"Expected {cfg.transactions_jsonl} and {cfg.ecm_jsonl}"
        )

    logger.info(f"Reading transactions JSONL: {trx_path}")
    df_trx = pd.read_json(trx_path, lines=True)

    logger.info(f"Reading ECM JSONL: {ecm_path}")
    df_ecm = pd.read_json(ecm_path, lines=True)

    return df_trx, df_ecm


def clean_and_normalize_raw(
    df_trx: pd.DataFrame,
    df_ecm: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform basic type cleaning and normalization on raw JSONL-loaded data.

    - Parse dates/hours where present
    - Ensure IDs are treated consistently as strings
    """
    df_trx = df_trx.copy()
    df_ecm = df_ecm.copy()

    # Normalize transaction hour
    if cfg.trx_hour_col in df_trx.columns:
        df_trx[cfg.trx_hour_col] = safe_parse_int(df_trx[cfg.trx_hour_col])

    # Parse known datetime columns (string-coded)
    date_cols = set(cfg.date_cols)
    int_date_cols = set(cfg.int_date_cols)
    for col in date_cols:
        if col in df_trx.columns:
            df_trx[col] = safe_parse_date(df_trx[col])
        if col in df_ecm.columns:
            df_ecm[col] = safe_parse_date(df_ecm[col])
    for col in int_date_cols:
        if col in df_trx.columns:
            df_trx[col] = parse_int_yyyymmdd(df_trx[col])
        if col in df_ecm.columns:
            df_ecm[col] = parse_int_yyyymmdd(df_ecm[col])

    # Normalize amount
    if cfg.trx_amount_col in df_trx.columns:
        df_trx[cfg.trx_amount_col] = safe_parse_float(df_trx[cfg.trx_amount_col])

    if "CLOSING_HOUR" in df_ecm.columns:
        df_ecm["CLOSING_HOUR"] = safe_parse_int(df_ecm["CLOSING_HOUR"])

    # Ensure key IDs are strings (avoid mixing numeric/object)
    id_cols = [
        cfg.trx_id_col,
        cfg.customer_col,
        cfg.account_col,
        cfg.dest_account_col,
        cfg.device_col,
        cfg.ip_col,
        cfg.phone_col,
        cfg.email_col,
    ]
    for col in id_cols:
        for df in (df_trx, df_ecm):
            if col in df.columns:
                df[col] = df[col].astype(str)

    return df_trx, df_ecm


def merge_trx_and_ecm(
    df_trx: pd.DataFrame,
    df_ecm: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Merge transaction and ECM datasets, deduplicate, and create binary label.

    - Left-join on CORRELATIVE_NO (or cfg.trx_id_col)
    - Derive label from cfg.ecm_label_col in ECM
    - Drop ECM columns that could leak investigation outcomes
    """
    df_trx = df_trx.copy()
    df_ecm = df_ecm.copy()

    if cfg.trx_id_col not in df_trx.columns:
        raise KeyError(f"Transaction ID column {cfg.trx_id_col} not found in trx data")
    if cfg.trx_id_col not in df_ecm.columns:
        raise KeyError(f"Transaction ID column {cfg.trx_id_col} not found in ECM data")

    # Deduplicate: keep latest ECM per transaction if closing date/hour present
    if "CLOSING_DATE" in df_ecm.columns:
        df_ecm = df_ecm.sort_values(["CORRELATIVE_NO", "CLOSING_DATE"]).drop_duplicates(
            subset=[cfg.trx_id_col], keep="last"
        )
    else:
        df_ecm = df_ecm.drop_duplicates(subset=[cfg.trx_id_col], keep="last")

    # Merge
    merged = df_trx.merge(df_ecm, on=cfg.trx_id_col, how="left", suffixes=("", "_ECM"))
    logger.info(f"Merged shape: {merged.shape[0]} rows, {merged.shape[1]} columns")

    # Create label from FRAUD_IND (if available) or fall back to ECM label column
    if "FRAUD_IND" in merged.columns:
        # Use the true fraud indicator from transactions (not corrupted by make_data_messy)
        label_series = merged["FRAUD_IND"].astype(str).str.strip()
        merged[cfg.label_col] = np.where(
            label_series.isin(["1", "true", "True", "TRUE"]), 1, 0
        )
    elif cfg.ecm_label_col in merged.columns:
        # Fallback to ECM label if FRAUD_IND not available
        label_series = merged[cfg.ecm_label_col].astype(str).str.upper().str.strip()
        is_fraud = label_series.str.contains("FRAUD")
        is_genuine = label_series.str.contains("GENUINE") | label_series.str.contains("LEGIT")
        merged[cfg.label_col] = np.where(is_fraud, 1, 0)
    else:
        raise KeyError(f"Neither FRAUD_IND nor {cfg.ecm_label_col} found in merged data")

    logger.info(
        f"Label distribution after merge: "
        f"non-fraud={int((merged[cfg.label_col] == 0).sum())}, "
        f"fraud={int((merged[cfg.label_col] == 1).sum())}"
    )

    # Drop outcome/ECM fields that would leak the label
    leak_cols = [
        "FRAUD_IND",  # True fraud indicator (would leak the label)
        cfg.ecm_label_col,
        "RESULT_TYPE_CD",
        "SUBTYPE_RESULT_CD",
        "ANALYST_ID",
        "CASE_STATUS_CD",
        "CLOSING_DATE",
        "CLOSING_HOUR",
    ]
    leak_cols_present = [c for c in leak_cols if c in merged.columns]
    merged = merged.drop(columns=leak_cols_present, errors="ignore")

    # Drop exact duplicates after merge
    merged = merged.drop_duplicates(subset=[cfg.trx_id_col], keep="last")

    return merged


# ------------------------------------------------------------------------------
# Train / Validation / Test splitting
# ------------------------------------------------------------------------------

def split_train_val_test(
    df: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test with fraud preservation.

    Prefer stratified temporal split if transaction date is available;
    otherwise stratified random split.

    Key: Ensures fraud cases are distributed across all splits for proper evaluation.
    If any split has 0 fraud and total fraud > 0, rebalance splits to ensure each has fraud.
    """
    df = df.copy()
    y = df[cfg.label_col].values
    total_fraud = int(y.sum())

    if cfg.trx_date_col in df.columns:
        logger.info("Using stratified temporal split by TRX_DATE (preserves fraud ratio)")
        df = df.sort_values(cfg.trx_date_col)

        # Stratified temporal: preserve fraud ratio in each temporal bucket
        fraud_mask = df[cfg.label_col] == 1
        fraud_df = df[fraud_mask].sort_values(cfg.trx_date_col)
        non_fraud_df = df[~fraud_mask].sort_values(cfg.trx_date_col)

        n_fraud = len(fraud_df)
        n_non_fraud = len(non_fraud_df)

        # Calculate split sizes for fraud data
        n_train_fraud = int((1.0 - cfg.val_size - cfg.test_size) * n_fraud)
        n_val_fraud = int(cfg.val_size * n_fraud)

        # Calculate split sizes for non-fraud data
        n_train_non_fraud = int((1.0 - cfg.val_size - cfg.test_size) * n_non_fraud)
        n_val_non_fraud = int(cfg.val_size * n_non_fraud)

        # Split fraud and non-fraud temporally
        train_fraud = fraud_df.iloc[:n_train_fraud]
        val_fraud = fraud_df.iloc[n_train_fraud: n_train_fraud + n_val_fraud]
        test_fraud = fraud_df.iloc[n_train_fraud + n_val_fraud:]

        train_non_fraud = non_fraud_df.iloc[:n_train_non_fraud]
        val_non_fraud = non_fraud_df.iloc[n_train_non_fraud: n_train_non_fraud + n_val_non_fraud]
        test_non_fraud = non_fraud_df.iloc[n_train_non_fraud + n_val_non_fraud:]

        # Combine fraud and non-fraud
        train = pd.concat([train_fraud, train_non_fraud], ignore_index=True)
        val = pd.concat([val_fraud, val_non_fraud], ignore_index=True)
        test = pd.concat([test_fraud, test_non_fraud], ignore_index=True)

        logger.info(
            f"  Stratified temporal split: fraud split as "
            f"{n_train_fraud}/{n_val_fraud}/{len(test_fraud)}"
        )
    else:
        logger.info("Using stratified random split (no TRX_DATE available)")
        train_val, test = train_test_split(
            df,
            test_size=cfg.test_size,
            stratify=y,
            random_state=cfg.random_state,
        )
        y_train_val = train_val[cfg.label_col].values
        train, val = train_test_split(
            train_val,
            test_size=cfg.val_size / (1.0 - cfg.test_size),
            stratify=y_train_val,
            random_state=cfg.random_state,
        )

    train_fraud = int(train[cfg.label_col].sum())
    val_fraud = int(val[cfg.label_col].sum())
    test_fraud = int(test[cfg.label_col].sum())

    logger.info(
        f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)} "
        f"(fraud in train={train_fraud}, fraud in val={val_fraud}, fraud in test={test_fraud})"
    )

    # Handle case where any split has 0 fraud
    if total_fraud > 0 and (train_fraud == 0 or val_fraud == 0 or test_fraud == 0):
        logger.warning(
            f"Imbalanced fraud distribution: train={train_fraud}, "
            f"val={val_fraud}, test={test_fraud}"
        )
        logger.warning("Rebalancing splits to ensure each has at least 1 fraud case...")

        # Collect all data and labels
        all_data = pd.concat([train, val, test], ignore_index=True)
        fraud_cases = all_data[all_data[cfg.label_col] == 1]
        non_fraud_cases = all_data[all_data[cfg.label_col] == 0]

        # Ensure each split gets at least 1 fraud
        min_fraud_per_split = 1
        required_fraud = 3  # one for each split
        available_fraud = len(fraud_cases)

        if available_fraud >= required_fraud:
            # Distribute fraud evenly
            fraud_train_count = max(min_fraud_per_split, available_fraud // 3)
            fraud_val_count = max(
                min_fraud_per_split, (available_fraud - fraud_train_count) // 2
            )
            fraud_test_count = available_fraud - fraud_train_count - fraud_val_count

            # Randomly distribute fraud cases
            fraud_indices = np.random.RandomState(cfg.random_state).permutation(
                available_fraud
            )
            fraud_train_idx = fraud_indices[:fraud_train_count]
            fraud_val_idx = fraud_indices[
                fraud_train_count: fraud_train_count + fraud_val_count
            ]
            fraud_test_idx = fraud_indices[fraud_train_count + fraud_val_count:]

            # Distribute non-fraud proportionally
            available_non_fraud = len(non_fraud_cases)
            non_fraud_train_count = int(available_non_fraud * 0.70)
            non_fraud_val_count = int(available_non_fraud * 0.15)
            non_fraud_test_count = (
                available_non_fraud - non_fraud_train_count - non_fraud_val_count
            )

            non_fraud_indices = np.random.RandomState(
                cfg.random_state
            ).permutation(available_non_fraud)
            non_fraud_train_idx = non_fraud_indices[:non_fraud_train_count]
            non_fraud_val_idx = non_fraud_indices[
                non_fraud_train_count: non_fraud_train_count + non_fraud_val_count
            ]
            non_fraud_test_idx = non_fraud_indices[
                non_fraud_train_count + non_fraud_val_count:
            ]

            # Reconstruct splits
            train = pd.concat(
                [fraud_cases.iloc[fraud_train_idx], non_fraud_cases.iloc[non_fraud_train_idx]],
                ignore_index=True,
            )
            val = pd.concat(
                [fraud_cases.iloc[fraud_val_idx], non_fraud_cases.iloc[non_fraud_val_idx]],
                ignore_index=True,
            )
            test = pd.concat(
                [fraud_cases.iloc[fraud_test_idx], non_fraud_cases.iloc[non_fraud_test_idx]],
                ignore_index=True,
            )

            logger.info(
                "Rebalanced split sizes: train=%d, val=%d, test=%d "
                "(fraud in train=%d, fraud in val=%d, fraud in test=%d)",
                len(train),
                len(val),
                len(test),
                int(train[cfg.label_col].sum()),
                int(val[cfg.label_col].sum()),
                int(test[cfg.label_col].sum()),
            )
        else:
            logger.warning(
                f"Very few fraud cases ({available_fraud}); cannot guarantee all splits have fraud"
            )

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ------------------------------------------------------------------------------
# Train sampling for ultra-rare fraud
# ------------------------------------------------------------------------------

def make_training_sample(
    df: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """
    Construct a TRAINING sample by:
      - keeping ALL fraud rows
      - downsampling non-fraud rows to reach cfg.train_target_fraud_rate
        (or to max_nonfraud_train, whichever is smaller).

    This is applied ONLY on the training split; val/test keep natural prevalence.
    """
    label = cfg.label_col
    if label not in df.columns:
        raise KeyError(f"Label column '{label}' not found in training dataframe")

    df = df.copy()
    pos = df[df[label] == 1]
    neg = df[df[label] == 0]

    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        logger.warning(
            "make_training_sample: no positives or no negatives in train; "
            "skipping downsampling."
        )
        return df

    orig_rate = n_pos / (n_pos + n_neg + 1e-9)
    target_rate = cfg.train_target_fraud_rate
    if target_rate <= 0 or target_rate >= 0.5:
        logger.info(
            "train_target_fraud_rate=%.4f is out of (0, 0.5); skipping train downsampling",
            target_rate,
        )
        return df

    # Solve target_rate = n_pos / (n_pos + n_neg_sample)
    target_neg = int(n_pos * (1.0 - target_rate) / target_rate)
    if cfg.max_nonfraud_train is not None:
        target_neg = min(target_neg, cfg.max_nonfraud_train)

    target_neg = min(target_neg, n_neg)
    if target_neg <= 0:
        logger.warning("make_training_sample: computed target_neg <= 0; skipping")
        return df

    neg_sample = neg.sample(n=target_neg, random_state=cfg.random_state)
    df_sampled = pd.concat([pos, neg_sample], ignore_index=True)
    df_sampled = df_sampled.sample(frac=1.0, random_state=cfg.random_state).reset_index(
        drop=True
    )

    new_rate = len(pos) / (len(df_sampled) + 1e-9)
    logger.info(
        "Training sampling: original n=%d (fraud=%d, non-fraud=%d, rate=%.6f) -> "
        "sampled n=%d (fraud=%d, non-fraud=%d, rate=%.6f)",
        len(df),
        n_pos,
        n_neg,
        orig_rate,
        len(df_sampled),
        len(pos),
        len(neg_sample),
        new_rate,
    )

    return df_sampled


# ------------------------------------------------------------------------------
# Feature engineering: time, entity/graph-like, etc.
# ------------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add time-based and recency features extracted from parsed datetime columns."""
    df = df.copy()

    trx_dt = None
    if cfg.trx_date_col in df.columns:
        trx_dt = pd.to_datetime(df[cfg.trx_date_col], errors="coerce")
        df["TRX_DAYOFWEEK"] = trx_dt.dt.dayofweek
        df["TRX_DAY"] = trx_dt.dt.day
        df["TRX_MONTH"] = trx_dt.dt.month
        df["TRX_YEAR"] = trx_dt.dt.year

    if cfg.trx_hour_col in df.columns:
        hour = pd.to_numeric(df[cfg.trx_hour_col], errors="coerce")
        df["TRX_HOUR_CLEAN"] = hour
        night_mask = (hour >= 0) & ((hour <= 6) | (hour >= 23))
        df["TRX_IS_NIGHT"] = night_mask.fillna(False).astype(int)
        if "TRX_DAYOFWEEK" in df.columns:
            weekend_mask = df["TRX_DAYOFWEEK"].isin([5, 6])
            df["TRX_IS_WEEKEND"] = weekend_mask.fillna(False).astype(int)
        elif trx_dt is not None:
            weekend_mask = trx_dt.dt.dayofweek.isin([5, 6])
            df["TRX_IS_WEEKEND"] = weekend_mask.fillna(False).astype(int)
        else:
            df["TRX_IS_WEEKEND"] = 0

    if cfg.trx_amount_col in df.columns:
        amount = pd.to_numeric(df[cfg.trx_amount_col], errors="coerce")
        df["TRX_AMOUNT_LOG"] = np.log1p(amount.clip(lower=0))

    # Additional known dates (account opening, last update, etc.)
    additional_dates = (set(cfg.date_cols) | set(cfg.int_date_cols)) - {cfg.trx_date_col}
    for col in additional_dates:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_YEAR"] = parsed.dt.year
        df[f"{col}_MONTH"] = parsed.dt.month
        df[f"{col}_DAY"] = parsed.dt.day
        if trx_dt is not None:
            delta = (trx_dt - parsed).dt.days
            df[f"DAYS_SINCE_{col}"] = delta

    if "TRX_IS_WEEKEND" not in df.columns and trx_dt is not None:
        weekend_mask = trx_dt.dt.dayofweek.isin([5, 6])
        df["TRX_IS_WEEKEND"] = weekend_mask.fillna(False).astype(int)

    return df


def compute_entity_frequencies(
    df: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    """
    For each specified column, compute a simple frequency count and add as new feature.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            continue
        freq = df[col].value_counts(dropna=False)
        freq_map = freq.to_dict()
        df[f"{col}_FREQ"] = df[col].map(freq_map).astype("float32")

    return df


def compute_entity_fraud_stats(
    train: pd.DataFrame,
    cfg: Config,
    entity_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    For each entity column, compute fraud counts and rates from *training data only*.

    Returns a dict mapping entity column name -> stats dataframe with:
        [entity_col, entity_col + "_FRAUD_COUNT", "_TOTAL_COUNT", "_FRAUD_RATE"]
    """
    stats_dict: Dict[str, pd.DataFrame] = {}

    for col in entity_cols:
        if col not in train.columns:
            continue
        grouped = train.groupby(col)[cfg.label_col].agg(["sum", "count"]).reset_index()
        grouped.columns = [col, "FRAUD_COUNT", "TOTAL_COUNT"]
        grouped["FRAUD_RATE"] = grouped["FRAUD_COUNT"] / grouped["TOTAL_COUNT"].clip(
            lower=1
        )
        grouped.rename(
            columns={
                "FRAUD_COUNT": f"{col}_FRAUD_COUNT",
                "TOTAL_COUNT": f"{col}_TOTAL_COUNT",
                "FRAUD_RATE": f"{col}_FRAUD_RATE",
            },
            inplace=True,
        )
        stats_dict[col] = grouped

    return stats_dict


def apply_entity_fraud_stats(
    df: pd.DataFrame,
    stats_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Attach precomputed fraud stats to df for each entity column."""
    df = df.copy()
    for col, stats in stats_dict.items():
        if col not in df.columns:
            continue
        df = df.merge(stats, on=col, how="left")
        for suffix in ["_FRAUD_COUNT", "_TOTAL_COUNT", "_FRAUD_RATE"]:
            stat_col = col + suffix
            if stat_col in df.columns:
                df[stat_col] = df[stat_col].fillna(0.0)
    return df


def engineer_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform full feature engineering pipeline on train/val/test:

    - Time / amount features
    - Global entity frequencies
    - Entity fraud stats based on training data only
    """
    logger.info("[STEP] Feature engineering (time + entity/graph-like features)")

    # Basic time & amount features
    train_fe = add_time_features(train, cfg)
    val_fe = add_time_features(val, cfg)
    test_fe = add_time_features(test, cfg)

    # Entity frequency features
    entity_cols = [
        cfg.customer_col,
        cfg.account_col,
        cfg.dest_account_col,
        cfg.device_col,
        cfg.ip_col,
        cfg.phone_col,
        cfg.email_col,
    ]
    train_fe = compute_entity_frequencies(train_fe, entity_cols)
    val_fe = compute_entity_frequencies(val_fe, entity_cols)
    test_fe = compute_entity_frequencies(test_fe, entity_cols)

    # Entity fraud stats (train only -> then merged into val/test)
    stats_dict = compute_entity_fraud_stats(train_fe, cfg, entity_cols)
    train_fe = apply_entity_fraud_stats(train_fe, stats_dict)
    val_fe = apply_entity_fraud_stats(val_fe, stats_dict)
    test_fe = apply_entity_fraud_stats(test_fe, stats_dict)

    return train_fe, val_fe, test_fe


# ------------------------------------------------------------------------------
# Resampling on Encoded Numeric Data (unused helper)
# ------------------------------------------------------------------------------

def resample_train_val_combined_after_encoding(
    X_train,
    X_val,
    y_train,
    y_val,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    UNUSED helper kept for reference: Apply SMOTE to combined train+val AFTER
    feature encoding (on numeric data only).
    """
    original_train_size = len(X_train)
    original_val_size = len(X_val)
    total_size = original_train_size + original_val_size

    logger.info("[STEP] Applying SMOTE to combined train+val for extreme imbalance handling")
    logger.info(f"  Original train size: {original_train_size}, val size: {original_val_size}")
    logger.info(
        f"  Original fraud distribution: train={int(y_train.sum())}, "
        f"val={int(y_val.sum())}"
    )

    # Combine train + val (X and y already numeric from feature encoding)
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])

    # Calculate number of minority samples
    n_minority = int(y_combined.sum())
    logger.info(f"  Total minority (fraud) samples in combined: {n_minority}")

    # Apply SMOTE if available
    if SMOTETomek is None:
        logger.warning("SMOTE not available; returning original train/val without resampling")
        return X_train, X_val, y_train, y_val

    logger.info("  Applying SMOTE + TomekLinks to create synthetic fraud cases...")
    try:
        k_neighbors = max(1, min(5, n_minority - 1)) if n_minority > 1 else 1
        logger.info(f"    Using k_neighbors={k_neighbors} (adaptive for {n_minority} minority samples)")

        sampler = SMOTETomek(k_neighbors=k_neighbors, random_state=cfg.random_state)
        X_resampled, y_resampled = sampler.fit_resample(X_combined, y_combined)
    except Exception as e:
        logger.warning(f"  SMOTE+TomekLinks failed ({e}); attempting pure SMOTE without undersampling...")
        try:
            k_neighbors = max(1, min(5, n_minority - 1)) if n_minority > 1 else 1
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=cfg.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_combined, y_combined)
            logger.info(f"    Pure SMOTE succeeded with k_neighbors={k_neighbors}")
        except Exception as e2:
            logger.warning(f"  Pure SMOTE also failed ({e2}); returning original train/val")
            return X_train, X_val, y_train, y_val

    new_total_size = len(X_resampled)
    logger.info(
        f"  After SMOTE: {new_total_size} rows "
        f"(created {new_total_size - total_size} synthetic fraud cases)"
    )
    logger.info(
        "  Resampled fraud distribution: %d fraud, %d non-fraud",
        int(y_resampled.sum()),
        int((y_resampled == 0).sum()),
    )

    # Shuffle resampled data before re-splitting
    shuffle_idx = np.random.RandomState(cfg.random_state).permutation(new_total_size)
    X_resampled = X_resampled[shuffle_idx]
    y_resampled = y_resampled[shuffle_idx]
    logger.info("  Shuffled resampled data to distribute fraud cases evenly")

    # Re-split back into train/val using original proportions
    train_fraction = original_train_size / total_size
    split_point = int(train_fraction * new_total_size)

    X_train_resampled = X_resampled[:split_point]
    X_val_resampled = X_resampled[split_point:]
    y_train_resampled = y_resampled[:split_point]
    y_val_resampled = y_resampled[split_point:]

    logger.info(
        "  Re-split: train=%d (fraud=%d), val=%d (fraud=%d)",
        len(X_train_resampled),
        int(y_train_resampled.sum()),
        len(X_val_resampled),
        int(y_val_resampled.sum()),
    )

    return X_train_resampled, X_val_resampled, y_train_resampled, y_val_resampled


# ------------------------------------------------------------------------------
# Intelligent Cardinality-based Encoding
# ------------------------------------------------------------------------------

def analyze_column_cardinality(
    df: pd.DataFrame,
    categorical_cols: List[str],
    low_card_threshold: int = 50,
    high_card_threshold: int = 500,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Analyze actual cardinality of categorical columns and categorize them.

    Returns three lists:
    - low_card_ohe: columns with cardinality < low_card_threshold (use OHE)
    - med_card_target: columns with cardinality between thresholds (use target encoding)
    - high_card_target: columns with cardinality > high_card_threshold (use target encoding)
    """
    low_card_ohe = []
    med_card_target = []
    high_card_target = []

    cardinality_report = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue

        nunique = df[col].nunique()
        cardinality_report[col] = nunique

        if nunique < low_card_threshold:
            low_card_ohe.append(col)
        elif nunique < high_card_threshold:
            med_card_target.append(col)
        else:
            high_card_target.append(col)

    logger.info("[CARDINALITY ANALYSIS]")
    logger.info(f"  Low cardinality (OHE, <{low_card_threshold}): {len(low_card_ohe)} columns")
    if low_card_ohe:
        for col in low_card_ohe[:5]:
            logger.info(f"    - {col}: {cardinality_report[col]} unique values")
        if len(low_card_ohe) > 5:
            logger.info(f"    ... and {len(low_card_ohe) - 5} more")

    logger.info(
        "  Medium cardinality (Target Encode, %d-%d): %d columns",
        low_card_threshold,
        high_card_threshold,
        len(med_card_target),
    )
    if med_card_target:
        for col in med_card_target[:5]:
            logger.info(f"    - {col}: {cardinality_report[col]} unique values")
        if len(med_card_target) > 5:
            logger.info(f"    ... and {len(med_card_target) - 5} more")

    logger.info(
        "  High cardinality (Target Encode, >%d): %d columns",
        high_card_threshold,
        len(high_card_target),
    )
    if high_card_target:
        for col in high_card_target[:5]:
            logger.info(f"    - {col}: {cardinality_report[col]} unique values")
        if len(high_card_target) > 5:
            logger.info(f"    ... and {len(high_card_target) - 5} more")

    return low_card_ohe, med_card_target, high_card_target


def compute_target_encoding_stats(
    train: pd.DataFrame,
    cfg: Config,
    target_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Compute target encoding (fraud rate) for high-cardinality columns from training data only.

    Returns dict: column_name -> stats_df with [column_value, FRAUD_RATE]
    """
    stats_dict: Dict[str, pd.DataFrame] = {}

    for col in target_cols:
        if col not in train.columns:
            continue

        # Group by column and compute fraud rate
        grouped = train.groupby(col)[cfg.label_col].agg(["sum", "count"]).reset_index()
        grouped.columns = [col, "fraud_count", "total_count"]
        grouped["FRAUD_RATE"] = grouped["fraud_count"] / grouped["total_count"].clip(
            lower=1
        )

        # Keep only column and fraud rate
        grouped = grouped[[col, "FRAUD_RATE"]].copy()
        grouped.rename(columns={"FRAUD_RATE": f"{col}_TARGET_ENCODED"}, inplace=True)

        stats_dict[col] = grouped

    return stats_dict


def apply_target_encoding(
    df: pd.DataFrame,
    stats_dict: Dict[str, pd.DataFrame],
    default_rate: float = 0.0,
) -> pd.DataFrame:
    """Apply precomputed target encoding (fraud rates) to dataframe."""
    df = df.copy()

    for col, stats in stats_dict.items():
        if col not in df.columns:
            continue

        # Merge stats and fill unknown values with default rate
        encoded_col = f"{col}_TARGET_ENCODED"
        df = df.merge(stats, on=col, how="left")
        df[encoded_col] = df[encoded_col].fillna(default_rate).astype("float32")

        # Drop original column since it's now encoded
        df = df.drop(columns=[col], errors="ignore")

    return df


# ------------------------------------------------------------------------------
# Preprocessing & feature matrix construction
# ------------------------------------------------------------------------------

def build_feature_matrix(
    train_fe: pd.DataFrame,
    val_fe: pd.DataFrame,
    test_fe: pd.DataFrame,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline]:
    """
    Build X/y for train/val/test using a scikit-learn ColumnTransformer pipeline.

    - Automatically detects numeric vs categorical columns
    - Imputes missing values
    - One-hot encodes categoricals (with handle_unknown='ignore')
    - Leaves some high-cardinality ID columns out of OHE (they are represented by numeric freq/fraud stats)
    """
    logger.info("[STEP] Preparing ML feature matrix and preprocessing pipeline")

    # Columns to exclude completely from features
    exclude_cols = {
        cfg.label_col,
        cfg.trx_id_col,
    }

    entity_cols = set(cfg.entity_columns())
    id_like_cols = set(cfg.id_like_cols)
    parsed_date_cols = set(cfg.date_cols) | set(cfg.int_date_cols)
    drop_cols = entity_cols | id_like_cols | parsed_date_cols

    # Identify candidate feature columns
    all_cols = list(train_fe.columns)
    feature_cols = [c for c in all_cols if c not in exclude_cols and c not in drop_cols]
    drop_count = len(drop_cols & set(all_cols))
    if drop_count:
        logger.info(
            "Excluding %d raw ID/entity/date columns from model features",
            drop_count,
        )

    # Identify numeric vs categorical
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_fe[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Analyze cardinality of all categorical columns
    low_card_ohe, med_card_target, high_card_target = analyze_column_cardinality(
        train_fe, categorical_cols, low_card_threshold=50, high_card_threshold=500
    )

    whitelist = set(cfg.target_encode_whitelist)
    candidate_target_cols = med_card_target + high_card_target
    target_encode_cols = [
        c for c in candidate_target_cols
        if c in whitelist and c not in entity_cols
    ]
    dropped_high_card = sorted(set(candidate_target_cols) - set(target_encode_cols))
    if dropped_high_card:
        logger.info(
            "Skipping target encoding for %d high-cardinality columns "
            "(ID-like or not whitelisted)",
            len(dropped_high_card),
        )
        for col in dropped_high_card[:5]:
            logger.info("    - %s", col)
    feature_cols = [c for c in feature_cols if c not in dropped_high_card]

    logger.info(f"  Numeric feature columns: {len(numeric_cols)}")
    logger.info(f"  Low cardinality categorical (OHE): {len(low_card_ohe)}")
    logger.info(f"  High cardinality categorical (Target Encoding): {len(target_encode_cols)}")

    # Compute target encoding statistics from training data
    if target_encode_cols:
        logger.info("Computing target encoding (fraud rates) for high-cardinality columns...")
        target_stats = compute_target_encoding_stats(train_fe, cfg, target_encode_cols)

        # Apply target encoding to all datasets
        train_fe = apply_target_encoding(train_fe, target_stats, default_rate=0.0)
        val_fe = apply_target_encoding(val_fe, target_stats, default_rate=0.0)
        test_fe = apply_target_encoding(test_fe, target_stats, default_rate=0.0)

        logger.info(f"Applied target encoding to {len(target_encode_cols)} columns")

    # Rebuild feature columns list after target encoding
    feature_cols = [
        c for c in train_fe.columns
        if c not in exclude_cols and c not in target_encode_cols and c not in drop_cols
    ]

    # Re-identify numeric vs categorical after target encoding
    numeric_cols = []
    categorical_cols_final: List[str] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_fe[col]):
            numeric_cols.append(col)
        else:
            categorical_cols_final.append(col)

    # Keep only low-cardinality categorical columns (high-cardinality are already target-encoded)
    categorical_cols_final = [c for c in categorical_cols_final if c in low_card_ohe]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    ohe_kwargs = {"handle_unknown": "ignore"}
    ohe_params = inspect.signature(OneHotEncoder.__init__).parameters
    if "min_frequency" in ohe_params:
        ohe_kwargs["min_frequency"] = 10
    if "sparse_output" in ohe_params:
        ohe_kwargs["sparse_output"] = False
    elif "sparse" in ohe_params:
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
            ("cat", categorical_transformer, categorical_cols_final),
        ]
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
        ]
    )

    # Fit preprocessor only on train
    X_train = train_fe[feature_cols]
    y_train = train_fe[cfg.label_col].values.astype(int)

    X_val = val_fe[feature_cols]
    y_val = val_fe[cfg.label_col].values.astype(int)

    X_test = test_fe[feature_cols]
    y_test = test_fe[cfg.label_col].values.astype(int)

    logger.info("Fitting preprocessing pipeline on train data...")
    model_pipeline.fit(X_train)

    logger.info("Transforming train/val/test with preprocessing pipeline...")
    X_train_t = model_pipeline.transform(X_train)
    X_val_t = model_pipeline.transform(X_val)
    X_test_t = model_pipeline.transform(X_test)

    logger.info(
        "Feature matrix shapes: X_train=%s, X_val=%s, X_test=%s",
        X_train_t.shape,
        X_val_t.shape,
        X_test_t.shape,
    )

    return (
        X_train_t,
        y_train,
        X_val_t,
        y_val,
        X_test_t,
        y_test,
        model_pipeline,
    )


# ------------------------------------------------------------------------------
# Imbalance handling (synthetic oversampling)
# ------------------------------------------------------------------------------

def resample_training_data(
    X_train,
    y_train,
    cfg: Config,
):
    """
    Apply advanced imbalance handling with adaptive k_neighbors for small minority classes.

    Preferred: SMOTE-Tomek with BorderlineSMOTE (focusing on decision boundary).

    NOTE: This runs on top of the PRIOR downsampling of non-fraud in train.
    """
    if not cfg.use_smote_tomek:
        logger.info("SMOTE-Tomek disabled via config; using original training data")
        return X_train, y_train

    n_minority = int(y_train.sum())
    logger.info("Training data: %d total, %d fraud (minority)", len(X_train), n_minority)

    # Adaptive k_neighbors for extremely small minority classes
    k_neighbors = max(1, min(5, n_minority - 1)) if n_minority > 1 else 1
    logger.info("Using adaptive k_neighbors=%d (for %d minority samples)", k_neighbors, n_minority)

    if n_minority < 2:
        logger.warning("Very few fraud cases (%d); cannot apply SMOTE", n_minority)
        return X_train, y_train

    smote_kind = (cfg.smote_kind or "regular").lower()
    has_borderline = BorderlineSMOTE is not None

    def _log_and_return(X, y, desc: str):
        logger.info(
            "%s output: %s, fraud=%d, non-fraud=%d",
            desc,
            X.shape,
            int(y.sum()),
            int((y == 0).sum()),
        )
        return X, y

    def _apply_borderline_then_tomek():
        try:
            sampler = BorderlineSMOTE(
                k_neighbors=k_neighbors,
                random_state=cfg.random_state,
                kind="borderline-1",
            )
            X_smote, y_smote = sampler.fit_resample(X_train, y_train)
            fraud_after_smote = int(y_smote.sum())

            if TomekLinks is not None:
                tomek = TomekLinks()
                X_clean, y_clean = tomek.fit_resample(X_smote, y_smote)
                fraud_after_tomek = int(y_clean.sum())

                if fraud_after_tomek > n_minority:
                    logger.info(
                        "✓ BorderlineSMOTE created %d fraud, TomekLinks refined to %d",
                        fraud_after_smote,
                        fraud_after_tomek,
                    )
                    return _log_and_return(X_clean, y_clean, "BorderlineSMOTE + TomekLinks")
                else:
                    logger.warning(
                        "TomekLinks removed too many samples (%d fraud); skipping",
                        fraud_after_tomek,
                    )
                    return None

            if fraud_after_smote > n_minority:
                logger.info(
                    "✓ BorderlineSMOTE created synthetic fraud: %d → %d",
                    n_minority,
                    fraud_after_smote,
                )
                return _log_and_return(X_smote, y_smote, "BorderlineSMOTE")
            else:
                logger.warning(
                    "BorderlineSMOTE output unchanged (%d fraud); need fallback",
                    fraud_after_smote,
                )
                return None

        except Exception as e:
            logger.warning("BorderlineSMOTE failed (%s); falling back to pure SMOTE", e)
            return None

    if smote_kind == "borderline" and has_borderline:
        logger.info("Attempting BorderlineSMOTE-focused resampling...")
        result = _apply_borderline_then_tomek()
        if result is not None:
            return result
        logger.info("Fallback: Using pure SMOTE with adaptive k_neighbors")

    if SMOTETomek is None:
        logger.info("SMOTETomek not available; trying pure SMOTE")
        if SMOTE is None:
            logger.warning("SMOTE not available; using original training data")
            return X_train, y_train

    # Fallback: Try pure SMOTE with k_neighbors=1
    try:
        smote_k = max(1, min(k_neighbors, 1))
        logger.info("Applying pure SMOTE with k_neighbors=%d...", smote_k)
        smote_estimator = SMOTE(k_neighbors=smote_k, random_state=cfg.random_state)
        X_res, y_res = smote_estimator.fit_resample(X_train, y_train)
        fraud_after = int(y_res.sum())
        if fraud_after > n_minority:
            logger.info(
                "✓ SMOTE successfully created synthetic fraud: %d → %d",
                n_minority,
                fraud_after,
            )
            return _log_and_return(X_res, y_res, "SMOTE")
        else:
            logger.warning(
                "SMOTE output unchanged (%d fraud); trying RandomOverSampler",
                fraud_after,
            )
    except Exception as e:
        logger.warning("Pure SMOTE failed (%s); trying RandomOverSampler as fallback", e)

    # Last resort: Simple random oversampling (duplicates minority samples)
    try:
        from imblearn.over_sampling import RandomOverSampler
        logger.info("Applying RandomOverSampler (random duplication of fraud cases)...")
        sampler = RandomOverSampler(random_state=cfg.random_state)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        fraud_after = int(y_res.sum())
        logger.info(
            "✓ RandomOverSampler created duplicate fraud cases: %d → %d",
            n_minority,
            fraud_after,
        )
        return _log_and_return(X_res, y_res, "RandomOverSampler")
    except Exception as e:
        logger.warning("RandomOverSampler failed (%s); returning original training data", e)
        logger.warning(
            "WARNING: Model will train on original highly imbalanced data "
            "(no synthetic oversampling)!"
        )
        return X_train, y_train


# ------------------------------------------------------------------------------
# Feature selection and importance ranking
# ------------------------------------------------------------------------------

def select_important_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    threshold: float = "median",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Select important features using SelectFromModel with a RandomForest estimator.
    """
    logger.info("[STEP] Feature Selection based on RandomForest importance")

    rf_selector = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        random_state=42,
        n_jobs=-1,
    )
    rf_selector.fit(X_train, y_train)

    selector = SelectFromModel(
        rf_selector,
        prefit=True,
        threshold=threshold,
    )

    selected_mask = selector.get_support()
    X_selected = X_train[:, selected_mask]
    selected_names = [feature_names[i] for i, sel in enumerate(selected_mask) if sel]

    logger.info(
        "  Selected %d / %d features (reduction: %.1f%%)",
        len(selected_names),
        len(feature_names),
        100 * (1 - len(selected_names) / len(feature_names)),
    )
    logger.info("  Kept features: %s", selected_names[:10])
    if len(selected_names) > 10:
        logger.info("  ... and %d more", len(selected_names) - 10)

    return X_selected, selected_mask, selected_names


def compute_feature_importance(
    model,
    feature_names: List[str],
    importance_type: str = "weight",
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Compute and rank feature importance from a trained model.
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
        return pd.DataFrame()

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    logger.info("  Top %d features:", top_k)
    for _, row in importance_df.head(top_k).iterrows():
        logger.info("    %s: %.6f", row["feature"], row["importance"])

    return importance_df


def explain_model_with_shap(
    model,
    X_sample: np.ndarray,
    X_background: np.ndarray = None,
    model_name: str = "Model",
    max_samples: int = 100,
) -> Optional[Any]:
    """
    Generate SHAP explanations for model predictions.
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not installed; skipping model explainability analysis")
        return None

    logger.info("[STEP] Computing SHAP explanations for %s", model_name)

    # Unwrap calibrated models to the underlying tree model
    inner_model = getattr(model, "base_estimator", model)

    if len(X_sample) > max_samples:
        indices = np.random.choice(len(X_sample), max_samples, replace=False)
        X_explain = X_sample[indices]
        logger.info(
            "  Using %d samples for explanation (out of %d)", max_samples, len(X_sample)
        )
    else:
        X_explain = X_sample

    if X_background is None:
        n_background = min(100, len(X_sample) // 2)
        bg_indices = np.random.choice(len(X_sample), n_background, replace=False)
        X_background = X_sample[bg_indices]
        logger.info("  Using %d background samples for SHAP", n_background)

    try:
        if hasattr(inner_model, "predict_proba"):
            explainer = shap.TreeExplainer(inner_model)
            shap_values = explainer.shap_values(X_explain)
            logger.info("  SHAP TreeExplainer created; computed %d samples", len(shap_values))
        else:
            explainer = shap.KernelExplainer(
                inner_model.predict_proba
                if hasattr(inner_model, "predict_proba")
                else inner_model.predict,
                X_background,
            )
            shap_values = explainer.shap_values(X_explain)
            logger.info("  SHAP KernelExplainer created; computed %d samples", len(shap_values))

        return explainer, shap_values
    except Exception as e:
        logger.warning("SHAP explanation failed (%s); skipping", e)
        return None


# ------------------------------------------------------------------------------
# Threshold tuning and evaluation utilities
# ------------------------------------------------------------------------------

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
            best_score = score
            best_t = t
            best_metrics = {
                "precision": float(p),
                "recall": float(r),
                "f_beta": float((1 + beta**2) * (p * r) / (beta**2 * p + r + eps)),
            }

    return float(best_t), float(best_score), best_metrics


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

    This is useful when business has a fixed analyst capacity per day.
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

    f_beta = (1 + 1.0**2) * precision * recall / (1.0**2 * precision + recall + eps) if (precision + recall) > 0 else 0.0  # F1

    alert_rate = pred.mean()

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f_beta": float(f_beta),
        "alert_rate": float(alert_rate),
        "k": int(k),
        "threshold": t,
    }
    return t, metrics


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
    prec_k = y_top.mean()
    recall_k = y_top.sum() / max(1, y_true.sum())
    return k, float(prec_k), float(recall_k)


# ------------------------------------------------------------------------------
# Probability calibration helper
# ------------------------------------------------------------------------------

def maybe_calibrate_model(
    model,
    X_calib,
    y_calib,
    cfg: Config,
    model_name: str,
):
    """
    Optionally wrap a classifier with CalibratedClassifierCV using the validation set
    as calibration data (cv='prefit').

    This adjusts probability outputs so they better match real-world prevalence
    while keeping the underlying classifier fixed.
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


# ------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------

def evaluate_model(
    name: str,
    model,
    X_val,
    y_val,
    X_test,
    y_test,
    cfg: Config,
    feature_names: List[str] = None,
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
    logger.info(
        "[%s] Val proba percentiles P50..P99.9: %s",
        name,
        val_percentiles.tolist(),
    )
    logger.info(
        "[%s] Test proba percentiles P50..P99.9: %s",
        name,
        test_percentiles.tolist(),
    )

    # Threshold strategy
    threshold_strategy = getattr(cfg, "threshold_strategy", "metric").lower()
    threshold_metric = getattr(cfg, "threshold_metric", "f_beta")

    if threshold_strategy == "cost":
        best_t, best_cost, threshold_metrics = find_cost_optimal_threshold(
            y_val,
            val_proba,
            fn_cost=getattr(cfg, "fn_cost", 10.0),
            fp_cost=getattr(cfg, "fp_cost", 1.0),
        )
        best_score = -best_cost  # higher is better convention
        logger.info(
            "[%s] Best threshold (cost-based on val) = %.4f, cost=%.6f, "
            "FPR=%.4f, TPR=%.4f",
            name,
            best_t,
            best_cost,
            threshold_metrics.get("fpr", 0.0),
            threshold_metrics.get("tpr", 0.0),
        )
    elif threshold_strategy == "volume" and getattr(cfg, "target_alert_rate", None) is not None:
        best_t, threshold_metrics = find_threshold_for_target_alert_rate(
            y_val,
            val_proba,
            cfg.target_alert_rate,
        )
        best_score = threshold_metrics.get("f_beta", 0.0)
        logger.info(
            "[%s] Best threshold (volume-based on val) = %.4f, alert_rate=%.4f, "
            "precision=%.4f, recall=%.4f",
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

    # Standard metrics at best threshold
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

    # Confusion matrix analysis
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    logger.info(
        "[%s] Confusion Matrix: TP=%d, FP=%d, FN=%d, TN=%d | Specificity=%.4f",
        name,
        tp,
        fp,
        fn,
        tn,
        specificity,
    )

    if cfg.verbose_reports:
        logger.info("[%s] Classification report (Test):", name)
        logger.info("\n%s", classification_report(y_test, test_pred, digits=3))

    # Precision@K metrics
    k, prec_k, rec_k = precision_recall_at_k(y_test, test_proba, frac=cfg.top_k_frac)
    logger.info(
        "[%s] Precision@top-%.1f%% (K=%d): precision=%.4f, recall=%.4f",
        name,
        cfg.top_k_frac * 100,
        k,
        prec_k,
        rec_k,
    )

    # Feature importance
    importance_df = None
    if feature_names:
        importance_df = compute_feature_importance(model, feature_names)

    # SHAP explainability
    shap_result = None
    if feature_names:
        shap_result = explain_model_with_shap(model, X_test[:100], model_name=name)

    metrics = {
        "name": name,
        "val_auc": val_auc,
        "val_ap": val_ap,
        "val_f1": val_f1,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_specificity": specificity,
        "best_threshold": best_t,
        "best_threshold_metric_val": best_score,
        "precision_at_k": prec_k,
        "recall_at_k": rec_k,
        "k": k,
        "confusion_matrix": {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)},
        "importance_df": importance_df,
        "shap_result": shap_result,
        "threshold_strategy": threshold_strategy,
        "threshold_metrics_val": threshold_metrics,
    }

    return metrics


# ------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------

def train_random_forest(
    X_train,
    y_train,
    X_val,
    y_val,
    cfg: Config,
):
    """
    Train a regularized RandomForestClassifier with class weighting and warm-starting.
    """
    logger.info("=" * 70)
    logger.info("Training RandomForestClassifier (regularized + class weighting)")

    total_trees = max(1, int(cfg.rf_n_estimators))
    chunk = max(1, min(int(cfg.rf_warm_start_chunk), total_trees))
    warm = total_trees > chunk

    # Compute class weights manually for informational logging
    n_fraud = (y_train == 1).sum()
    n_non_fraud = (y_train == 0).sum()
    fraud_weight = n_non_fraud / (n_fraud + 1e-8) if n_fraud > 0 else 1.0
    logger.info("  Class weights: fraud=%.2f, non-fraud=1.0 (auto-balanced)", fraud_weight)

    rf = RandomForestClassifier(
        n_estimators=chunk if warm else total_trees,
        max_depth=10,
        min_samples_split=50,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=cfg.rf_n_jobs,
        class_weight="balanced_subsample",
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
            val_proba = rf.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            logger.info(
                "  -> RandomForest progress: %d/%d trees "
                "(val AUC=%.4f, %.1fs)",
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


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config,
):
    """
    Train an XGBoost binary classifier with a sane default configuration,
    optional class weighting, and validation-based early stopping.
    """
    if XGBClassifier is None:
        logger.warning("XGBoost not installed; skipping XGBoost model.")
        return None

    logger.info("=" * 70)
    logger.info("Training XGBoostClassifier with optional class weighting (no grid search)")

    # Compute class imbalance statistics on TRAIN SAMPLE (after downsampling)
    n_fraud = (y_train == 1).sum()
    n_non_fraud = (y_train == 0).sum()
    raw_ratio = float(n_non_fraud) / float(n_fraud + 1e-8) if n_fraud > 0 else 1.0
    logger.info("  Observed class imbalance ratio (non-fraud / fraud): %.2f", raw_ratio)

    if getattr(cfg, "xgb_use_class_weight", False):
        scale_pos_weight = min(raw_ratio, getattr(cfg, "xgb_scale_pos_clip", raw_ratio))
        logger.info("  Using scale_pos_weight=%.2f", scale_pos_weight)
    else:
        scale_pos_weight = 1.0
        logger.info("  Not using class weighting; scale_pos_weight=%.2f", scale_pos_weight)

    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=cfg.random_state,
    )

    logger.info("Training XGBoost with early stopping on validation set (50 rounds)")
    try:
        xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False,
        )
        if hasattr(xgb, "best_iteration") and xgb.best_iteration is not None:
            logger.info("XGBoost best_iteration=%s", xgb.best_iteration)
    except Exception as e:
        logger.warning("Early stopping API failed (%s); training XGBoost without it", e)
        xgb.fit(X_train, y_train)

    return xgb


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------

def run_pipeline(cfg: Config) -> Dict[str, Dict[str, Any]]:
    """
    Orchestrate the full pipeline end-to-end.

    Returns a dict: model_name -> metrics dict.
    """
    logger.info("=" * 70)
    logger.info("TECHM FRAUD DETECTION PIPELINE - GOD TIER LOCAL VERSION")
    logger.info("=" * 70)
    logger.info("Config:\n%s", json.dumps({k: str(v) for k, v in asdict(cfg).items()}, indent=2))

    # Ensure data dir exists
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load JSONL sample data
    df_trx_raw, df_ecm_raw = load_sample_jsonl(cfg)

    # 2. Normalize types
    df_trx_clean, df_ecm_clean = clean_and_normalize_raw(df_trx_raw, df_ecm_raw, cfg)

    # 3. Use clean sample data (no intentional corruption)
    logger.info("[STEP] Using clean sample data (skipped intentional data corruption)")

    # 4. Merge and label
    merged = merge_trx_and_ecm(df_trx_clean, df_ecm_clean, cfg)

    # 5. Split train/val/test (VAL/TEST remain at real-world prevalence)
    train_df, val_df, test_df = split_train_val_test(merged, cfg)

    # 6. Feature engineering (stats computed from FULL train_df)
    train_fe_full, val_fe, test_fe = engineer_features(train_df, val_df, test_df, cfg)

    # 6b. For model training, build a TRAIN SAMPLE with higher fraud rate
    train_fe = make_training_sample(train_fe_full, cfg)

    # 7. Build feature matrices (categorical → numeric encoding: OHE + target encoding)
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocess_pipeline,
    ) = build_feature_matrix(train_fe, val_fe, test_fe, cfg)

    # 8. Imbalance handling:
    X_train_rf, y_train_rf = resample_training_data(X_train, y_train, cfg)
    X_train_boost, y_train_boost = X_train, y_train

    # Extract feature names for importance analysis and explainability
    preprocessor = preprocess_pipeline.named_steps["preprocessor"]
    feature_names: List[str] = []
    if hasattr(preprocessor, "transformers_"):
        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                feature_names.extend(columns)
            elif name == "cat":
                if hasattr(transformer.named_steps.get("ohe"), "get_feature_names_out"):
                    try:
                        ohe_names = transformer.named_steps["ohe"].get_feature_names_out(
                            columns
                        )
                        feature_names.extend(ohe_names)
                    except Exception:
                        feature_names.extend(columns)
                else:
                    feature_names.extend(columns)

    logger.info("[STEP] Final feature matrix has %d features", len(feature_names))

    results: Dict[str, Dict[str, Any]] = {}

    # 9. Train & evaluate RandomForest
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

    # 10. Train & evaluate XGBoost
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

    # 11. Summary and best model selection
    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETE!")
    logger.info("=" * 70)
    logger.info("Models trained: %s", list(results.keys()))

    # Decide what metric to use for "best model"
    ranking_metric = "test_auc"  # you can switch to "precision_at_k" or another key in metrics

    if results:
        best_model_name = max(
            results.keys(),
            key=lambda name: results[name].get(ranking_metric, 0.0),
        )
        best_metrics = results[best_model_name]
        logger.info("\n[BEST MODEL] Summary:")
        logger.info("  Model: %s", best_model_name)
        logger.info("  Ranking metric (%s): %.4f", ranking_metric, best_metrics.get(ranking_metric, 0))
        logger.info("  Test AUC: %.4f", best_metrics.get("test_auc", 0))
        logger.info("  Test PR-AUC: %.4f", best_metrics.get("test_ap", 0))
        logger.info("  Test F1: %.4f", best_metrics.get("test_f1", 0))
        logger.info("  Test Precision: %.4f", best_metrics.get("test_precision", 0))
        logger.info("  Test Recall: %.4f", best_metrics.get("test_recall", 0))
        logger.info("  Optimal Threshold: %.4f", best_metrics.get("best_threshold", 0))

    return results


def main() -> None:
    seed_everything(42)
    cfg = Config()
    cfg = apply_env_overrides(cfg)
    run_pipeline(cfg)


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)
