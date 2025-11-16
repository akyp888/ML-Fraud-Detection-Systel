#!/usr/bin/env python3
"""
preprocess.py
==============

All non-model-specific parts of the TechM fraud detection pipeline:

- Global Config dataclass
- Logging setup and seeding
- Data loading / cleaning / merging
- Train / validation / test splitting
- Feature engineering (time/entity features + target encoding)
- Building numeric feature matrices
- Optional SMOTE / resampling on encoded train data
"""

import os
import sys
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Optional imbalance handling
try:
    from imblearn.combine import SMOTETomek  # type: ignore
    from imblearn.over_sampling import BorderlineSMOTE, SMOTE  # type: ignore
    from imblearn.under_sampling import TomekLinks  # type: ignore
except Exception:
    SMOTETomek = None
    BorderlineSMOTE = None
    SMOTE = None
    TomekLinks = None

# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger("techm-fraud-god-tier")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(_handler)


# ------------------------------------------------------------------------------
# Config dataclass (pipeline-wide configuration)
# ------------------------------------------------------------------------------

@dataclass
class Config:
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "local_data/sample"
    sample_dir: Path = Path(__file__).resolve().parent.parent / "local_data/sample"
    transactions_jsonl: str = "transactions.jsonl"
    ecm_jsonl: str = "ecm.jsonl"

    # Columns (schema)
    label_col: str = "label"
    ecm_label_col: str = "RESULT_TYPE_CD"
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

    # Threshold tuning / business logic (used in cli.evaluate_model)
    f_beta: float = 2.0
    top_k_frac: float = 0.01
    threshold_metric: str = "f_beta"
    threshold_strategy: str = "metric"  # "metric", "cost", or "alert_rate"
    fn_cost: float = 10.0
    fp_cost: float = 1.0
    target_alert_rate: Optional[float] = 0.01  # e.g. 0.005 for 0.5% of tx flagged

    # Probability calibration
    use_probability_calibration: bool = False
    calibration_method: str = "sigmoid"  # "sigmoid" or "isotonic"

    # Real world prevalence + training sampling for ultra-rare fraud
    real_fraud_rate: Optional[float] = None
    train_target_fraud_rate: float = 0.02  # 2% fraud in training sample
    max_nonfraud_train: Optional[int] = None

    # Imbalance handling on encoded train data
    use_smote_tomek: bool = False
    smote_kind: str = "borderline"  # or "regular"

    # Misc
    verbose_reports: bool = True

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
    We keep this intentionally small; most config is edited directly in this file.
    """
    env_map: Dict[str, Tuple[str, Any]] = {
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


# ------------------------------------------------------------------------------
# Data loading and merging
# ------------------------------------------------------------------------------

def load_sample_jsonl(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load transactions.jsonl and ecm.jsonl from cfg.sample_dir.
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
    """
    df_trx = df_trx.copy()
    df_ecm = df_ecm.copy()

    if cfg.trx_id_col not in df_trx.columns:
        raise KeyError(f"Transaction ID column {cfg.trx_id_col} not found in trx data")
    if cfg.trx_id_col not in df_ecm.columns:
        raise KeyError(f"Transaction ID column {cfg.trx_id_col} not found in ECM data")

    # Deduplicate ECM: keep latest per transaction if closing date present
    if "CLOSING_DATE" in df_ecm.columns:
        df_ecm = df_ecm.sort_values(["CORRELATIVE_NO", "CLOSING_DATE"]).drop_duplicates(
            subset=[cfg.trx_id_col], keep="last"
        )
    else:
        df_ecm = df_ecm.drop_duplicates(subset=[cfg.trx_id_col], keep="last")

    merged = df_trx.merge(df_ecm, on=cfg.trx_id_col, how="left", suffixes=("", "_ECM"))
    logger.info(f"Merged shape: {merged.shape[0]} rows, {merged.shape[1]} columns")

    # Create label from FRAUD_IND (if available) or fall back to ECM label column
    if "FRAUD_IND" in merged.columns:
        label_series = merged["FRAUD_IND"].astype(str).str.strip()
        merged[cfg.label_col] = np.where(
            label_series.isin(["1", "true", "True", "TRUE"]), 1, 0
        )
    elif cfg.ecm_label_col in merged.columns:
        label_series = merged[cfg.ecm_label_col].astype(str).str.upper().str.strip()
        is_fraud = label_series.str.contains("FRAUD")
        is_genuine = label_series.str.contains("GENUINE") | label_series.str.contains("LEGIT")
        merged[cfg.label_col] = np.where(is_fraud, 1, 0)
    else:
        raise KeyError(f"Neither FRAUD_IND nor {cfg.ecm_label_col} found in merged data")

    logger.info(
        "Label distribution after merge: non-fraud=%d, fraud=%d",
        int((merged[cfg.label_col] == 0).sum()),
        int((merged[cfg.label_col] == 1).sum()),
    )

    # Drop outcome/ECM fields that would leak the label
    leak_cols = [
        "FRAUD_IND",
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
    """
    df = df.copy()
    y = df[cfg.label_col].values
    total_fraud = int(y.sum())

    if cfg.trx_date_col in df.columns:
        logger.info("Using stratified temporal split by TRX_DATE (preserves fraud ratio)")
        df = df.sort_values(cfg.trx_date_col)

        fraud_mask = df[cfg.label_col] == 1
        fraud_df = df[fraud_mask].sort_values(cfg.trx_date_col)
        non_fraud_df = df[~fraud_mask].sort_values(cfg.trx_date_col)

        n_fraud = len(fraud_df)
        n_non_fraud = len(non_fraud_df)

        n_train_fraud = int((1.0 - cfg.val_size - cfg.test_size) * n_fraud)
        n_val_fraud = int(cfg.val_size * n_fraud)

        n_train_non_fraud = int((1.0 - cfg.val_size - cfg.test_size) * n_non_fraud)
        n_val_non_fraud = int(cfg.val_size * n_non_fraud)

        train_fraud = fraud_df.iloc[:n_train_fraud]
        val_fraud = fraud_df.iloc[n_train_fraud: n_train_fraud + n_val_fraud]
        test_fraud = fraud_df.iloc[n_train_fraud + n_val_fraud:]

        train_non_fraud = non_fraud_df.iloc[:n_train_non_fraud]
        val_non_fraud = non_fraud_df.iloc[n_train_non_fraud: n_train_non_fraud + n_val_non_fraud]
        test_non_fraud = non_fraud_df.iloc[n_train_non_fraud + n_val_non_fraud:]

        train = pd.concat([train_fraud, train_non_fraud], ignore_index=True)
        val = pd.concat([val_fraud, val_non_fraud], ignore_index=True)
        test = pd.concat([test_fraud, test_non_fraud], ignore_index=True)

        logger.info(
            "  Stratified temporal split: fraud split as %d/%d/%d",
            n_train_fraud,
            n_val_fraud,
            len(test_fraud),
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
        "Split sizes: train=%d, val=%d, test=%d (fraud in train=%d, fraud in val=%d, fraud in test=%d)",
        len(train),
        len(val),
        len(test),
        train_fraud,
        val_fraud,
        test_fraud,
    )

    if total_fraud > 0 and (train_fraud == 0 or val_fraud == 0 or test_fraud == 0):
        logger.warning(
            "Imbalanced fraud distribution: train=%d, val=%d, test=%d",
            train_fraud,
            val_fraud,
            test_fraud,
        )
        logger.warning("Rebalancing splits to ensure each has at least 1 fraud case...")

        all_data = pd.concat([train, val, test], ignore_index=True)
        fraud_cases = all_data[all_data[cfg.label_col] == 1]
        non_fraud_cases = all_data[all_data[cfg.label_col] == 0]

        min_fraud_per_split = 1
        required_fraud = 3
        available_fraud = len(fraud_cases)

        if available_fraud >= required_fraud:
            fraud_train_count = max(min_fraud_per_split, available_fraud // 3)
            fraud_val_count = max(
                min_fraud_per_split, (available_fraud - fraud_train_count) // 2
            )
            fraud_test_count = available_fraud - fraud_train_count - fraud_val_count

            rng = np.random.RandomState(cfg.random_state)
            fraud_indices = rng.permutation(available_fraud)
            fraud_train_idx = fraud_indices[:fraud_train_count]
            fraud_val_idx = fraud_indices[fraud_train_count: fraud_train_count + fraud_val_count]
            fraud_test_idx = fraud_indices[fraud_train_count + fraud_val_count:]

            available_non_fraud = len(non_fraud_cases)
            non_fraud_train_count = int(available_non_fraud * 0.70)
            non_fraud_val_count = int(available_non_fraud * 0.15)
            non_fraud_test_count = available_non_fraud - non_fraud_train_count - non_fraud_val_count

            non_fraud_indices = rng.permutation(available_non_fraud)
            non_fraud_train_idx = non_fraud_indices[:non_fraud_train_count]
            non_fraud_val_idx = non_fraud_indices[
                non_fraud_train_count: non_fraud_train_count + non_fraud_val_count
            ]
            non_fraud_test_idx = non_fraud_indices[
                non_fraud_train_count + non_fraud_val_count:
            ]

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
                "Rebalanced split sizes: train=%d, val=%d, test=%d (fraud train=%d, val=%d, test=%d)",
                len(train),
                len(val),
                len(test),
                int(train[cfg.label_col].sum()),
                int(val[cfg.label_col].sum()),
                int(test[cfg.label_col].sum()),
            )
        else:
            logger.warning(
                "Very few fraud cases (%d); cannot guarantee all splits have fraud",
                available_fraud,
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
      - downsampling non-fraud rows to reach cfg.train_target_fraud_rate.
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
            "make_training_sample: no positives or no negatives in train; skipping downsampling."
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

    target_neg = int(n_pos * (1.0 - target_rate) / target_rate)
    if cfg.max_nonfraud_train is not None:
        target_neg = min(target_neg, cfg.max_nonfraud_train)

    target_neg = min(target_neg, n_neg)
    if target_neg <= 0:
        logger.warning("make_training_sample: computed target_neg <= 0; skipping")
        return df

    neg_sample = neg.sample(n=target_neg, random_state=cfg.random_state)
    df_sampled = pd.concat([pos, neg_sample], ignore_index=True)
    df_sampled = df_sampled.sample(frac=1.0, random_state=cfg.random_state).reset_index(drop=True)

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
    """
    stats_dict: Dict[str, pd.DataFrame] = {}
    for col in entity_cols:
        if col not in train.columns:
            continue
        grouped = train.groupby(col)[cfg.label_col].agg(["sum", "count"]).reset_index()
        grouped.columns = [col, "FRAUD_COUNT", "TOTAL_COUNT"]
        grouped["FRAUD_RATE"] = grouped["FRAUD_COUNT"] / grouped["TOTAL_COUNT"].clip(lower=1)
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

    train_fe = add_time_features(train, cfg)
    val_fe = add_time_features(val, cfg)
    test_fe = add_time_features(test, cfg)

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

    stats_dict = compute_entity_fraud_stats(train_fe, cfg, entity_cols)
    train_fe = apply_entity_fraud_stats(train_fe, stats_dict)
    val_fe = apply_entity_fraud_stats(val_fe, stats_dict)
    test_fe = apply_entity_fraud_stats(test_fe, stats_dict)

    return train_fe, val_fe, test_fe


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
    logger.info("  Low cardinality (OHE, <%d): %d columns", low_card_threshold, len(low_card_ohe))
    if low_card_ohe:
        for col in low_card_ohe[:5]:
            logger.info("    - %s: %d unique values", col, cardinality_report[col])
        if len(low_card_ohe) > 5:
            logger.info("    ... and %d more", len(low_card_ohe) - 5)

    logger.info(
        "  Medium cardinality (Target Encode, %d-%d): %d columns",
        low_card_threshold,
        high_card_threshold,
        len(med_card_target),
    )
    if med_card_target:
        for col in med_card_target[:5]:
            logger.info("    - %s: %d unique values", col, cardinality_report[col])
        if len(med_card_target) > 5:
            logger.info("    ... and %d more", len(med_card_target) - 5)

    logger.info(
        "  High cardinality (Target Encode, >%d): %d columns",
        high_card_threshold,
        len(high_card_target),
    )
    if high_card_target:
        for col in high_card_target[:5]:
            logger.info("    - %s: %d unique values", col, cardinality_report[col])
        if len(high_card_target) > 5:
            logger.info("    ... and %d more", len(high_card_target) - 5)

    return low_card_ohe, med_card_target, high_card_target


def compute_target_encoding_stats(
    train: pd.DataFrame,
    cfg: Config,
    target_cols: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Compute target encoding (fraud rate) for high-cardinality columns from training data only.
    """
    stats_dict: Dict[str, pd.DataFrame] = {}
    for col in target_cols:
        if col not in train.columns:
            continue
        grouped = train.groupby(col)[cfg.label_col].agg(["sum", "count"]).reset_index()
        grouped.columns = [col, "fraud_count", "total_count"]
        grouped["FRAUD_RATE"] = grouped["fraud_count"] / grouped["total_count"].clip(lower=1)
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
        encoded_col = f"{col}_TARGET_ENCODED"
        df = df.merge(stats, on=col, how="left")
        df[encoded_col] = df[encoded_col].fillna(default_rate).astype("float32")
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
):
    """
    Build X/y for train/val/test using a scikit-learn ColumnTransformer pipeline.
    """
    logger.info("[STEP] Preparing ML feature matrix and preprocessing pipeline")

    exclude_cols = {
        cfg.label_col,
        cfg.trx_id_col,
    }

    entity_cols = set(cfg.entity_columns())
    id_like_cols = set(cfg.id_like_cols)
    parsed_date_cols = set(cfg.date_cols) | set(cfg.int_date_cols)
    drop_cols = entity_cols | id_like_cols | parsed_date_cols

    all_cols = list(train_fe.columns)
    feature_cols = [c for c in all_cols if c not in exclude_cols and c not in drop_cols]
    drop_count = len(drop_cols & set(all_cols))
    if drop_count:
        logger.info("Excluding %d raw ID/entity/date columns from model features", drop_count)

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_fe[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

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
            "Skipping target encoding for %d high-cardinality columns (ID-like or not whitelisted)",
            len(dropped_high_card),
        )
        for col in dropped_high_card[:5]:
            logger.info("    - %s", col)
    feature_cols = [c for c in feature_cols if c not in dropped_high_card]

    logger.info("  Numeric feature columns: %d", len(numeric_cols))
    logger.info("  Low cardinality categorical (OHE): %d", len(low_card_ohe))
    logger.info("  High cardinality categorical (Target Encoding): %d", len(target_encode_cols))

    if target_encode_cols:
        logger.info("Computing target encoding (fraud rates) for high-cardinality columns...")
        target_stats = compute_target_encoding_stats(train_fe, cfg, target_encode_cols)
        train_fe = apply_target_encoding(train_fe, target_stats, default_rate=0.0)
        val_fe = apply_target_encoding(val_fe, target_stats, default_rate=0.0)
        test_fe = apply_target_encoding(test_fe, target_stats, default_rate=0.0)
        logger.info("Applied target encoding to %d columns", len(target_encode_cols))

    feature_cols = [
        c for c in train_fe.columns
        if c not in exclude_cols and c not in target_encode_cols and c not in drop_cols
    ]

    numeric_cols = []
    categorical_cols_final: List[str] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(train_fe[col]):
            numeric_cols.append(col)
        else:
            categorical_cols_final.append(col)

    categorical_cols_final = [c for c in categorical_cols_final if c in low_card_ohe]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    ohe_kwargs: Dict[str, Any] = {"handle_unknown": "ignore"}
    ohe_params = OneHotEncoder.__init__.__code__.co_varnames
    # Backwards/forwards compatible: handle sparse vs sparse_output & min_frequency
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

    return X_train_t, y_train, X_val_t, y_val, X_test_t, y_test, model_pipeline


# ------------------------------------------------------------------------------
# Imbalance handling (synthetic oversampling on encoded train data)
# ------------------------------------------------------------------------------

def resample_training_data(
    X_train,
    y_train,
    cfg: Config,
):
    """
    Apply advanced imbalance handling with adaptive k_neighbors for small minority classes.

    Preferred: SMOTE-Tomek with BorderlineSMOTE (focusing on decision boundary).
    """
    if not cfg.use_smote_tomek:
        logger.info("SMOTE-Tomek disabled via config; using original training data")
        return X_train, y_train

    n_minority = int(y_train.sum())
    logger.info("Training data: %d total, %d fraud (minority)", len(X_train), n_minority)

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

    try:
        from imblearn.over_sampling import RandomOverSampler  # type: ignore
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
            "WARNING: Model will train on original highly imbalanced data (no synthetic oversampling)!"
        )
        return X_train, y_train
