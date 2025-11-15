#!/usr/bin/env python3
"""
Generate synthetic sample data for:
  - T_RT_FRD_JUMP_TRX  -> transactions.jsonl
  - T_RT_FRD_JUMP_ECM_TRX -> ecm.jsonl

All fields are exactly those from the data dictionary you provided.
No extra fields, no missing fields.

Usage:
    python generate_sample_jsonl.py \
        --n-transactions 200000 \
        --fraud-ratio 0.000037 \
        --out-dir ./local_data/sample

fraud_ratio is a FRACTION (not %):
  - 0.000037  ~ 0.0037%
  - 0.00037   ~ 0.037%
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import math
import random
import string
from typing import Any, Dict, List, Optional, Tuple


CHANNELS = ["WEB", "MOBILE", "ATM"]
CUSTOMER_SEGMENTS = ["Mass", "Mass Affluent", "SMB", "Private", "Corporate"]
PROFESSIONS = ["Engineer", "Teacher", "Retailer", "Consultant", "Doctor", "Analyst"]
ECONOMIC_ACTIVITIES = ["Services", "Manufacturing", "Agriculture", "Mining", "Technology"]
LANGUAGE_CHOICES = ["es", "en", "pt"]
DEVICE_OS_OPTIONS = [
    ("Android", "ANDROID_13"),
    ("iOS", "IOS_17"),
    ("Windows", "WIN_11"),
    ("macOS", "MAC_OS"),
]

COUNTRY_LOCATIONS: Dict[str, List[Dict[str, str]]] = {
    "CO": [
        {"city": "Bogota", "region": "Cundinamarca"},
        {"city": "Medellin", "region": "Antioquia"},
        {"city": "Cali", "region": "Valle del Cauca"},
    ],
    "US": [
        {"city": "New York", "region": "NY"},
        {"city": "Miami", "region": "FL"},
        {"city": "San Francisco", "region": "CA"},
    ],
    "MX": [
        {"city": "Mexico City", "region": "CDMX"},
        {"city": "Monterrey", "region": "Nuevo Leon"},
        {"city": "Guadalajara", "region": "Jalisco"},
    ],
    "BR": [
        {"city": "Sao Paulo", "region": "SP"},
        {"city": "Rio de Janeiro", "region": "RJ"},
        {"city": "Porto Alegre", "region": "RS"},
    ],
    "AR": [
        {"city": "Buenos Aires", "region": "BA"},
        {"city": "Cordoba", "region": "CB"},
        {"city": "Rosario", "region": "SF"},
    ],
    "ES": [
        {"city": "Madrid", "region": "Madrid"},
        {"city": "Barcelona", "region": "Catalonia"},
        {"city": "Seville", "region": "Andalusia"},
    ],
}

FIRST_NAMES = [
    "Maria",
    "Juan",
    "Laura",
    "Carlos",
    "Ana",
    "Luis",
    "Camila",
    "Pedro",
    "Daniel",
    "Sofia",
]
LAST_NAMES = [
    "Gomez",
    "Rodriguez",
    "Martinez",
    "Perez",
    "Ruiz",
    "Sanchez",
    "Lopez",
    "Torres",
    "Castro",
    "Vargas",
]


# ---------------------------------------------------------------------------
# 1. Schema definition (exact field names + raw type labels from dictionary)
# ---------------------------------------------------------------------------

TRX_FIELD_TYPES: Dict[str, str] = {
    # ------------- From T_RT_FRD_JUMP_TRX-357 -----------------
    "RT_FRD_JUMP_KEY": "INTEGER",
    "SOURCE_ID": "VARCHAR2(100)",
    "SOURCE_CD": "VARCHAR2(50)",
    "AUDIT_DT": "DATE",
    "LAST_UPD_DT": "DATE",
    "LOG_YEAR": "INTEGER",
    "LOG_MONTH": "INTEGER",
    "LOG_DAY": "INTEGER",
    "SEND_DATE": "INTEGER",
    "SEND_HOUR": "INTEGER",
    "RECEPTION_DATE": "INTEGER",
    "RECEPTION_HOUR": "INTEGER",
    "GENDER_ALERT_IND": "VARCHAR2(5)",
    "ENTITY_CD": "VARCHAR2(50)",
    "CUSTOMER_NO": "VARCHAR2(50)",
    "INTERNET_USER_CD": "VARCHAR2(50)",
    "COLPATRIA_DOCUMENT_TYPE_NAME": "VARCHAR2(200)",
    "CLIENT_CD": "VARCHAR2(50)",
    "AUTHN_TYPE_COLPATRIA_NAME": "VARCHAR2(200)",
    "COLPATRIA_AUTHN_RESPONSE_CD": "VARCHAR2(50)",
    "TRX_ORIGIN_CD": "VARCHAR2(50)",
    "OTP_IND": "VARCHAR2(5)",
    "CHALLENGE_QUESTIONS_SENT_IND": "VARCHAR2(5)",
    "CORRECT_ANSWERS_PR_IND": "VARCHAR2(5)",
    "DEVICE_PATTERN_DESC": "VARCHAR2(500)",
    "PERCENTAGE_OF_SIMILARITY_PCT": "VARCHAR2(10)",
    "TYPE_PERSON_IND": "VARCHAR2(5)",
    "HOST_RESPONSE_CD": "VARCHAR2(50)",
    "TRX_GROUP_CD": "VARCHAR2(50)",
    "LOCAL_OR_INTERNATIONAL_CD": "VARCHAR2(50)",
    "PRODUCT_CD": "VARCHAR2(50)",
    "SUBPRODUCT_CD": "VARCHAR2(50)",
    "ACC_PRODUCT_NO": "VARCHAR2(50)",
    "ORG_ACC_OP_BRANCH_CD": "VARCHAR2(50)",
    "EXEC_ACC_ORG_CHANNEL_CD": "VARCHAR2(50)",
    "REVERSE_IND": "VARCHAR2(5)",
    "TRX_DATE": "INTEGER",
    "TRX_HOUR": "INTEGER",
    "TRX_DT": "VARCHAR2(19)",
    "CONNECTION_IP_NO": "VARCHAR2(50)",
    "CITY_OR_LOCATION_NAME": "VARCHAR2(200)",
    "IP_COUNTRY_CD": "VARCHAR2(50)",
    "IP_REGION_NAME": "VARCHAR2(200)",
    "IP_REGION_CD": "VARCHAR2(50)",
    "LOCAL_IP_ADDRESS_NO": "VARCHAR2(50)",
    "INT_SERVICE_PROVIDER_NAME": "VARCHAR2(250)",
    "LANGUAGE_CD": "VARCHAR2(50)",
    "USER_CONFIRM_CD": "VARCHAR2(50)",
    "LAST_MOVEMENT_ACCOUNT_DATE": "VARCHAR2(19)",
    "ACCOUNT_OPENING_DATE": "VARCHAR2(19)",
    "MARK_VIP_ACCOUNT_TYPE_CD": "VARCHAR2(50)",
    "DIAL_REGISTERED_INT_ACC_CD": "VARCHAR2(50)",
    "COLPATRIA_ACCOUNT_NO": "VARCHAR2(50)",
    "CURRENCY_CD": "VARCHAR2(50)",
    "TRX_TOTAL_AMT": "DECIMAL(22,4)",
    "TRX_US_DOLLARS_AMT": "DECIMAL(22,4)",
    "EXCHANGE_RATE": "DECIMAL(22,12)",
    "DESTINATION_ACCOUNT_NO": "VARCHAR2(50)",
    "CLIENT_NAME": "VARCHAR2(200)",
    "DST_ACC_PLACE_NAME": "VARCHAR2(200)",
    "CLIENT_RIM_TARGET_CD": "VARCHAR2(50)",
    "DST_ACC_BRANCH_NAME": "VARCHAR2(200)",
    "ACC_HOLDER_DST_NAME": "VARCHAR2(200)",
    "DST_ACC_TEL_NO": "VARCHAR2(50)",
    "DST_ACC_EMAIL_LINE": "VARCHAR2(500)",
    "DST_ACC_OP_DATE": "VARCHAR2(19)",
    "TARGET_BANK_CD": "VARCHAR2(50)",
    "DESTINATION_PRODUCT_TYPE_CD": "VARCHAR2(50)",
    "DESTINATION_COUNTRY_CD": "VARCHAR2(50)",
    "DEBIT_CREDIT_OR_OTHER_CD": "VARCHAR2(50)",
    "ACCOUNT_STATUS_PRODUCT_CD": "VARCHAR2(50)",
    "SIGN_VALUE_BALANCE_IND": "VARCHAR2(5)",
    "AVAILABLE_AMT": "DECIMAL(22,4)",
    "HOME_ADDRESS_NAME": "VARCHAR2(350)",
    "PHONE_RESIDENCE_NO": "VARCHAR2(50)",
    "PHONE_OFFICE_NO": "VARCHAR2(50)",
    "PHONE_CELL_NO": "VARCHAR2(50)",
    "EMAIL_LINE": "VARCHAR2(500)",
    "EMAIL_DOMAIN_NAME": "VARCHAR2(200)",
    "CUSTOMER_BONDING_DATE": "VARCHAR2(19)",
    "CUSTOMER_SEGMENT_NAME": "VARCHAR2(200)",
    "REFERENCE_NO": "VARCHAR2(50)",
    "SERVICE_PAYMENT_REF_3_NO": "VARCHAR2(50)",
    "SERVICE_PAYMENT_REF_4_NO": "VARCHAR2(50)",
    "MESSAGE_TYPE_CD": "VARCHAR2(50)",
    "P03_CD": "VARCHAR2(50)",
    "FUTURE_USE_1_DESC": "VARCHAR2(500)",
    "FUTURE_USE_2_NO": "VARCHAR2(50)",
    "FUTURE_USE_3_NO": "VARCHAR2(50)",
    "FRAUD_IND": "VARCHAR2(5)",
    "ALERT_SOURCE_CD": "VARCHAR2(50)",
    "CORRELATIVE_NO": "VARCHAR2(50)",
    "TYPED_KEYS_NO": "VARCHAR2(50)",
    "RATE_TYPE_CD": "VARCHAR2(50)",
    "REAL_TIME_IND": "VARCHAR2(5)",
    "USER_DEVICE_PATTERN_DDS_NO": "VARCHAR2(115)",
    "SESSION_ID": "VARCHAR2(100)",
    "DEVICE_HASH_NO": "VARCHAR2(50)",
    "HASHINTEGRITY_NO": "VARCHAR2(50)",
    "COOKIE_TEXT": "VARCHAR2(1000)",
    "LOCAL_STORAGE_VALUE_TEXT": "VARCHAR2(1000)",
    "DEVICE_OPERATING_SYSTEM_NO": "VARCHAR2(50)",
    "DEVICE_NO": "VARCHAR2(50)",
    "IMEI_NO": "VARCHAR2(50)",
    "MACADDRESS_NO": "VARCHAR2(50)",
    "HOSTNAME_NAME": "VARCHAR2(200)",
    "PATHNAME_NAME": "VARCHAR2(200)",
    "PROTOCOL_NAME": "VARCHAR2(200)",
    "CERTIFIED_IDENTIFICATION_DESC": "VARCHAR2(500)",
    "HASLIEDRESOLUTION_CD": "VARCHAR2(50)",
    "HASLIEDOS_CD": "VARCHAR2(50)",
    "HASLIEDBROWSER_CD": "VARCHAR2(50)",
    "DEVICE_POSITION_NAME": "VARCHAR2(200)",
    "DEVICE_PRESSURE_NO": "VARCHAR2(50)",
    "CLIENT_NO": "VARCHAR2(50)",
    "ACCOUNT_NO": "VARCHAR2(50)",
    "RESULTING_DBFD_GUID_DESC": "VARCHAR2(500)",
    "DEVICE_RISK_SCORE": "INTEGER",
    "BROWSING_HABITS_SCORE": "INTEGER",
    "COMP_ORG_ACC_SCORE": "INTEGER",
    "DST_ACC_ANALYSIS_SCORE": "INTEGER",
    "CONDITIONS_MET_TEXT": "VARCHAR2(1000)",
    "TRANSACTION_DT": "Date8",
    "TRANSACTION_DT_TIME": "DATETIME",
    "EMPLOYEE_ACC_ORI_IND": "VARHCAR1",
    "SVC_PAYMENT_CDO": "VARCHAR6",
    "ISTOR_IND": "VARCHAR1",
    "EMULATOR_IND": "VARCHAR1",
    "PORT_IND": "VARCHAR10",
    "HAS_LIEDLANGUAGES_CD": "VARCHAR1",
    "FIRST_NAME": "VARCHAR50",
    "SECOND_NAME": "VARCHAR50",
    "SURNAME": "VARCHAR50",
    "SECOND_SURNAME": "VARCHAR50",
    "MARRIED_SURNAME": "VARCHAR25",
    "GENDER": "VARCHAR1",
    "CIVIL_STATUS": "VARCHAR1",
    "BIRTH_DATE": "INTEGER10",
    "NATIONALITY_NAME": "VARCHAR3",
    "COUNTRY_BIRTH_NAME": "VARCHAR3",
    "AGENRO": "VARCHAR3",
    "COUNTRY_RESIDENCE_NAME": "VARCHAR30",
    "PROFESSION_NAME": "VARCHAR30",
    "ECONOMIC_NAME": "VARCHAR50",
    "DEPENDENCY_RELATIONSHIP_NAME": "VARCHAR1",
    "COMPANY_WORK_NAME": "VARCHAR50",
    "DIRECTION_WORK_NAME": "VARCHAR50",
    "PUBLIC_SERVANT_PEP_NAME": "VARCHAR1",
    "FAMILY_MEMBER_PEP_NAME": "VARCHAR1",
    "PEP_COLLABORATOR_NAME": "VARCHAR1",
    "OS_NAME": "VARCHAR20",
    "SERIAL_NRO": "VARCHAR20",
    "APPLICATION_CD": "VARCHAR10",
    "AUTH_CD": "VARCHAR20",
    "ZIP_CD": "VARCHAR10",
    "POSTAL_CD": "VARCHAR5",
    "MOVEMENT_DEVICE_CD": "VARCHAR1",
    "DEVICEPRESSUREPERPOINT_CD": "VARCHAR5",
    "ISROGUEPROXY_CD": "VARCHAR10",
    "GUID_NRO": "VARCHAR47",
    "SESSIONSTORAGE_CD": "VARCHAR1",
    "INDEXEDDB_NRO": "VARCHAR1",
    "ORIGIN_NRO": "VARCHAR40",
    "AUTH_2_CD": "VARCHAR25",
    "AUTH_3_CD": "VARCHAR25",
    "RESPONSE_AUTH_2_IND": "VARCHAR5",
    "RESPONSE AUTH_3_IND": "VARCHAR5",  # NOTE: space is intentional
    "ORIG_ACC_HOLDER_NAME": "VARCHAR40",
    "SURNAME_ORIG_ACC_HOLDER": "VARCHAR40",
    "DESTINATION_ENTITY_NAME": "VARCHAR80",
    "RESULTING_REGISTRATION_CD": "VARCHAR5",
    "LAST_CONNECTION_JUMP_DATE": "INTEGER8",
    "CONS_MASIVIAN_HOUR": "INTEGER6",
    "CONS_MASIVIAN_DATE": "INTEGER8",
    "RESPONSE_MASIVIAN_HOUR": "INTEGER6",
    "RESPONSE_MASIVIAN_DATE": "INTEGER8",
    "CELLULAR_CONS_MASIVIAN_NRO": "VARCHAR15",
    "CONFIRMED_OPER_MASIVIAN_NAME": "VARCHAR15",
    "RESP_VALUE_SIM_CHANGE_CD": "VARCHAR1",
    "SHERLOCK_MASIVIAN_ANOMALY_CD": "VARCHAR4",
    "MISTRUST_SHERLOCK_MASIVIAN_CD": "VARCHAR4",
    "SHERLOCK_MASIVIAN_PORTAB_CD": "VARCHAR1",
    "OPERATOR_ANT_NAME": "VARCHAR15",
    "SHERLOCK_MASSIVIAN_AGE_CD": "VARCHAR5",
    "PAYMENT_REFE_NRO": "INTEGER30",
    "PAYMENT_AGREEMENT_CD": "VARCHAR15",
    "FAVOR_BRAND_REF_PAYMENT_CD": "VARCHAR1",
    "REF_LAST_PAYMENT_HOUR": "INTEGER6",
    "LAST_REF_PAYMENT_DATE": "INTEGER8",
    "LAST_VALUE_PAYMENT_REF_NRO": "VARCHAR14",
    "PAYMENT_MADE_REFE_CNT": "VARCHAR4",
    "TIME_PAYMENT_NAME": "VARCHAR15",
    "PAYMENT_LIMIT_ATM": "INTEGER14",
    "PROG_DAYS_BEFORE_EXP_NO": "INTEGER2",
    "LAST_UP_DATE": "INTEGER8",
    "LAST_UP_HOUR": "INTEGER6",
}

ECM_FIELD_TYPES: Dict[str, str] = {
    "RECEPTION_DATE": "INTEGER",
    "RECEPTION_HOUR": "INTEGER",
    "GENDER_ALERT_IND": "VARCHAR2(5)",
    "INVESTIGATION_DATE": "INTEGER",
    "RESEARCH_HOUR": "INTEGER",
    "CLOSING_DATE": "VARCHAR2(19)",
    "CLOSING_HOUR": "INTEGER",
    "ALERT_CONDITIONS_TEXT": "VARCHAR2(1000)",
    "RESULT_TYPE_CD": "VARCHAR2(50)",
    "SUBTYPE_RESULT_CD": "VARCHAR2(50)",
    "TRX_IND": "VARCHAR2(50)",
    "CORRECTION_IND": "VARCHAR2(5)",
    "CORRELATIVE_NO": "VARCHAR2(50)",
    "SESSION_ID": "VARCHAR2(60)",
    "TRX_DATE": "INTEGER",
    "TRX_HOUR": "INTEGER",
}


# ---------------------------------------------------------------------------
# 2. Random helpers
# ---------------------------------------------------------------------------

def random_date(start_year=2020, end_year=2024) -> datetime:
    """Random datetime between Jan 1 of start_year and Dec 31 of end_year."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days),
                             seconds=random.randint(0, 24 * 3600 - 1))


def date_to_int_yyyymmdd(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d"))


def date_to_str_yyyy_mm_dd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def datetime_to_str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def random_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def random_phone(length: int = 10) -> str:
    return "".join(random.choice(string.digits) for _ in range(length))


def random_email() -> str:
    user = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    domain = random.choice(["gmail.com", "yahoo.com", "hotmail.com", "bank.com"])
    return f"{user}@{domain}"


def random_string(max_len: int) -> str:
    """Generate a random string with length between 1 and max_len
    (or 3 and max_len if max_len is large enough)."""
    if max_len <= 0:
        max_len = 1

    # For very short fields (e.g. VARCHAR1, VARCHAR2), allow length 1..max_len.
    # For normal fields, keep your old behaviour (3..max_len).
    if max_len < 3:
        min_len = 1
    else:
        min_len = 3

    length = random.randint(min_len, max_len)
    chars = string.ascii_letters + string.digits + " _-"
    return "".join(random.choice(chars) for _ in range(length))


def random_country_code() -> str:
    return random.choice(list(COUNTRY_LOCATIONS.keys()))


def random_currency() -> str:
    return random.choice(["COP", "USD", "EUR"])


def random_gender_flag() -> str:
    return random.choice(["M", "F", "U"])


def random_yes_no_flag() -> str:
    return random.choice(["Y", "N"])


def random_language() -> str:
    return random.choice(["es", "en", "pt", "fr"])


def random_amount() -> float:
    # Skewed distribution: many small, few large amounts
    base = 10 ** random.uniform(1, 5)  # 10..100000
    return round(base, 2)


def random_exchange_rate() -> float:
    return round(random.uniform(3000, 5000), 4)


def random_int_by_digits(digits: int) -> int:
    lo = 10 ** (digits - 1)
    hi = 10 ** digits - 1
    return random.randint(lo, hi)


def sigmoid(z: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        return 0.0 if z < 0 else 1.0


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def random_full_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def pick_location(country: str) -> Dict[str, str]:
    return random.choice(COUNTRY_LOCATIONS.get(country, COUNTRY_LOCATIONS["CO"]))


def pick_alert_text(reasons: List[str], triggered: bool) -> str:
    if not triggered:
        return "No anomaly detected"
    if not reasons:
        reasons = ["Rule threshold exceeded"]
    return " | ".join(reasons)


def create_device_profile(customer_no: str) -> Dict[str, str]:
    suffix = "".join(random.choice("0123456789ABCDEF") for _ in range(10))
    device_id = f"DEV-{customer_no[-4:]}-{suffix}"
    os_name, os_code = random.choice(DEVICE_OS_OPTIONS)
    return device_id, {
        "DEVICE_OPERATING_SYSTEM_NO": os_code,
        "OS_NAME": os_name,
    }


def sample_transaction_datetime(night_prob: float) -> datetime:
    start = datetime.now() - timedelta(days=730)
    dt = start + timedelta(days=random.randint(0, 729))
    if random.random() < night_prob:
        hour = random.choice([0, 1, 2, 3, 4, 23])
    else:
        hour = random.randint(7, 21)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return dt.replace(hour=hour, minute=minute, second=second, microsecond=0)


def build_customer_profiles(
    n_customers: int,
    device_registry: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    customers: List[Dict[str, Any]] = []
    for idx in range(n_customers):
        customer_no = f"CUST-{idx:06d}"
        country = random_country_code()
        location = pick_location(country)
        profile: Dict[str, Any] = {
            "customer_no": customer_no,
            "client_no": f"CL-{idx:06d}",
            "client_name": random_full_name(),
            "home_country": country,
            "city": location["city"],
            "region": location["region"],
            "segment": random.choice(CUSTOMER_SEGMENTS),
            "profession": random.choice(PROFESSIONS),
            "economic_activity": random.choice(ECONOMIC_ACTIVITIES),
            "preferred_currency": random.choice(["COP", "COP", "USD", "EUR"]),
            "language": "es" if country in {"CO", "MX", "AR", "ES"} else ("pt" if country == "BR" else "en"),
            "amount_mu": random.uniform(8.0, 10.0),
            "night_owl_prob": random.uniform(0.05, 0.35),
            "intl_propensity": random.uniform(0.05, 0.4),
            "channel_weights": [random.uniform(0.3, 1.0) for _ in CHANNELS],
        }
        profile["accounts"] = [
            f"ACC-{customer_no[-4:]}-{j:02d}" for j in range(random.randint(1, 3))
        ]
        profile["devices"] = []
        for _ in range(random.randint(1, 3)):
            device_id, metadata = create_device_profile(customer_no)
            profile["devices"].append(device_id)
            device_registry[device_id] = metadata
        profile["ips"] = [random_ip() for _ in range(random.randint(1, 4))]
        profile["phones"] = [random_phone() for _ in range(random.randint(1, 3))]
        profile["emails"] = [random_email() for _ in range(random.randint(1, 3))]
        customers.append(profile)
    return customers


def build_destination_accounts(n_destinations: int) -> List[Dict[str, Any]]:
    dests: List[Dict[str, Any]] = []
    for idx in range(n_destinations):
        country = random_country_code()
        location = pick_location(country)
        dests.append(
            {
                "account_no": f"DST-{idx:07d}",
                "entity_name": f"Beneficiary {idx:05d}",
                "country": country,
                "city": location["city"],
                "branch": f"{location['city']} Branch",
                "risk_score": random.uniform(0.1, 0.9),
                "weight": random.uniform(0.5, 3.0),
            }
        )
    return dests


def choose_channel(profile: Dict[str, Any]) -> str:
    weights = profile.get("channel_weights") or [1.0 for _ in CHANNELS]
    return random.choices(CHANNELS, weights=weights, k=1)[0]


def select_device_for_customer(
    profile: Dict[str, Any],
    history: Dict[str, Any],
    device_registry: Dict[str, Dict[str, str]],
) -> Tuple[str, Dict[str, str], bool]:
    use_new = random.random() < 0.15 or not profile["devices"]
    if use_new:
        device_id, metadata = create_device_profile(profile["customer_no"])
        profile["devices"].append(device_id)
        device_registry[device_id] = metadata
    else:
        device_id = random.choice(profile["devices"])
        metadata = device_registry[device_id]
    is_new = device_id not in history["seen_devices"]
    history["seen_devices"].add(device_id)
    return device_id, metadata, is_new


def select_ip_for_customer(profile: Dict[str, Any], history: Dict[str, Any]) -> Tuple[str, bool]:
    use_new = random.random() < 0.1 or not profile["ips"]
    if use_new:
        ip = random_ip()
        profile["ips"].append(ip)
    else:
        ip = random.choice(profile["ips"])
    is_new = ip not in history["seen_ips"]
    history["seen_ips"].add(ip)
    return ip, is_new


# ---------------------------------------------------------------------------
# 3. Value generator per field/type
# ---------------------------------------------------------------------------

def generate_trx_value(
    field: str,
    type_label: str,
    is_fraud: bool,
    trx_index: int,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    if context and field in context:
        return context[field]
    tl = type_label.upper()

    # Special cases / semantics first
    if field == "RT_FRD_JUMP_KEY":
        return trx_index + 1

    if field == "FRAUD_IND":
        # Represented as "1"/"0" in VARCHAR2(5)
        return "1" if is_fraud else "0"

    if field == "CORRELATIVE_NO":
        # Use as transaction id / join key
        return f"TRX-{trx_index+1:010d}"

    if field == "SESSION_ID":
        # Some transactions share sessions
        session_base = trx_index // random.randint(5, 20)
        return f"SESS-{session_base:08d}"

    if field == "CURRENCY_CD":
        return random_currency()

    if field == "CONNECTION_IP_NO" or field == "LOCAL_IP_ADDRESS_NO":
        return random_ip()

    if "EMAIL" in field:
        return random_email()

    if "PHONE" in field or "TEL" in field:
        # lengths vary; we'll just do up to 10-12 digits
        return random_phone(length=10)

    if field in {"TRX_TOTAL_AMT", "TRX_US_DOLLARS_AMT"}:
        amt = random_amount()
        if field == "TRX_US_DOLLARS_AMT":
            # simple conversion
            return round(amt / random.uniform(3800, 4200), 2)
        return amt

    if field == "EXCHANGE_RATE":
        return random_exchange_rate()

    if "COUNTRY" in field:
        return random_country_code()

    if "LANGUAGE" in field:
        return random_language()

    if field in {"GENDER_ALERT_IND", "GENDER"}:
        return random_gender_flag()

    # Date/time-ish fields treated specially
    if "DATE" in field and "BIRTH_DATE" not in field and "UP_DATE" not in field:
        dt = random_date()
        if "INTEGER" in tl or tl.endswith("8") or tl.endswith("10"):
            return date_to_int_yyyymmdd(dt)
        else:
            return datetime_to_str(dt)

    if field in {"AUDIT_DT", "LAST_UPD_DT"}:
        dt = random_date()
        return date_to_str_yyyy_mm_dd(dt)

    if "BIRTH_DATE" in field:
        # Birth date ~ 18-80 years ago
        now = datetime.now()
        years_ago = random.randint(18, 80)
        birth = now - timedelta(days=years_ago * 365)
        if "INTEGER" in tl:
            return int(birth.strftime("%Y%m%d"))
        return date_to_str_yyyy_mm_dd(birth)

    if "HOUR" in field:
        # integer hours 0-23 or hhmmss style for INTEGER6
        if tl.endswith("6"):
            # hhmmss
            hh = random.randint(0, 23)
            mm = random.randint(0, 59)
            ss = random.randint(0, 59)
            return int(f"{hh:02d}{mm:02d}{ss:02d}")
        else:
            return random.randint(0, 23)

    if field in {"TRANSACTION_DT", "LAST_UP_DATE", "LAST_CONNECTION_JUMP_DATE",
                 "CONS_MASIVIAN_DATE", "RESPONSE_MASIVIAN_DATE",
                 "LAST_REF_PAYMENT_DATE"}:
        dt = random_date()
        return int(dt.strftime("%Y%m%d"))

    if field == "TRANSACTION_DT_TIME":
        dt = random_date()
        return datetime_to_str(dt)

    # Now generic type-based behaviour
    if "DECIMAL" in tl:
        return random_amount() if "AMT" in field or "LIMIT" in field else round(random.uniform(0, 100000), 4)

    if "INTEGER" in tl:
        # handle various INTEGERX lengths
        if tl.endswith("30"):
            return random_int_by_digits(10)
        if tl.endswith("14"):
            return random_int_by_digits(9)
        if tl.endswith("10"):
            return random_int_by_digits(8)
        if tl.endswith("8"):
            return random_int_by_digits(8)
        if tl.endswith("6"):
            return random_int_by_digits(6)
        if tl.endswith("2"):
            return random_int_by_digits(2)
        return random.randint(0, 999999)

    # Char / varchar things
    if "CHAR" in tl or "VARCHAR" in tl:
        # Use some light domain-specific codes if obvious
        if field.endswith("_CD") or field.endswith("_IND"):
            return random.choice(["A", "B", "C", "D", "E", "Y", "N", "0", "1"])
        max_len = 10
        # extract length from VARCHARx if present
        for tag in ["VARCHAR2(", "VARCHAR", "VARHCAR"]:
            if tag in tl:
                try:
                    part = tl.split(tag, 1)[1]
                    digits = ""
                    for ch in part:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    if digits:
                        max_len = min(int(digits), 50)
                except Exception:
                    pass
        return random_string(max_len)

    if "DATE" in tl or "DATETIME" in tl:
        dt = random_date()
        return datetime_to_str(dt)

    # Fallback
    return random_string(20)


def generate_ecm_value(
    field: str,
    type_label: str,
    trx_row: Dict[str, Any],
    is_fraud: bool,
    context: Optional[Dict[str, Any]] = None,
) -> Any:
    if context and field in context:
        return context[field]
    tl = type_label.upper()

    # Link fields
    if field == "CORRELATIVE_NO":
        return trx_row["CORRELATIVE_NO"]
    if field == "SESSION_ID":
        # clip to ECM max length (60) just in case
        return str(trx_row.get("SESSION_ID", ""))[:60]

    # Mirror TRX date/hour when sensible
    if field == "TRX_DATE":
        return trx_row.get("TRX_DATE", date_to_int_yyyymmdd(random_date()))
    if field == "TRX_HOUR":
        return trx_row.get("TRX_HOUR", random.randint(0, 23))

    # Investigation/closing times
    if field in {"RECEPTION_DATE", "INVESTIGATION_DATE"}:
        dt = random_date()
        return date_to_int_yyyymmdd(dt)

    if field in {"RECEPTION_HOUR", "RESEARCH_HOUR", "CLOSING_HOUR"}:
        return random.randint(0, 23)

    if field == "CLOSING_DATE":
        dt = random_date()
        return datetime_to_str(dt)

    # Fraud-related-ish outcome codes (not labels, just outcome flavour)
    if field == "RESULT_TYPE_CD":
        return "CONFIRMED_FRAUD" if is_fraud else random.choice(["GENUINE", "DISCARDED", "FALSE_POSITIVE"])

    if field == "SUBTYPE_RESULT_CD":
        return random.choice(["RULE_ENGINE", "MANUAL_REVIEW", "CUSTOMER_CALLBACK", "OTHER"])

    if field == "TRX_IND":
        return "Y"

    if field == "CORRECTION_IND":
        return random_yes_no_flag()

    if field == "GENDER_ALERT_IND":
        return random_gender_flag()

    if field == "ALERT_CONDITIONS_TEXT":
        if is_fraud:
            return "High-value international transfer with unusual device pattern"
        else:
            return "Standard rule-based alert; no anomaly confirmed"

    # Types
    if "INTEGER" in tl:
        return random.randint(0, 999999)

    if "VARCHAR" in tl or "CHAR" in tl:
        max_len = 10
        for tag in ["VARCHAR2(", "VARCHAR"]:
            if tag in tl:
                try:
                    part = tl.split(tag, 1)[1]
                    digits = ""
                    for ch in part:
                        if ch.isdigit():
                            digits += ch
                        else:
                            break
                    if digits:
                        max_len = min(int(digits), 100)
                except Exception:
                    pass
        return random_string(max_len)

    if "DATE" in tl or "DATETIME" in tl:
        dt = random_date()
        return datetime_to_str(dt)

    # Fallback
    return random_string(20)


# ---------------------------------------------------------------------------
# 4. Main generation logic
# ---------------------------------------------------------------------------

def generate_datasets(
    n_transactions: int,
    fraud_ratio: float,
    out_dir: Path,
    seed: int = 42,
) -> None:
    random.seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    trx_path = out_dir / "transactions.jsonl"
    ecm_path = out_dir / "ecm.jsonl"

    trimmed_ratio = min(max(fraud_ratio, 1e-4), 0.95)
    base_logit = math.log(trimmed_ratio / (1 - trimmed_ratio))

    n_customers = max(200, n_transactions // 5)
    n_destinations = max(100, n_transactions // 10)
    device_registry: Dict[str, Dict[str, str]] = {}
    customers = build_customer_profiles(n_customers, device_registry)
    dest_accounts = build_destination_accounts(n_destinations)
    dest_weights = [dest["weight"] for dest in dest_accounts]

    customer_history = {
        cust["customer_no"]: {
            "seen_devices": set(),
            "seen_ips": set(),
            "seen_destinations": set(),
            "fraud_count": 0,
        }
        for cust in customers
    }

    print(f"[INFO] Generating {n_transactions} transactions (target fraud ratio ≈ {fraud_ratio:.6f})")

    trx_fields = list(TRX_FIELD_TYPES.keys())
    ecm_fields = list(ECM_FIELD_TYPES.keys())

    observed_fraud = 0

    with trx_path.open("w", encoding="utf-8") as f_trx, ecm_path.open("w", encoding="utf-8") as f_ecm:
        for i in range(n_transactions):
            profile = random.choice(customers)
            history = customer_history[profile["customer_no"]]
            account_no = random.choice(profile["accounts"])

            device_id, device_meta, is_new_device = select_device_for_customer(profile, history, device_registry)
            ip_address, is_new_ip = select_ip_for_customer(profile, history)
            phone = random.choice(profile["phones"])
            email = random.choice(profile["emails"])

            dest = random.choices(dest_accounts, weights=dest_weights, k=1)[0]
            if dest["country"] == profile["home_country"] and random.random() < profile["intl_propensity"]:
                intl_options = [d for d in dest_accounts if d["country"] != profile["home_country"]]
                if intl_options:
                    dest = random.choice(intl_options)

            is_new_dest = dest["account_no"] not in history["seen_destinations"]
            history["seen_destinations"].add(dest["account_no"])

            trx_dt = sample_transaction_datetime(profile["night_owl_prob"])
            trx_date_int = int(trx_dt.strftime("%Y%m%d"))
            trx_hour = trx_dt.hour

            currency = profile["preferred_currency"]
            raw_amount = math.exp(random.gauss(profile["amount_mu"], 0.8)) - 1.0
            amount = round(max(5.0, raw_amount), 2)
            if currency == "USD":
                exchange_rate = 1.0
                usd_amount = round(amount, 2)
            else:
                exchange_rate = random.uniform(3700, 4300)
                usd_amount = round(amount / exchange_rate, 2)

            log_amount_centered = math.log1p(amount) - 9.0
            is_night = trx_hour < 6 or trx_hour >= 23
            is_international = dest["country"] != profile["home_country"]
            has_prior_fraud = history["fraud_count"] > 0

            z = (
                base_logit
                + 0.8 * log_amount_centered
                + 1.1 * int(is_new_device)
                + 0.7 * int(is_new_ip)
                + 0.9 * int(is_new_dest)
                + 0.5 * int(is_night)
                + 0.6 * int(is_international)
                + 1.0 * int(has_prior_fraud)
                + 0.6 * (dest["risk_score"] - 0.5)
                + random.gauss(0, 0.8)
            )
            fraud_prob = sigmoid(z)
            is_fraud = random.random() < fraud_prob
            if (n_transactions - i) == 1 and observed_fraud == 0:
                is_fraud = True

            if is_fraud:
                history["fraud_count"] += 1
                observed_fraud += 1

            alert_score = clip01(
                0.4
                + 0.25 * int(is_new_device)
                + 0.2 * int(is_new_dest)
                + 0.15 * int(is_new_ip)
                + 0.15 * int(is_night)
                + 0.15 * int(is_international)
                + 0.1 * max(0.0, log_amount_centered)
                + 0.1 * (dest["risk_score"] - 0.5)
                + random.gauss(0, 0.12)
            )
            alert_triggered = alert_score > 0.55 or random.random() < 0.05

            reasons: List[str] = []
            if is_new_device:
                reasons.append("Device not seen for this customer")
            if is_new_dest:
                reasons.append("Beneficiary never used")
            if is_international:
                reasons.append("International destination")
            if log_amount_centered > 0.5:
                reasons.append("Amount far above normal")
            if is_night:
                reasons.append("Night-time activity")
            if alert_score > 0.75:
                reasons.append("Risk score extremely high")
            alert_text = pick_alert_text(reasons, alert_triggered)

            if alert_triggered:
                if is_fraud:
                    result_type = random.choices(["FRAUD", "GENUINE"], weights=[0.8, 0.2])[0]
                else:
                    result_type = random.choices(["GENUINE", "FRAUD"], weights=[0.95, 0.05])[0]
            else:
                result_type = "GENUINE"

            trx_ind = "ALERTED" if alert_triggered else "NO_ALERT"
            if random.random() < 0.05:
                trx_ind = "ALERTED" if trx_ind == "NO_ALERT" else "NO_ALERT"
            correction_ind = "Y" if (result_type == "FRAUD" and not is_fraud and random.random() < 0.7) else "N"
            subtype = random.choice(["INVESTIGATE", "CLOSED", "ESCALATED", "DISMISSED"])

            channel = choose_channel(profile)
            trx_context = {
                "CUSTOMER_NO": profile["customer_no"],
                "CLIENT_NO": profile["client_no"],
                "CLIENT_NAME": profile["client_name"],
                "ACCOUNT_NO": account_no,
                "ACC_PRODUCT_NO": account_no,
                "DESTINATION_ACCOUNT_NO": dest["account_no"],
                "DST_ACC_PLACE_NAME": dest["city"],
                "DST_ACC_BRANCH_NAME": dest["branch"],
                "DESTINATION_COUNTRY_CD": dest["country"],
                "DESTINATION_ENTITY_NAME": dest["entity_name"],
                "DST_ACC_EMAIL_LINE": f"{dest['entity_name'].lower().replace(' ', '_')}@destbank.com",
                "DST_ACC_TEL_NO": random_phone(),
                "CURRENCY_CD": currency,
                "TRX_TOTAL_AMT": amount,
                "TRX_US_DOLLARS_AMT": usd_amount,
                "EXCHANGE_RATE": exchange_rate,
                "TRX_DATE": trx_date_int,
                "TRANSACTION_DT": trx_date_int,
                "TRX_HOUR": trx_hour,
                "TRX_DT": trx_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "TRANSACTION_DT_TIME": trx_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "TRX_ORIGIN_CD": channel,
                "TRX_GROUP_CD": channel,
                "LOCAL_OR_INTERNATIONAL_CD": "INTERNATIONAL" if is_international else "LOCAL",
                "CITY_OR_LOCATION_NAME": profile["city"],
                "IP_REGION_NAME": profile["region"],
                "IP_REGION_CD": f"{profile['home_country']}-{profile['region'][:3].upper()}",
                "IP_COUNTRY_CD": profile["home_country"],
                "CONNECTION_IP_NO": ip_address,
                "LOCAL_IP_ADDRESS_NO": ip_address,
                "DEVICE_HASH_NO": device_id,
                "DEVICE_OPERATING_SYSTEM_NO": device_meta["DEVICE_OPERATING_SYSTEM_NO"],
                "OS_NAME": device_meta["OS_NAME"],
                "PHONE_CELL_NO": phone,
                "PHONE_RESIDENCE_NO": phone,
                "EMAIL_LINE": email,
                "EMAIL_DOMAIN_NAME": email.split("@")[1],
                "CUSTOMER_SEGMENT_NAME": profile["segment"],
                "PROFESSION_NAME": profile["profession"],
                "ECONOMIC_NAME": profile["economic_activity"],
                "LANGUAGE_CD": profile["language"],
                "AVAILABLE_AMT": round(amount * random.uniform(1.5, 4.0), 2),
                "DESTINATION_PRODUCT_TYPE_CD": random.choice(["WIRE", "ACH", "SWIFT"]),
                "CLIENT_RIM_TARGET_CD": "CORP" if profile["segment"] == "Corporate" else "RET",
                "DEVICE_RISK_SCORE": int(30 + 60 * random.random()),
                "BROWSING_HABITS_SCORE": int(30 + 60 * random.random()),
                "COMP_ORG_ACC_SCORE": int(30 + 60 * random.random()),
                "DST_ACC_ANALYSIS_SCORE": int(40 + 50 * dest["risk_score"]),
                "FRAUD_IND": "1" if is_fraud else "0",
            }

            ecm_context = {
                "RESULT_TYPE_CD": result_type,
                "SUBTYPE_RESULT_CD": subtype,
                "TRX_IND": trx_ind,
                "CORRECTION_IND": correction_ind,
                "ALERT_CONDITIONS_TEXT": alert_text,
            }

            trx_row: Dict[str, Any] = {}
            for field in trx_fields:
                type_label = TRX_FIELD_TYPES[field]
                trx_row[field] = generate_trx_value(field, type_label, is_fraud, i, trx_context)

            ecm_row: Dict[str, Any] = {}
            for field in ecm_fields:
                type_label = ECM_FIELD_TYPES[field]
                ecm_row[field] = generate_ecm_value(field, type_label, trx_row, is_fraud, ecm_context)

            f_trx.write(json.dumps(trx_row, ensure_ascii=False) + "\n")
            f_ecm.write(json.dumps(ecm_row, ensure_ascii=False) + "\n")

    observed_ratio = observed_fraud / n_transactions
    print(f"[INFO] Wrote transactions to: {trx_path}")
    print(f"[INFO] Wrote ECM records to:  {ecm_path}")
    print(f"[INFO] Observed fraud rows: {observed_fraud} ({observed_ratio:.4%})")


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate JSONL sample data for T_RT_FRD_JUMP_TRX and T_RT_FRD_JUMP_ECM_TRX"
    )
    parser.add_argument(
        "--n-transactions",
        type=int,
        default=200000,
        help="Number of transactions to generate (default: 200000)",
    )
    parser.add_argument(
        "--fraud-ratio",
        type=float,
        default=0.000037,  # ~0.0037%
        help="Fraud ratio as a fraction (e.g., 0.000037 ≈ 0.0037%%)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./local_data/sample",
        help="Output directory for transactions.jsonl and ecm.jsonl",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    generate_datasets(
        n_transactions=args.n_transactions,
        fraud_ratio=args.fraud_ratio,
        out_dir=out_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
