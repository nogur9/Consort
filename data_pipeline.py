"""Data loading and preprocessing utilities for the CONSORT dashboard."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from app_config import (
    ALIASES,
    CONSORT_GROUPS,
    DATE_FORMATS,
    EXCLUDE_SHEETS,
    GROUPS_FILE,
    GROUPS_RENAME,
    MAX_WAITING_DAYS_DEFAULT,
    PRIORITY_MAP,
    SUITABLE_FOR_PP_RENAME
)


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #
def _locate_first_existing(candidates: Iterable[Path]) -> Path:
    """Return the first existing path from the candidates list."""
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(
        f"None of the candidate files were found: {', '.join(str(p) for p in candidates)}"
    )


def _resolve_data_source(data_source):
    if data_source is None:
        raise ValueError
    if isinstance(data_source, (bytes, bytearray)):
        return BytesIO(data_source)
    return data_source


def _load_workbooks(data_source=None) -> Tuple[pd.ExcelFile, pd.ExcelFile]:
    """Load the source workbooks for patient data and group assignments."""
    data_input = _resolve_data_source(data_source)
    return pd.ExcelFile(data_input), pd.ExcelFile(GROUPS_FILE)


# --------------------------------------------------------------------------- #
# Step 1 - Raw extraction and normalization
# --------------------------------------------------------------------------- #
def _drop_trailing_s(value):
    if pd.isna(value):
        return value
    value_str = str(value).strip().lower()
    return value_str[:-1] if value_str.endswith("s") else value_str


def _build_rename_map() -> Dict[str, str]:
    return {alias: canonical for canonical, values in ALIASES.items() for alias in values}


def _normalize_sheet(sheet_df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """Keep only relevant columns, rename aliases, and normalize IDs."""
    rename_map = _build_rename_map()
    columns = list(ALIASES.keys()) + ["sheet", "clean_id"]

    sheet_df = sheet_df.copy()
    sheet_df["sheet"] = sheet_name
    sheet_df.rename(columns=rename_map, inplace=True, errors="ignore")
    sheet_df["clean_id"] = sheet_df["raw_id"].astype(str).apply(_drop_trailing_s)

    if "group" in sheet_df.columns:
        sheet_df["group"] = sheet_df["group"].replace(GROUPS_RENAME)

    if "suitable_for_pp" in sheet_df.columns:
        sheet_df["suitable_for_pp"] = sheet_df["suitable_for_pp"].replace(SUITABLE_FOR_PP_RENAME)

    if "Clinic" in sheet_df.columns:
        sheet_df['Clinic'] = sheet_df['Clinic'].str.strip()

    sheet_columns = [col for col in columns if col in sheet_df.columns]
    return sheet_df[sheet_columns].reset_index(drop=True)


def _extract_patient_rows(xls: pd.ExcelFile, empty_tables: List[str]) -> pd.DataFrame:
    frames = []
    for sheet in xls.sheet_names:
        if sheet in empty_tables:
            continue
        frames.append(_normalize_sheet(xls.parse(sheet), sheet))
    return pd.concat(frames, ignore_index=True)


def _build_group_lookup(groups_xls: pd.ExcelFile) -> Dict[str, str]:
    frames = []
    for sheet in groups_xls.sheet_names:
        groups_df = groups_xls.parse(sheet)
        groups_df = groups_df.dropna(subset=["Participant code"])
        groups_df["clean_id"] = groups_df["Participant code"].astype(str).str.lower()
        groups_df["clean_id"] = groups_df["clean_id"].apply(_drop_trailing_s)
        groups_df = groups_df.rename(
            {"Assignment": "group", "Participant code": "raw_id"}, axis=1
        )
        groups_df["group"] = groups_df["group"].replace(GROUPS_RENAME)
        frames.append(groups_df[["clean_id", "group"]])

    lookup: Dict[str, str] = {}
    for _, row in pd.concat(frames).iterrows():
        lookup[row.clean_id] = row.group
    return lookup


def load_and_normalize_data(data_source=None) -> Tuple[pd.DataFrame, List[str]]:
    """Run preprocessing step 1: load workbooks, normalize columns, fill groups."""
    xls, groups_xls = _load_workbooks(data_source=data_source)

    empty_tables = []
    for sheet in xls.sheet_names:
        if xls.parse(sheet).shape[0] == 0:
            empty_tables.append(sheet)


    patient_df = _extract_patient_rows(xls, empty_tables)
    group_lookup = _build_group_lookup(groups_xls)

    patient_df["group"] = patient_df["group"].fillna(
        patient_df["clean_id"].map(group_lookup)
    )
    patient_df["group"] = patient_df["group"].replace(GROUPS_RENAME)
    return patient_df, empty_tables


# --------------------------------------------------------------------------- #
# Step 2 - Priority aggregation and date parsing
# --------------------------------------------------------------------------- #
def _parse_date(value):
    """Parse a single date value, returning pd.NaT if unparseable."""
    # Skip NaN values
    if pd.isna(value):
        return pd.NaT
    
    # Try each format
    for fmt in DATE_FORMATS:
        try:
            if isinstance(value, str):
                value = value.strip()
                # Skip empty strings after stripping
                if not value:
                    return pd.NaT
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def _parse_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns and validate that all values are either valid dates or NaN.
    Raises ValueError with details of invalid values if any are found.
    Empty strings are treated as valid (equivalent to NaN).
    """
    result = df.copy()
    date_columns = [col for col in result.columns if "date" in col]
    
    invalid_values_by_column = {}
    
    for date_col in date_columns:
        original_values = result[date_col].copy()
        parsed_values = result[date_col].apply(_parse_date)
        
        # Find values that were not NaN/empty originally but became NaT after parsing
        # Empty strings (after stripping) are considered valid, so we exclude them
        is_empty_string = original_values.apply(
            lambda x: isinstance(x, str) and not x.strip() if pd.notna(x) else False
        )
        invalid_mask = (
            original_values.notna() 
            & ~is_empty_string  # Exclude empty strings
            & parsed_values.isna()  # But became NaT after parsing
        )
        
        if invalid_mask.any():
            invalid_values = original_values[invalid_mask].unique()
            invalid_values_by_column[date_col] = invalid_values.tolist()
        
        result[date_col] = parsed_values
    
    # If any invalid values were found, raise an error with details
    if invalid_values_by_column:
        error_parts = ["Invalid date values found in the following columns:"]
        for col, invalid_vals in invalid_values_by_column.items():
            # Limit display to first 20 unique values per column to avoid huge error messages
            display_vals = invalid_vals[:20]
            more_count = len(invalid_vals) - 20
            vals_str = ", ".join(repr(str(v)) for v in display_vals)
            if more_count > 0:
                vals_str += f" ... and {more_count} more"
            error_parts.append(f"  - {col}: {vals_str}")
        
        error_parts.append("\nPlease fix these values in your data file. Dates must be parseable or empty/NaN.")
        raise ValueError("\n".join(error_parts))
    
    return result


def _aggregate_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["prio"] = temp["sheet"].map(PRIORITY_MAP)
    temp = temp.sort_values(["clean_id", "prio"])
    result = temp.groupby("clean_id", as_index=False).first()
    return result.drop(columns=["prio"])


def _augment_with_sheet_dummies(df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(temp_df["sheet"]).astype(int)
    dummies["clean_id"] = temp_df["clean_id"]
    one_hot = dummies.groupby("clean_id", as_index=True).max().reset_index()

    enriched = df.merge(one_hot, on="clean_id", how="left")
    enriched[dummies.columns] = enriched[dummies.columns].fillna(0)
    return enriched


def aggregate_patient_records(df: pd.DataFrame) -> pd.DataFrame:
    """Run preprocessing step 2."""
    parsed = _parse_date_columns(df)
    aggregated = _aggregate_by_priority(parsed)
    enriched = _augment_with_sheet_dummies(aggregated, parsed)

    enriched["first_contact_date"] = (
        enriched["intake_date"]
        .fillna(enriched["clinic_application_date"])
        .fillna(enriched["signing_date"])
    )
    enriched["therapy_starting_date"] = enriched["therapy_start_date"]



    return enriched


# --------------------------------------------------------------------------- #
# Step 3 - CONSORT logic and derived metrics
# --------------------------------------------------------------------------- #
def _create_consort_rules(empty_tables: List[str]) -> Dict[str, Dict[str, List[str]]]:
    rules = {
        "N": {
            "isin": [
                "CAU",
                "IPC-SSC",
                "אי הסכמה למחקר",
                "אי התאמה למחקר",
                'אין שת"פ טיפולי',
                "משתתפים פעילים",
                "נשירה מחקרית",
                "נשירה קלינית- לאחר ת. טיפול",
                "סיימו טיפול",
                "עלייה לרמה 2",
                "פספוסי גיוסים",
            ],
            "not_in": [],
        },
        "Eligible": {
            "isin": [
                "CAU",
                "IPC-SSC",
                "אי הסכמה למחקר",
                'אין שת"פ טיפולי',
                "משתתפים פעילים",
                "נשירה מחקרית",
                "נשירה קלינית- לאחר ת. טיפול",
                "סיימו טיפול",
                "עלייה לרמה 2",
            ],
            "not_in": [],
        },
        "Randomized": {
            "isin": [
                "CAU",
                "IPC-SSC",
                'אין שת"פ טיפולי',
                "משתתפים פעילים",
                "נשירה מחקרית",
                "נשירה קלינית- לאחר ת. טיפול",
                "סיימו טיפול",
                "עלייה לרמה 2",
            ],
            "not_in": [],
        },
        "Research Dropout": {
            "isin": ["נשירה מחקרית"],
            "not_in": [],
        },
        "Clinical Dropout": {
            "isin": ["נשירה קלינית- לאחר ת. טיפול"],
            "not_in": [],
        },
        "In Waiting List": {
            "isin": ["משתתפים פעילים", "נשירה מחקרית"],
            "not_in": ["CAU", "IPC-SSC"] + EXCLUDE_SHEETS,
        },
        "Not In Waiting List": {
            "isin": ["משתתפים פעילים", "נשירה מחקרית"],
            "not_in": ["CAU", "IPC-SSCx"] + EXCLUDE_SHEETS,
        },
        "Finished": {"isin": ["סיימו טיפול"],
                     "not_in": []
        },
        "Active": {
            "isin": ["משתתפים פעילים", "CAU", "IPC-SSC"],
            "not_in": EXCLUDE_SHEETS,
        },
        "Not Cooperative": {'isin': ['אין שת"פ טיפולי'], "not_in": []},

    }
    for sheet in rules["N"]["isin"]:
        if sheet not in ["CAU", "IPC-SSC"]:
            rules[f"{sheet}__טבלת"] = {'isin': [sheet], "not_in": []}
            CONSORT_GROUPS.append(f"{sheet}__טבלת")


    for key, rule in rules.items():
        for bucket in ("isin", "not_in"):
            rules[key][bucket] = [
                value for value in rule[bucket] if value not in empty_tables
            ]
    return rules


def _apply_consort_rules(df: pd.DataFrame, empty_tables: List[str]) -> pd.DataFrame:
    rules = _create_consort_rules(empty_tables)
    result = df.copy()

    def isin_group(row: pd.Series, rule: Dict[str, List[str]]) -> bool:
        in_positive = any(bool(row[sheet]) for sheet in rule["isin"])
        in_negative = any(bool(row[sheet]) for sheet in rule["not_in"])
        return in_positive and not in_negative

    for group_name in CONSORT_GROUPS:
        result[group_name] = result.apply(isin_group, axis=1, args=(rules[group_name],))
    return result


def _to_bool(value):
    if pd.isna(value):
        return np.nan
    else:
        return bool(value)


def enrich_with_consort_metrics(df: pd.DataFrame, empty_tables: List[str]) -> pd.DataFrame:
    """Run preprocessing step 3."""
    enriched = _apply_consort_rules(df, empty_tables)

    start = "first_contact_date"
    end = "therapy_starting_date"


    enriched["waiting_duration"] = pd.to_datetime(enriched[end]) - pd.to_datetime(
        enriched[start]
    )
    enriched["waiting_duration"] = enriched["waiting_duration"].dt.days
    enriched = enriched[~(enriched["waiting_duration"] > MAX_WAITING_DAYS_DEFAULT)]

    enriched["did_started_therapy"] = pd.notna(enriched[end])
    if "suitable_for_pp" in enriched.columns:
        enriched["suitable_for_pp"] = enriched["suitable_for_pp"].apply(_to_bool)

    return enriched



# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_patient_dataset(data_source=None) -> pd.DataFrame:
    """Convenience function that runs all preprocessing stages."""
    normalized, empty_tables = load_and_normalize_data(data_source=data_source)
    aggregated = aggregate_patient_records(normalized)
    return enrich_with_consort_metrics(aggregated, empty_tables)

