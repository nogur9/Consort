"""Data loading and preprocessing utilities for the CONSORT dashboard."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import streamlit as st

import numpy as np
import pandas as pd

from app_config import (
    ALIASES,
    CONSORT_GROUPS,
    DATE_FORMATS,
    CONSORT_RULES,
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
    
    # Check for missing IDs and display row numbers
    missing_id_mask = sheet_df["raw_id"].isna()
    if missing_id_mask.any():
        missing_rows = sheet_df[missing_id_mask].index.tolist()
        # Add 2 to account for 0-based index and Excel header row (Excel rows start at 1, header is row 1)
        excel_rows = [idx + 2 for idx in missing_rows]
        raise ValueError(
            f"In the table '{sheet_name}', there are missing IDs in rows: {excel_rows}"
        )
    
    sheet_df["clean_id"] = sheet_df["raw_id"].astype(str).apply(_drop_trailing_s)
    
    # Check for non-unique clean_id within the same sheet
    duplicates = sheet_df[sheet_df["clean_id"].duplicated(keep=False)]
    if not duplicates.empty:
        duplicate_ids = duplicates["clean_id"].unique()
        error_parts = [f"In the table '{sheet_name}', there are multiple rows with the same ID value:"]
        for dup_id in duplicate_ids:
            dup_rows = sheet_df[sheet_df["clean_id"] == dup_id].index.tolist()
            excel_rows = [idx + 2 for idx in dup_rows]
            error_parts.append(
                f"  The ID '{dup_id}' appears in rows: {excel_rows}"
            )
        raise ValueError("\n".join(error_parts))

    if "group" in sheet_df.columns:
        sheet_df["group"] = sheet_df["group"].astype(str).replace(GROUPS_RENAME)
    elif sheet_name == "CAU":
        sheet_df["group"] = "CAU"
    elif sheet_name == "IPC-SSC":
        sheet_df["group"] = "Stepped Care"
    else:
        sheet_df["group"] =  pd.NA

    if "suitable_for_pp" in sheet_df.columns:
        sheet_df["suitable_for_pp"] = sheet_df["suitable_for_pp"].astype(str).replace(SUITABLE_FOR_PP_RENAME)
        print(f"_normalize_sheet {sheet_name = } meow {sheet_df.suitable_for_pp.unique() = }")
    else:
        sheet_df["suitable_for_pp"] = pd.NA

    if "Clinic" in sheet_df.columns:
        #print("Clinic", f"{sheet_df['Clinic'].dtype = }", f"{sheet_name = }")
        sheet_df['Clinic'] = sheet_df['Clinic'].astype(str).str.strip()
    else:
        sheet_df["Clinic"] =  pd.NA

    sheet_columns = [col for col in columns if col in sheet_df.columns]
    return sheet_df[sheet_columns].reset_index(drop=True)


def _extract_patient_rows(xls: pd.ExcelFile, empty_tables: List[str]) -> pd.DataFrame:
    frames = []
    for sheet in xls.sheet_names:
        if sheet in empty_tables:
            # print(f"{empty_tables = }")
            continue
        sheet_df = _normalize_sheet(xls.parse(sheet), sheet)
        frames.append(sheet_df)
        st.write(f"{str(sheet) = }\n{sheet_df.columns}")
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
    invalid_rows_by_column = {}
    
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
            invalid_rows = result[invalid_mask].index.tolist()
            # Add 2 to account for 0-based index and Excel header row (Excel rows start at 1, header is row 1)
            excel_rows = [idx + 2 for idx in invalid_rows]
            invalid_values = original_values[invalid_mask].unique()
            invalid_values_by_column[date_col] = invalid_values.tolist()
            invalid_rows_by_column[date_col] = {
                'excel_rows': excel_rows,
                'df_indices': invalid_rows
            }
        
        result[date_col] = parsed_values

    # If any invalid values were found, raise an error with details
    if invalid_values_by_column:
        error_parts = ["There are invalid date values that cannot be read in the following columns:"]
        for col, invalid_vals in invalid_values_by_column.items():
            # Limit display to first 20 unique values per column to avoid huge error messages
            display_vals = invalid_vals[:20]
            more_count = len(invalid_vals) - 20
            vals_str = ", ".join(repr(str(v)) for v in display_vals)
            if more_count > 0:
                vals_str += f" ... and {more_count} more"
            
            # Add row information
            row_info = invalid_rows_by_column[col]
            display_rows = row_info['excel_rows'][:20]
            excel_rows_str = str(display_rows)
            if len(row_info['excel_rows']) > 20:
                excel_rows_str = excel_rows_str[:-1] + f", ... ({len(row_info['excel_rows'])} total rows)]"
            
            error_parts.append(
                f"  In column '{col}': invalid values {vals_str} appear in rows: {excel_rows_str}"
            )
        
        error_parts.append("\nPlease fix these values in your data file. Dates should be in a standard format (like YYYY-MM-DD or DD/MM/YYYY) or left empty.")
        raise ValueError("\n".join(error_parts))
    
    return result


def _aggregate_by_priority(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["prio"] = temp["sheet"].map(PRIORITY_MAP)

    s = (
        df
        .dropna(subset=["group"])
        .groupby('clean_id')["group"]
        .nunique()
    )
    bad_ids = s[s != 1]
    if not bad_ids.empty:
        error_parts = ["The following IDs appear in multiple tables with different group assignments:"]
        for bad_id in bad_ids.index:
            # Find all rows with this ID and their groups/sheets
            id_rows = df[df["clean_id"] == bad_id]
            groups_info = id_rows[["sheet", "group"]].drop_duplicates()
            group_list = groups_info.groupby("group")["sheet"].apply(list).to_dict()
            group_str = ", ".join(f"group '{grp}' in tables {sheets}" for grp, sheets in group_list.items())
            error_parts.append(f"  The ID '{bad_id}' appears with {group_str}")
        raise ValueError("\n".join(error_parts))


    temp = temp.sort_values(["clean_id", "prio"])


    result = temp.groupby("clean_id", as_index=False).first(skipna=False)
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

    enriched['Clinic'] = enriched.Clinic.replace({"nan": np.nan})

    if enriched.first_contact_date.isna().any():
        missing_mask = enriched.first_contact_date.isna()
        missing_rows = enriched[missing_mask]
        row_numbers = [idx + 1 for idx in missing_rows.index.tolist()]
        raw_ids = missing_rows.raw_id.to_list()
        error_msg = (
            f"There are {len(raw_ids)} records missing an intake date.\n"
            f"These records are in rows: {row_numbers}\n"
            f"Record IDs: {raw_ids}"
        )
        raise ValueError(error_msg)

    return enriched


# --------------------------------------------------------------------------- #
# Step 3 - CONSORT logic and derived metrics
# --------------------------------------------------------------------------- #
def _create_consort_rules(empty_tables: List[str]) -> Dict[str, Dict[str, List[str]]]:
    rules = CONSORT_RULES.copy()
    for sheet in rules["N"]["isin"]:
        #if sheet not in ["CAU", "IPC-SSC"]:
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
        return True
        # in_positive = any(bool(row[sheet]) for sheet in rule["isin"])
        # in_negative = any(bool(row[sheet]) for sheet in rule["not_in"])
        # return in_positive and not in_negative

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
    enriched["did_started_therapy"] = enriched[end].notna().astype(int)

    # End date = started therapy date if started, today if still waiting
    enriched[end] = pd.to_datetime(enriched[end]).fillna(pd.Timestamp.today())


    enriched["waiting_duration"] = pd.to_datetime(enriched[end]) - pd.to_datetime(
        enriched[start]
    )
    enriched["waiting_duration"] = enriched["waiting_duration"].dt.days
    enriched = enriched[~(enriched["waiting_duration"] > MAX_WAITING_DAYS_DEFAULT)]


    if "suitable_for_pp" in enriched.columns:
        enriched["suitable_for_pp"] = enriched["suitable_for_pp"].apply(_to_bool)



    if (enriched["waiting_duration"] < 0).any():
        negative_mask = enriched["waiting_duration"] < 0
        negative_rows = enriched[negative_mask]
        row_numbers = [idx + 1 for idx in negative_rows.index.tolist()]
        raw_ids = negative_rows.raw_id.to_list()
        durations = negative_rows.waiting_duration.tolist()
        error_msg = (
            f"There are {len(raw_ids)} records where the therapy start date is before the intake date.\n"
            f"These records are in rows: {row_numbers}\n"
            f"Record IDs: {raw_ids}\n"
            f"Waiting durations (days): {durations}"
        )
        raise ValueError(error_msg)

    return enriched



# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_patient_dataset(data_source=None) -> pd.DataFrame:
    """Convenience function that runs all preprocessing stages."""
    normalized, empty_tables = load_and_normalize_data(data_source=data_source)
    aggregated = aggregate_patient_records(normalized)
    return enrich_with_consort_metrics(aggregated, empty_tables)

