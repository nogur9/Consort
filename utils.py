import pandas as pd
import numpy as np
from typing import List
from consts import *



def truthy_mask(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    # create a mask of rows where ANY of the selected one-hot sheet columns are truthy (1, True, '1')
    masks = [(df[col] == 1) for col in cols]
    combined = masks[0]
    for m in masks[1:]:
        combined = combined | m
    return combined


def waiting_stats(s: pd.Series) -> dict:

    s_num = pd.to_numeric(s, errors='coerce')
    return {
        'count_non_missing': int(s_num.notna().sum()),
        'missing': int(s_num.isna().sum()),
        'mean': float(s_num.mean()) if s_num.notna().any() else None,
        'std': float(s_num.std()) if s_num.notna().any() else None,
        'min': float(s_num.min()) if s_num.notna().any() else None,
        'max': float(s_num.max()) if s_num.notna().any() else None,
        'in_waiting_list': int((s == 'in waiting list').sum())
    }


def format_stat_row(d: dict) -> dict:
    # make printable with rounding
    return {
        'non_missing': d['count_non_missing'],
        'missing': d['missing'],
        'mean': round(d['mean'], 2) if d['mean'] is not None else '-',
        'std': round(d['std'], 2) if d['std'] is not None else '-',
        'min': round(d['min'], 2) if d['min'] is not None else '-',
        'max': round(d['max'], 2) if d['max'] is not None else '-',
    }


def group_value_counts(df_sel: pd.DataFrame, group_cols: List[str]):

    a, b = group_cols[0], group_cols[1]
    A = ~df_sel[a].fillna(0).astype(str).str.strip().isin(['0', '', 'nan', 'None'])
    B = ~df_sel[b].fillna(0).astype(str).str.strip().isin(['0', '', 'nan', 'None'])
    combos = pd.Series(
        np.select([A & B, A & ~B, ~A & B, ~A & ~B],
                  [f"both: {a}+{b}", f"only {a}", f"only {b}", 'neither']),
        index=df_sel.index
    )
    combo_counts = combos.value_counts().rename_axis('group_combo').reset_index(name='n')
    return combo_counts