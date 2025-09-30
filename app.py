import streamlit as st
import pandas as pd
from consts import *
from utils import truthy_mask, group_value_counts, waiting_stats, format_stat_row

st.set_page_config(layout="wide", page_title="CONSORT - One-hot sheets EDA")
st.title("CONSORT-style EDA (one-hot sheets)")
st.markdown( "Upload a CSV (or use the example file) where each sheet/subgroup is a one-hot column, there are two group columns (one-hot) and a `waiting_time` column."
    "Select sheets to include and get unique-ID counts, group counts, waiting-time descriptives and the raw data used for the calculation. Click the stratify button to repeat the calculations per group."
)


# ------------------ UI ------------------

df = pd.read_excel(r"consort_data.xlsx")


st.sidebar.header("Selection")
selected_sheets = st.sidebar.multiselect("Select sheet (one-hot) columns to include:", options=SHEET_NAMES,
                                         default=SHEET_NAMES[6:8])



st.markdown("---")


# ------------------ Core computations ------------------

mask = truthy_mask(df, selected_sheets)
selected_df = df[mask].copy()


unique_ids_count = selected_df[ID_COL].nunique()

combo_df = group_value_counts(selected_df, GROUP_COLUMNS)


# Waiting stats overall
waiting_stats_overall = waiting_stats(selected_df[WAITING_COL])


# ------------------ Display results ------------------

st.header("Results for selected sheets")
col1, col2 = st.columns([1,1])
col1.metric("Unique IDs (selected rows)", unique_ids_count)

with col2:
    st.subheader("Group counts")
    if not combo_df.empty:
        st.markdown("**Combinations (first two group columns)**")
        st.table(combo_df)

    else:
        st.info("No group columns present in the selected data (or none of the expected group columns were found).")

st.subheader("Waiting time descriptives")
if waiting_stats_overall:
    st.table(pd.DataFrame([format_stat_row(waiting_stats_overall)], index=['selected_rows']))
else:
    st.info("No waiting time column detected in the selected data.")

st.dataframe(selected_df)

# ------------------ Step 2: Stratify button ------------------

st.write('---')
if st.button("Stratify calculations by group (per-group stats)"):
    present = [c for c in GROUP_COLUMNS if c in df.columns]
    if not present:
        st.error("No group columns specified or detected to stratify by.")
    else:
        st.header("Stratified results")
        bool_df = pd.DataFrame({c: ~df[c].fillna(0).astype(str).str.strip().isin(['0', '', 'nan', 'None']) for c in present})
        combo_keys = bool_df.astype(int).astype(str).agg('-'.join, axis=1)
        df['_combo_key'] = combo_keys
        combos = df['_combo_key'].unique()
        results = []
        for combo in combos:
            mask_combo = df['_combo_key'] == combo
            df_combo = df[mask_combo]
            df_combo_sel = df_combo[truthy_mask(df_combo, selected_sheets)]
            n_unique = int(df_combo_sel[ID_COL].nunique()) if ID_COL in df_combo_sel.columns else int(df_combo_sel.shape[0])
            stats = waiting_stats(df_combo_sel[WAITING_COL])
            in_waiting_list =  int((df_combo_sel[WAITING_COL] == "in waiting list").sum())

            results.append({
                'combo': combo,
                'unique_ids_in_selected_sheets': n_unique,
                'in_waiting_list': in_waiting_list,
                'waiting_non_missing': stats['count_non_missing'] if stats else None,
                'waiting_missing': stats['missing'] if stats else None,
                'waiting_mean': round(stats['mean'],2) if stats and stats['mean'] is not None else None,
                'waiting_std': round(stats['std'],2) if stats and stats['std'] is not None else None,
                'waiting_min': round(stats['min'],2) if stats and stats['min'] is not None else None,
                'waiting_max': round(stats['max'],2) if stats and stats['max'] is not None else None,
            })
        res_df = pd.DataFrame(results).sort_values('unique_ids_in_selected_sheets', ascending=False)
        st.table(res_df)

st.markdown("---")
st.markdown("""**Notes:**
- This file uses the constants at the top to match your exact column names. Edit them there if anything differs.
- The app treats any non-zero / non-empty value in the one-hot columns as 'present'. If your dataset uses different encodings (e.g., 'yes'/'no'), change the truthiness logic in `truthy_mask`.
""")
