# CONSORT Analysis Streamlit Dashboard

A comprehensive Streamlit application for analyzing CONSORT data with descriptive statistics, visualizations, and survival analysis.

## Features

- **Data Preprocessing**: Automatically performs all preprocessing steps from the three Jupyter notebooks
- **CONSORT Group Analysis**: Analyze data filtered by different CONSORT groups
- **Descriptive Statistics**: View summary statistics for:
  - Patient count
  - Mean waiting duration
  - Proportion who started therapy
  - Proportion suitable for PP
- **Visualizations**: Interactive charts, including cumulative incidence with risk tables
- **Raw Data Explorer**: View patient-level rows grouped by treatment arm
- **Event Analysis**: Cumulative incidence (probability of event over time) replaces the prior Kaplan-Meier view
- **Filtering Options**:
  - Filter by CONSORT group
  - Filter by treatment groups (multi-select)
  - Filter by maximum waiting days
  - Filter by minimum intake date

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure your data files are in the `../data/` directory:
   - `‏‏של __מאסטר מעקב תורים iIPC_IPT - עותק.xlsx`
   - `טבלת הקצאה רנדומלית לתנאי הניסוי והבקרה.xlsx`

2. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

3. The app will open in your browser automatically.

## Project Structure

```
new_consort_analysis/
├── app_config.py          # Shared constants and configuration
├── data_pipeline.py       # Data loading + preprocessing logic
├── visualizations.py      # Plotting and aggregation helpers
├── streamlit_app.py       # Streamlit UI (imports the modules above)
├── requirements.txt
└── README.md
```

## Data Processing Pipeline

The data layer mirrors the original notebooks and now lives in `data_pipeline.py`:

1. **Load & Normalize**: read Excel files, harmonize column names, normalize IDs, and impute groups
2. **Aggregate by Patient**: parse date columns, prioritize sheets, and build one row per patient (with sheet dummies)
3. **Enrich**: compute CONSORT group flags, waiting duration, therapy start flags, and clean booleans

## CONSORT Groups

The app supports analysis for the following CONSORT groups:
- N (Total enrolled)
- Eligible
- Randomized
- Dropout
- Research Dropout
- Clinical Dropout
- In Waiting List
- Finished
- Active
- Not Cooperative

## Notes

- Data is cached for performance
- All preprocessing steps are performed automatically
- The app filters out patients with waiting duration > 670 days by default (configurable)



