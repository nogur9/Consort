"""Centralized configuration for the CONSORT Streamlit app."""

from pathlib import Path
from typing import List

import numpy as np

# Base paths -----------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"


def _build_candidates(rel_paths: List[str]) -> List[Path]:
    """Create absolute candidate paths for a given set of relative strings."""
    candidates: List[Path] = []
    for rel_path in rel_paths:
        candidates.append((PROJECT_ROOT / rel_path).resolve())
        candidates.append((BASE_DIR / rel_path).resolve())
    # remove duplicates while preserving order
    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


DATA_FILE =  "data/anon_master.xlsx"


GROUPS_FILE ="data/טבלת הקצאה רנדומלית לתנאי הניסוי והבקרה.xlsx"

# Domain constants ------------------------------------------------------------
DEFAULT_ARMS = ["CAU", "Stepped Care", "Missing Group", 'טיפול בקבוצת הורים ']
CONSORT_GROUPS = [
    "N",
    "Eligible",
    "Randomized",
    "Dropout",
    "Research Dropout",
    "Clinical Dropout",
    "In Waiting List",
    "Finished",
    "Active",
    "Not Cooperative"
]

MAX_WAITING_DAYS_DEFAULT = 3650

ALIASES = {
    "raw_id": [";", "קוד", "קוד מטופל", "קוד נבדק"],
    "clinic_application_date": ["תאריך פנייה למרפאה"],
    "intake_date": ["תאריך אינטייק"],
    "group": ["הקצאה רנדומלית", "סוג טיפול"],
    "signing_date": ["תאריך חתימה"],
    "therapy_start_date": ["תאריך תחילת התערבות"],
    "therapy_end_date": [
        "שאלוני סוף התערבות (8 שבועות)",
        "תאריך סיום התערבות",
    ],
    "suitable_for_pp": [
        'מתאים לתומך ע"י קלינאי 1. כן 0. לא',
        'מתאימים לתומך? 1.כן 0.לא',
        'מתאימים לתומך? 1.כן 0.לא',
    ],
    "First name": ["שם פרטי", "שם פרטי של המטופל"],
    "Last name": ["שם משפחה", "שם משפחה של המטופל"],
    "Clinic": ['מרפאה']

}




SUITABLE_FOR_PP_RENAME = {
    '1': 1,
    '0': 0,
    'ללא אובדנות': np.nan,
    'לא- הומלץ רק שאלונים': 0,
    'רק שאלונים': np.nan,
    'מעקבים': np.nan,
    'ל.ר. ': np.nan,
    'ל.ר.': np.nan,
    'ל.ר': np.nan
}

GROUPS_RENAME = {
    "IPC": "Stepped Care",
    "Stepped care": "Stepped Care",
    "ARM 1= Stepped care": "Stepped Care",
    "ARM 2=CAU": "CAU",
    "ARM 2= CAU": "CAU",
}

DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%b-%Y",
    "%d.%m.%Y",
    "%Y-%m-%d %H:%M:%S",
    "%d.%m.%y",
    "%Y-%m-%d %H:%M:%S",
]

PRIORITY_MAP = {
    "סיימו טיפול": 0,
    "CAU": 1,
    "IPC-SSC": 2,
    "משתתפים פעילים": 3,
    "פספוסי גיוסים": 4,
    'אין שת"פ טיפולי': 5,
    "אי התאמה למחקר": 6,
    "אי הסכמה למחקר": 7,
    "נשירה מחקרית": 8,
    "נשירה קלינית- לאחר ת. טיפול": 9,
    "עלייה לרמה 2": 10,
}

EXCLUDE_SHEETS = [
    "נשירה מחקרית",
    "נשירה קלינית- לאחר ת. טיפול",
    'אין שת"פ טיפולי',
    "סיימו טיפול",
    "אי הסכמה למחקר",
    "אי התאמה למחקר",
    "פספוסי גיוסים",
]

METRIC_TABS = [
    {
        "title": "Count",
        "column": "count",
        "label": "Patient Count",
        "description": "Total number of patients in each arm.",
    },
    {
        "title": "Waiting Duration",
        "column": "waiting_duration_mean",
        "label": "Mean Waiting Duration (days)",
        "description": "Average waiting time until therapy start.",
    },
    {
        "title": "Started Therapy",
        "column": "did_started_therapy_mean",
        "label": "Proportion Who Started Therapy",
        "description": "Share of patients who already began therapy.",
    },
    {
        "title": "Suitable for PP",
        "column": "suitable_for_pp",
        "label": "Proportion Suitable for PP",
        "description": "Share of patients suitable for PP treatment.",
    },
]

