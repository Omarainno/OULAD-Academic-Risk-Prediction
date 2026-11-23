from pathlib import Path
import io, zipfile, requests
import pandas as pd

RAW_FILES = [
    "courses.csv", "assessments.csv", "studentAssessment.csv",
    "studentInfo.csv", "studentRegistration.csv", "vle.csv", "studentVle.csv"
]

# UCI public distribution endpoint for dataset 349.
UCI_URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics.zip"

def raw_ready(raw_dir: Path):
    return all((raw_dir / f).exists() for f in RAW_FILES)

def download_oulad(raw_dir: Path, timeout=120):
    raw_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(UCI_URL, timeout=timeout)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        members = {Path(n).name: n for n in z.namelist()}
        missing = [f for f in RAW_FILES if f not in members]
        if missing:
            raise RuntimeError(f"Downloaded archive does not contain expected files: {missing}")
        for f in RAW_FILES:
            with z.open(members[f]) as src, open(raw_dir / f, "wb") as dst:
                dst.write(src.read())

def load_oulad(raw_dir: Path):
    if not raw_ready(raw_dir):
        raise FileNotFoundError(
            "OULAD CSV files are missing. Run with automatic download enabled or place "
            "the seven CSV files in data/raw/."
        )
    return {f[:-4]: pd.read_csv(raw_dir / f) for f in RAW_FILES}

def validate_schema(d):
    expected = {
        "courses": {"code_module", "code_presentation", "module_presentation_length"},
        "assessments": {"code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"},
        "studentAssessment": {"id_assessment", "id_student", "date_submitted", "is_banked", "score"},
        "studentInfo": {"code_module", "code_presentation", "id_student", "final_result"},
        "studentRegistration": {"code_module", "code_presentation", "id_student", "date_registration"},
        "vle": {"id_site", "code_module", "code_presentation", "activity_type"},
        "studentVle": {"code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"},
    }
    for name, cols in expected.items():
        missing = cols - set(d[name].columns)
        if missing:
            raise ValueError(f"{name}.csv is missing required columns: {sorted(missing)}")
