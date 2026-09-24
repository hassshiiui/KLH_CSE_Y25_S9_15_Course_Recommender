"""Load and validate the course dataset."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "course_id",
    "title",
    "category",
    "level",
    "description",
    "skills",
}


def load_courses(path: str | Path) -> pd.DataFrame:
    """Read the courses CSV and return a cleaned DataFrame.

    Raises:
        FileNotFoundError: if the CSV does not exist.
        ValueError: if required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.drop_duplicates(subset="title").copy()
    for col in ["title", "category", "level", "description", "skills"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    return df.reset_index(drop=True)
