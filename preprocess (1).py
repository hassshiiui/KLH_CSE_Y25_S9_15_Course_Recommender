"""Text preprocessing helpers for course content."""

import re

import pandas as pd


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/symbols and collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_content(df: pd.DataFrame) -> pd.Series:
    """Combine the text columns into one 'content' string per course.

    The title, category and skills are repeated so they carry more weight
    than the longer description in the TF-IDF vectors.
    """
    skills = df["skills"].str.replace(";", " ", regex=False)
    combined = (
        (df["title"] + " ") * 2
        + (df["category"] + " ")
        + (skills + " ") * 2
        + df["description"]
    )
    return combined.apply(clean_text)
