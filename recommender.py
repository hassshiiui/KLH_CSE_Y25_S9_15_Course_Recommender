"""Content-based course recommender (TF-IDF + cosine similarity)."""

from difflib import get_close_matches

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import build_content, clean_text

OUTPUT_COLUMNS = ["title", "category", "level", "score"]


class CourseRecommender:
    """Recommend courses similar to a course or to free-text interests."""

    def __init__(self, courses: pd.DataFrame):
        self.courses = courses.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(build_content(self.courses))

    def _format(self, scores, exclude_idx=None, top_n=5, level=None) -> pd.DataFrame:
        result = self.courses.copy()
        result["score"] = scores.round(3)

        if exclude_idx is not None:
            result = result.drop(index=exclude_idx)
        if level:
            result = result[result["level"].str.lower() == level.lower()]

        result = result[result["score"] > 0]
        result = result.sort_values("score", ascending=False).head(top_n)
        return result[OUTPUT_COLUMNS].reset_index(drop=True)

    def find_course(self, title: str) -> int:
        """Return the row index of a course title (case-insensitive, fuzzy).

        Raises:
            ValueError: if no course title is close enough.
        """
        titles = self.courses["title"].tolist()
        lookup = {t.lower(): i for i, t in enumerate(titles)}

        key = title.strip().lower()
        if key in lookup:
            return lookup[key]

        close = get_close_matches(key, list(lookup), n=3, cutoff=0.6)
        if close:
            return lookup[close[0]]

        raise ValueError(f"Course not found: '{title}'")

    def recommend_by_course(self, title: str, top_n: int = 5, level: str | None = None) -> pd.DataFrame:
        """Courses most similar to the given course title."""
        idx = self.find_course(title)
        scores = cosine_similarity(self.matrix[idx], self.matrix).ravel()
        return self._format(scores, exclude_idx=idx, top_n=top_n, level=level)

    def recommend_by_interests(self, interests: str, top_n: int = 5, level: str | None = None) -> pd.DataFrame:
        """Courses that best match free-text interests or skills."""
        if not interests or not interests.strip():
            raise ValueError("Interests text must not be empty.")
        query_vec = self.vectorizer.transform([clean_text(interests)])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        return self._format(scores, top_n=top_n, level=level)
