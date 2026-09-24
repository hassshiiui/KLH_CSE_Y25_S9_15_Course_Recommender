"""Unit tests (run with `pytest` or `python -m unittest`)."""

import unittest

import pandas as pd

from src.data_loader import load_courses
from src.preprocess import build_content, clean_text
from src.recommender import CourseRecommender

DATA_PATH = "data/courses.csv"


class TestPreprocess(unittest.TestCase):
    def test_clean_text_lowercases_and_strips_symbols(self):
        self.assertEqual(clean_text("  Hello,   WORLD!! C++ "), "hello world c")

    def test_build_content_returns_one_row_per_course(self):
        df = load_courses(DATA_PATH)
        self.assertEqual(len(build_content(df)), len(df))


class TestDataLoader(unittest.TestCase):
    def test_load_courses(self):
        df = load_courses(DATA_PATH)
        self.assertGreater(len(df), 0)
        self.assertIn("title", df.columns)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_courses("data/does_not_exist.csv")


class TestRecommender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = CourseRecommender(load_courses(DATA_PATH))

    def test_recommend_by_course_excludes_itself(self):
        result = self.rec.recommend_by_course("Machine Learning Fundamentals", top_n=3)
        self.assertLessEqual(len(result), 3)
        self.assertNotIn("Machine Learning Fundamentals", result["title"].tolist())

    def test_similar_courses_are_related(self):
        result = self.rec.recommend_by_course("Deep Learning with TensorFlow", top_n=3)
        self.assertIn("Machine Learning", result["category"].tolist())

    def test_fuzzy_title_match(self):
        result = self.rec.recommend_by_course("python for beginer", top_n=2)
        self.assertFalse(result.empty)

    def test_unknown_course_raises(self):
        with self.assertRaises(ValueError):
            self.rec.recommend_by_course("Underwater Basket Weaving 101")

    def test_recommend_by_interests(self):
        result = self.rec.recommend_by_interests("neural networks and deep learning", top_n=3)
        self.assertIn("Deep Learning with TensorFlow", result["title"].tolist())

    def test_empty_interests_raises(self):
        with self.assertRaises(ValueError):
            self.rec.recommend_by_interests("   ")

    def test_level_filter(self):
        result = self.rec.recommend_by_interests("python", top_n=5, level="Beginner")
        self.assertTrue((result["level"] == "Beginner").all())


if __name__ == "__main__":
    unittest.main()
