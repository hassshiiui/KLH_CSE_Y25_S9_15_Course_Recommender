"""Command-line interface for the course recommender."""

import argparse
import sys

from src.data_loader import load_courses
from src.recommender import CourseRecommender

DEFAULT_DATA = "data/courses.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Course recommender")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--course", help="Recommend courses similar to this course title")
    group.add_argument("--interests", help="Recommend courses for these interests/skills")
    parser.add_argument("--top-n", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--level", choices=["Beginner", "Intermediate", "Advanced"], help="Filter by level")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Path to courses CSV")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        recommender = CourseRecommender(load_courses(args.data))
        if args.course:
            results = recommender.recommend_by_course(args.course, args.top_n, args.level)
        else:
            results = recommender.recommend_by_interests(args.interests, args.top_n, args.level)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if results.empty:
        print("No matching courses found.")
    else:
        print(results.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
