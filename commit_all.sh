#!/usr/bin/env bash
# Creates the whole commit history for the course-recommender project.
# Usage (from INSIDE the course-recommender folder):
#     bash ../commit_all.sh
# Optional: push automatically
#     REPO_URL=https://github.com/<you>/course-recommender.git bash ../commit_all.sh
set -e

git init -q
git branch -M main

# Keep the full README aside; commit a short one first, full one at the end.
cp README.md .README_full.tmp
cat > README.md << 'EOT'
# Course Recommender

A content-based course recommendation system that suggests courses using
TF-IDF vectors and cosine similarity.

## Goals

- Recommend courses similar to a course the user liked
- Recommend courses from free-text interests or skills
EOT

git add .gitignore src/__init__.py tests/__init__.py
git commit -q -m "chore: initialize project structure and .gitignore"

git add README.md
git commit -q -m "docs: add README with project overview and goals"

git add requirements.txt
git commit -q -m "chore: add requirements.txt with core dependencies"

git add data/courses.csv
git commit -q -m "data: add sample course dataset (courses.csv)"

git add src/data_loader.py
git commit -q -m "feat: add data loading and validation module"

git add src/preprocess.py
git commit -q -m "feat: add text preprocessing for course content"

git add src/recommender.py
git commit -q -m "feat: implement TF-IDF and cosine similarity recommender"

git add main.py
git commit -q -m "feat: add command-line interface"

git add tests/test_recommender.py
git commit -q -m "test: add unit tests for loader, preprocessing and recommender"

mv .README_full.tmp README.md
git add README.md
git commit -q -m "docs: add setup, usage and project structure to README"

echo "Done. Commit history:"
git log --oneline

if [ -n "$REPO_URL" ]; then
  git remote add origin "$REPO_URL"
  git push -u origin main
fi
