"""Characterisation snapshot of the recommendation engine's output.

tests/test_engine_snapshot.py compares the engine against the stored baseline. After an
intentional change to matching, ranking or formatting, regenerate it with:

    python tests/snapshot_engine.py --update
"""

import argparse
import difflib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE_PATH = REPO / "tests" / "snapshots" / "engine_baseline.json"

SIMILARITY_QUERIES = (
    "Avatar",
    "The Dark Knight",
    "Inception",
    "Batman",
    "inceptoin",
    "Toy Story",
    "Titanic",
    "The Godfather",
)
SIMILARITY_COUNT = 10

FEATURE_QUERIES = {
    "genres=Action": {"genres": ["Action"]},
    "genres=Drama,Romance": {"genres": ["Drama", "Romance"]},
    "director=Christopher Nolan": {"director_query": "Christopher Nolan"},
    "years=2000-2010": {"min_year": "2000", "max_year": "2010"},
    "genres=Comedy,min_rating=7": {"genres": ["Comedy"], "min_rating": "7"},
}
FEATURE_COUNT = 15

DETAIL_MOVIE_IDS = (19995, 155, 268, 2661)


def _ranked(results):
    return [[int(movie["id"]), movie["title"]] for movie in results]


def _details(movie):
    if movie is None:
        return None
    return {
        "title": movie["title"],
        "director": movie["director"],
        "genres": movie["genres"],
        "runtime_formatted": movie["runtime_formatted"],
        "genre_display_names": [genre["display_name"] for genre in movie["genre_badges"]],
        "similar_movie_ids": [int(similar["id"]) for similar in movie["similar_movies"]],
    }


def build_snapshot(engine, details):
    """Collect the engine outputs pinned by the baseline into a JSON-serialisable dict."""
    similarity = {}
    for query in SIMILARITY_QUERIES:
        results, warnings = engine.recommend_by_similarity(query, num=SIMILARITY_COUNT)
        similarity[query] = {"results": _ranked(results), "warnings": list(warnings)}

    features = {}
    for label, criteria in FEATURE_QUERIES.items():
        results, warnings = engine.recommend_by_features(**criteria, num=FEATURE_COUNT)
        features[label] = {"results": _ranked(results), "warnings": list(warnings)}

    # The baseline holds the Turkish details.
    detail_pages = {
        str(movie_id): _details(details.get_movie_details(movie_id, "tr"))
        for movie_id in DETAIL_MOVIE_IDS
    }

    catalog = {
        "all_genres": list(engine.all_genres),
        "movie_title_count": len(engine.all_movie_titles),
        "director_name_count": len(engine.all_director_names),
        "popular_movies_top10": [card["title"] for card in engine.popular_cards(10)],
    }

    return {
        "similarity": similarity,
        "features": features,
        "details": detail_pages,
        "catalog": catalog,
    }


def to_json(snapshot):
    return json.dumps(snapshot, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def diff_against_baseline(snapshot):
    """Return a unified diff from the stored baseline to ``snapshot``; empty when they match."""
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    return "".join(
        difflib.unified_diff(
            to_json(baseline).splitlines(keepends=True),
            to_json(snapshot).splitlines(keepends=True),
            fromfile="baseline",
            tofile="current",
        )
    )


def _load():
    # Same offline conditions as the test suite: with an empty key the engine never calls TMDb.
    os.environ["TMDB_API_KEY"] = ""
    sys.path.insert(0, str(REPO))
    from movierec.engine import RecommendationEngine
    from movierec.movie_details import MovieDetails

    dataset = REPO / "dataset"
    engine = RecommendationEngine(
        str(dataset / "tmdb_5000_movies.csv"), str(dataset / "tmdb_5000_credits.csv")
    )
    return engine, MovieDetails(engine)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--update", action="store_true", help="overwrite the baseline with the current output"
    )
    args = parser.parse_args(argv)

    snapshot = build_snapshot(*_load())
    if args.update:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(to_json(snapshot), encoding="utf-8")
        print(f"wrote {BASELINE_PATH.relative_to(REPO)}")
        return 0

    diff = diff_against_baseline(snapshot)
    if diff:
        sys.stdout.write(diff)
        return 1
    print("engine output matches the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
