"""Offline evaluation of the similar-movies lists, with every film in the dataset as a seed.

Each method recommends K = DETAIL_SIMILAR_COUNT films per seed, and every column is taken
over all (seed, recommendation) pairs:

- Genre overlap: mean Jaccard index |A & B| / |A | B| between the seed's genre set and the
  recommendation's; seeds without genres are skipped.
- Same director: share of recommendations directed by one of the seed's directors; seeds
  without a director count as no match.
- Catalog coverage: distinct recommended films / all films.
- Median votes: median vote_count of the recommendations, repeats included (popularity bias).

    python scripts/evaluate.py
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
METHODS = ("Pure cosine", "MovieRec (hybrid)", "Most-voted baseline")


def jaccard(a: set, b: set) -> float:
    """|a & b| / |a | b|; 0 when both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def metrics(movies, recommendations: dict) -> dict:
    """The four columns for {seed row: recommended rows} over the ``movies`` DataFrame."""
    genres = movies["genre_list"].map(set).tolist()
    directors = movies["directors"].map(set).tolist()
    votes = movies["vote_count"].tolist()
    pairs = [(seed, rec) for seed, recs in recommendations.items() for rec in recs]
    return {
        "Genre overlap": np.mean([jaccard(genres[s], genres[r]) for s, r in pairs if genres[s]]),
        "Same director": np.mean([bool(directors[s] & directors[r]) for s, r in pairs]),
        "Catalog coverage": len({rec for _, rec in pairs}) / len(movies),
        "Median votes": np.median([votes[rec] for _, rec in pairs]),
    }


def evaluate(engine) -> dict:
    """{method: metrics} with every dataset row as a seed."""
    from movierec.engine import DETAIL_SIMILAR_COUNT as count

    def pure_cosine(seed):
        scores = (engine.tfidf_matrix[seed] @ engine.tfidf_matrix.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")
        return order[order != seed][:count]

    most_voted = np.argsort(-engine.movies_df["vote_count"].to_numpy(), kind="stable")[: count + 1]
    rankers = (
        pure_cosine,
        lambda seed: engine.rank_neighbours(seed)[:count],
        lambda seed: most_voted[most_voted != seed][:count],
    )
    seeds = range(len(engine.movies_df))
    return {
        name: metrics(engine.movies_df, {seed: rank(seed).tolist() for seed in seeds})
        for name, rank in zip(METHODS, rankers, strict=True)
    }


def _load():
    # Same offline conditions as tests/snapshot_engine.py.
    os.environ["TMDB_API_KEY"] = ""
    sys.path.insert(0, str(REPO))
    from movierec.config import CREDITS_CSV, MOVIES_CSV
    from movierec.engine import RecommendationEngine

    return RecommendationEngine(MOVIES_CSV, CREDITS_CSV)


def main():
    engine = _load()
    print("| Method | Genre overlap | Same director | Catalog coverage | Median votes |")
    print("|---|---|---|---|---|")
    for name, row in evaluate(engine).items():
        print(
            f"| {name} | {row['Genre overlap']:.3f} | {row['Same director']:.1%} "
            f"| {row['Catalog coverage']:.1%} | {row['Median votes']:,.0f} |"
        )
    print(f"\nDataset median votes: {engine.movies_df['vote_count'].median():,.0f}")


if __name__ == "__main__":
    main()
