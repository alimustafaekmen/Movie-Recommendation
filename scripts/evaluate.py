"""Offline evaluation of the similar-movies lists, with every film in the dataset as a seed.

Each method recommends K = DETAIL_SIMILAR_COUNT films per seed, and every column is taken
over all (seed, recommendation) pairs:

- Genre overlap: mean Jaccard index |A & B| / |A | B| between the seed's genre set and the
  recommendation's; seeds without genres are skipped.
- Same director: share of recommendations directed by one of the seed's directors; seeds
  without a director count as no match.
- Shared lead actor: share of recommendations whose first three billed actors include one
  of the seed's first three. Cast is not a model input.
- Series recall: per seed, the share of its series siblings (films with the same title
  stem, a different title and at least 100 votes) found in its list, out of at most K;
  averaged over the seeds that have siblings. About 1 stem group in 10 is not a series.
- Catalog coverage: distinct recommended films / all films.
- Median votes: median vote_count of the recommendations, repeats included (popularity bias).

    python scripts/evaluate.py
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
METHODS = ("Pure cosine", "MovieRec (hybrid)", "Most-voted baseline")
LEAD_ACTORS = 3
SERIES_MIN_VOTES = 100  # Same floor as MIN_VOTES_THRESHOLD: lists rarely show films below it.
SEQUEL_MARKS = {"ii", "iii", "iv", "v", "vi", "vii", "viii", "part"}
# Stems with these words are too generic to name a series ("Pirates of", "The Man in").
STEM_STOP_WORDS = {"the", "a", "an", "of", "and", "in", "to", "on", "my", "for", "with", "at"}
STEM_STOP_WORDS |= {"me", "i", "you", "is"}


def jaccard(a: set, b: set) -> float:
    """|a & b| / |a | b|; 0 when both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def title_stem(title: str) -> str | None:
    """Series key of a title: "Star Trek II: The Wrath of Khan" -> "star trek"; None if generic."""
    words = re.findall(r"[a-z0-9']+", title.lower().split(":")[0])
    if words and words[0] in ("the", "a", "an"):
        words = words[1:]
    while words and (words[-1] in SEQUEL_MARKS or words[-1].isdigit()):
        words.pop()
    words = words[:2]
    if not words or STEM_STOP_WORDS & set(words):
        return None
    return " ".join(words)


def series_siblings(df: pd.DataFrame) -> dict:
    """{row: rows of the other films in its series with enough votes}, by title stem.

    df: engine.movies_df.
    """
    titles = df["title"].str.lower().tolist()
    votes = df["vote_count"].tolist()
    groups = defaultdict(list)
    for row, title in enumerate(titles):
        stem = title_stem(title)
        if stem is not None:
            groups[stem].append(row)
    siblings = {}
    for rows in groups.values():
        for row in rows:
            found = set()
            for other in rows:
                if titles[other] != titles[row] and votes[other] >= SERIES_MIN_VOTES:
                    found.add(other)
            if found:
                siblings[row] = found
    return siblings


def metrics(df: pd.DataFrame, recommendations: dict) -> dict:
    """The six columns for {seed row: recommended rows}.

    df: engine.movies_df.
    """
    from movierec.movie_details import cast_names

    genres = df["genre_list"].map(set).tolist()
    directors = df["directors"].map(set).tolist()
    # The cast column keeps billing order.
    leads = df["cast"].map(lambda cast: set(cast_names(cast, LEAD_ACTORS))).tolist()
    votes = df["vote_count"].tolist()
    siblings = series_siblings(df)
    pairs = []
    for seed, recs in recommendations.items():
        for rec in recs:
            pairs.append((seed, rec))
    series_recalls = []
    for seed, recs in recommendations.items():
        if seed not in siblings:
            continue
        hits = len(siblings[seed] & set(recs))
        series_recalls.append(hits / min(len(siblings[seed]), len(recs)))
    return {
        "Genre overlap": np.mean(
            [jaccard(genres[seed], genres[rec]) for seed, rec in pairs if genres[seed]]
        ),
        "Same director": np.mean([bool(directors[seed] & directors[rec]) for seed, rec in pairs]),
        "Shared lead actor": np.mean([bool(leads[seed] & leads[rec]) for seed, rec in pairs]),
        "Series recall": np.mean(series_recalls),
        "Catalog coverage": len({rec for _, rec in pairs}) / len(df),
        "Median votes": np.median([votes[rec] for _, rec in pairs]),
    }


def evaluate(engine) -> dict:
    """{method: metrics} with every dataset row as a seed."""
    # Imported here: _load puts the repo on sys.path first.
    from movierec.engine import DETAIL_SIMILAR_COUNT

    df = engine.movies_df
    # The most-voted rows, one extra so that a seed can be left out of its own list.
    by_votes = np.argsort(-df["vote_count"].to_numpy(), kind="stable")
    most_voted = by_votes[: DETAIL_SIMILAR_COUNT + 1]
    cosine_lists = {}
    hybrid_lists = {}
    baseline_lists = {}
    for seed in range(len(df)):
        # Pure cosine: the first steps of rank_neighbours, without the hybrid re-ranking.
        product = engine.tfidf_matrix[seed] @ engine.tfidf_matrix.T
        similarity = product.toarray().ravel()
        by_similarity = np.argsort(-similarity, kind="stable")
        others = by_similarity[by_similarity != seed]
        cosine_lists[seed] = others[:DETAIL_SIMILAR_COUNT].tolist()
        hybrid_lists[seed] = engine.rank_neighbours(seed)[:DETAIL_SIMILAR_COUNT].tolist()
        baseline = most_voted[most_voted != seed]
        baseline_lists[seed] = baseline[:DETAIL_SIMILAR_COUNT].tolist()
    cosine_name, hybrid_name, baseline_name = METHODS
    return {
        cosine_name: metrics(df, cosine_lists),
        hybrid_name: metrics(df, hybrid_lists),
        baseline_name: metrics(df, baseline_lists),
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
    print(
        "| Method | Genre overlap | Same director | Shared lead actor | Series recall "
        "| Catalog coverage | Median votes |"
    )
    print("|---|---|---|---|---|---|---|")
    for method, results in evaluate(engine).items():
        print(
            f"| {method} | {results['Genre overlap']:.3f} | {results['Same director']:.1%} "
            f"| {results['Shared lead actor']:.2%} | {results['Series recall']:.1%} "
            f"| {results['Catalog coverage']:.1%} | {results['Median votes']:,.0f} |"
        )
    print(f"\nDataset median votes: {engine.movies_df['vote_count'].median():,.0f}")


if __name__ == "__main__":
    main()
