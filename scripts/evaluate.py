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


def series_siblings(movies) -> dict:
    """{row: rows of the other films in its series with enough votes}, by title stem."""
    titles = movies["title"].str.lower().tolist()
    votes = movies["vote_count"].tolist()
    groups = defaultdict(list)
    for row, title in enumerate(titles):
        if stem := title_stem(title):
            groups[stem].append(row)
    siblings = {}
    for rows in groups.values():
        for row in rows:
            found = {r for r in rows if titles[r] != titles[row] and votes[r] >= SERIES_MIN_VOTES}
            if found:
                siblings[row] = found
    return siblings


def metrics(movies, recommendations: dict) -> dict:
    """The six columns for {seed row: recommended rows} over the ``movies`` DataFrame."""
    genres = movies["genre_list"].map(set).tolist()
    directors = movies["directors"].map(set).tolist()
    # The cast column keeps billing order.
    leads = movies["cast"].map(lambda cast: set(cast.split(", ")[:LEAD_ACTORS]) - {""}).tolist()
    votes = movies["vote_count"].tolist()
    siblings = series_siblings(movies)
    pairs = [(seed, rec) for seed, recs in recommendations.items() for rec in recs]
    found = [
        len(siblings[seed] & set(recs)) / min(len(siblings[seed]), len(recs))
        for seed, recs in recommendations.items()
        if seed in siblings
    ]
    return {
        "Genre overlap": np.mean([jaccard(genres[s], genres[r]) for s, r in pairs if genres[s]]),
        "Same director": np.mean([bool(directors[s] & directors[r]) for s, r in pairs]),
        "Shared lead actor": np.mean([bool(leads[s] & leads[r]) for s, r in pairs]),
        "Series recall": np.mean(found),
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
    print(
        "| Method | Genre overlap | Same director | Shared lead actor | Series recall "
        "| Catalog coverage | Median votes |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, row in evaluate(engine).items():
        print(
            f"| {name} | {row['Genre overlap']:.3f} | {row['Same director']:.1%} "
            f"| {row['Shared lead actor']:.2%} | {row['Series recall']:.1%} "
            f"| {row['Catalog coverage']:.1%} | {row['Median votes']:,.0f} |"
        )
    print(f"\nDataset median votes: {engine.movies_df['vote_count'].median():,.0f}")


if __name__ == "__main__":
    main()
