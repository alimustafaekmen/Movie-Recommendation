"""Write the small TMDB dataset in tests/fixtures/dataset/ from the full Kaggle CSVs.

The fixture lets the test suite run in CI, where the 46 MB dataset/ files are not available.
It holds real rows only: the movies the tests refer to, every film of the directors they
search for, and the most-voted films, so similarity and filter tests have neighbours to rank.
Credits are cut down to what the app reads. Regenerate after changing REQUIRED_IDS:

    python tests/fixtures/make_fixture_dataset.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from movierec.dataset import CAST_LIMIT, CREDIT_COLUMNS, MOVIE_COLUMNS  # noqa: E402

SOURCE = REPO / "dataset"
TARGET = Path(__file__).resolve().parent / "dataset"

# Movies the tests name: Avatar, The Dark Knight, Batman (1989 and 1966), Toy Story, Titanic,
# The Godfather, Forrest Gump, Pulp Fiction, The Matrix, Inception, Interstellar, The Avengers,
# Alien, Alien3, both films titled The Host, No Country for Old Men, Batman Returns,
# Despicable Me 2, and Megamind (the most-voted film that is both Science Fiction and Animation).
REQUIRED_IDS = [
    19995, 155, 268, 2661, 862, 597, 238, 13, 680, 603, 27205, 157336, 24428, 348, 8077, 72710,
    1255, 6977, 364, 93456, 38055,
]  # fmt: skip
DIRECTORS = [
    "Christopher Nolan",
    "Steven Spielberg",
    "Quentin Tarantino",
    "Ethan Coen",
    "Joel Coen",
]
MOST_VOTED = 100
# Low-vote films of a genre the app does not offer as a filter, for the "less known" fallback.
LOW_VOTE_GENRE = "Foreign"
LOW_VOTE_COUNT = 3


def compact(items):
    return json.dumps(items, ensure_ascii=False, separators=(",", ":"))


def trim_credits(row):
    """Keep the top-billed cast and the directors, with the fields the app reads."""
    cast = json.loads(row["cast"])[:CAST_LIMIT]
    crew = [person for person in json.loads(row["crew"]) if person["job"] == "Director"]
    return pd.Series(
        {
            "movie_id": row["movie_id"],
            "cast": compact([{key: p[key] for key in ("id", "name", "character")} for p in cast]),
            "crew": compact([{key: p[key] for key in ("id", "name", "job")} for p in crew]),
        }
    )


def pick_ids(movies, credits):
    directed = credits[
        credits["crew"].apply(
            lambda crew: any(
                person["job"] == "Director" and person["name"] in DIRECTORS
                for person in json.loads(crew)
            )
        )
    ]
    low_vote = movies[movies["genres"].str.contains(f'"{LOW_VOTE_GENRE}"')]
    ids = set(REQUIRED_IDS)
    ids |= set(directed["movie_id"])
    ids |= set(low_vote.nlargest(LOW_VOTE_COUNT, "vote_count")["id"])
    ids |= set(movies.nlargest(MOST_VOTED, "vote_count")["id"])
    return ids


def main():
    movies = pd.read_csv(SOURCE / "tmdb_5000_movies.csv", usecols=MOVIE_COLUMNS)
    credits = pd.read_csv(SOURCE / "tmdb_5000_credits.csv", usecols=CREDIT_COLUMNS)
    ids = pick_ids(movies, credits)

    movies = movies[movies["id"].isin(ids)].sort_values("id")
    credits = credits[credits["movie_id"].isin(ids)].sort_values("movie_id")
    credits = credits.apply(trim_credits, axis=1)

    TARGET.mkdir(parents=True, exist_ok=True)
    for name, frame in (("movies", movies), ("credits", credits)):
        path = TARGET / f"tmdb_5000_{name}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        print(f"{path.relative_to(REPO)}: {len(frame)} rows, {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
