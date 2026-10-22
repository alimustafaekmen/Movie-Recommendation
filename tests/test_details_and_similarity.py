import math

import numpy as np
import pandas as pd
import pytest

from movierec.config import POPULARITY_WEIGHT, SIMILARITY_WEIGHT, VOTE_COUNT_WEIGHT
from movierec.movie_details import cast_names, format_runtime
from movierec.similarity import build_tfidf_matrix, hybrid_score


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [
        (0, None),
        (45, "45 dk"),
        (60, "1 sa"),
        (152, "2 sa 32 dk"),
        ("152", "2 sa 32 dk"),
        (152.0, "2 sa 32 dk"),
        (math.nan, None),
        (None, None),
        ("", None),
        ("abc", None),
        (-5, None),
    ],
)
def test_format_runtime(minutes, expected):
    assert format_runtime(minutes) == expected


@pytest.mark.parametrize(
    ("cast", "expected"),
    [
        ("A, B, C", ["A", "B", "C"]),
        (" A , ,B", ["A", "B"]),
        ("", []),
        ("Sammy Davis, Jr., Frank Sinatra", ["Sammy Davis, Jr.", "Frank Sinatra"]),
        ("Page Joseph Falkinburg, Jr", ["Page Joseph Falkinburg, Jr"]),
        ("Jr., A", ["Jr.", "A"]),
        (", ".join(f"Name {n}" for n in range(15)), [f"Name {n}" for n in range(12)]),
    ],
)
def test_cast_names(cast, expected):
    assert cast_names(cast) == expected


def test_hybrid_similarity_is_normalised_within_the_pool():
    scores = hybrid_score(
        np.array([0.2, 0.1]),
        np.array([8.0, 8.0]),
        np.array([100, 100]),
        np.array([10.0, 10.0]),
    )

    assert scores[0] - scores[1] == pytest.approx(0.5 * SIMILARITY_WEIGHT)


@pytest.mark.parametrize(
    ("vote_count", "popularity", "weight"),
    [
        ([100, 10], [5.0, 5.0], VOTE_COUNT_WEIGHT),
        ([100, 100], [100.0, 10.0], POPULARITY_WEIGHT),
    ],
)
def test_hybrid_votes_and_popularity_are_log_scaled(vote_count, popularity, weight):
    scores = hybrid_score(
        np.array([0.2, 0.2]), np.array([8.0, 8.0]), np.array(vote_count), np.array(popularity)
    )

    # Linear scaling would put the gap at 0.9 * weight.
    gap = weight * (1 - np.log1p(10.0) / np.log1p(100.0))
    assert scores[0] - scores[1] == pytest.approx(gap)


def _film(directors, genres, keywords, overview):
    return {
        "directors": directors,
        "genre_list": genres,
        "keywords": keywords,
        "overview": overview,
        "tagline": "",
    }


def test_a_shared_first_name_does_not_make_directors_similar():
    films = pd.DataFrame(
        [
            _film(["Christopher Nolan"], ["Drama"], "dream", "A thief steals secrets."),
            _film(["Christopher Columbus"], ["Comedy"], "holiday", "A boy is left behind."),
        ]
    )

    matrix = build_tfidf_matrix(films)

    assert (matrix[0] @ matrix[1].T).toarray()[0, 0] == 0
