import math

import numpy as np
import pytest

from movierec.config import SIMILARITY_WEIGHT
from movierec.details import format_runtime
from movierec.engine import short_cast
from movierec.similarity import hybrid_score


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
        ("A, B, C", "A, B, C"),
        ("A, B, C, D", "A, B, C, ..."),
        (" A , ,B", "A, B"),
        ("", ""),
    ],
)
def test_short_cast(cast, expected):
    assert short_cast(cast) == expected


def test_hybrid_similarity_is_normalised_within_the_pool():
    scores = hybrid_score(
        np.array([0.2, 0.1]),
        np.array([8.0, 8.0]),
        np.array([100, 100]),
        np.array([10.0, 10.0]),
    )

    assert scores[0] - scores[1] == pytest.approx(0.5 * SIMILARITY_WEIGHT)
