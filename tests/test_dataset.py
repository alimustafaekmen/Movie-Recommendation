import math

import pandas as pd
import pytest
from conftest import FIXTURE_DATASET_DIR, KAGGLE_DATASET_DIR

from movierec.dataset import parse_name_list

CREW = '[{"job": "Producer", "name": "A"}, {"job": "Director", "name": "B"}]'
CO_DIRECTED = (
    '[{"job": "Director", "name": "Joel Coen"}, {"job": "Writer", "name": "X"}, '
    '{"job": "Director", "name": " Ethan Coen "}]'
)
GENRES = '[{"id": 878, "name": "Science Fiction"}, {"id": 28, "name": " Action "}]'


def test_parse_name_list():
    assert parse_name_list(GENRES) == ["Science Fiction", "Action"]
    assert parse_name_list(CREW, job="Director") == ["B"]
    assert parse_name_list(CO_DIRECTED, job="Director") == ["Joel Coen", "Ethan Coen"]


@pytest.mark.parametrize("raw", ["", "[]", None, math.nan, "{broken", '[{"job": "Director"}]'])
def test_parse_name_list_empty_or_invalid(raw):
    assert parse_name_list(raw, job="Director") == []


@pytest.mark.real_dataset
def test_fixture_texts_are_not_copied_from_tmdb():
    # The fixture is public, so it keeps each film's facts but not TMDb's own texts.
    columns = ["id", "overview", "tagline", "keywords"]
    fixture = pd.read_csv(FIXTURE_DATASET_DIR / "tmdb_5000_movies.csv", usecols=columns)
    kaggle = pd.read_csv(KAGGLE_DATASET_DIR / "tmdb_5000_movies.csv", usecols=columns)
    both = fixture.merge(kaggle, on="id", suffixes=("", "_tmdb"))

    assert len(both) == len(fixture)
    for column in columns[1:]:
        copied = both[column].notna() & (both[column] == both[f"{column}_tmdb"])
        assert not copied.any(), f"{column} copied for ids {both.loc[copied, 'id'].tolist()}"
