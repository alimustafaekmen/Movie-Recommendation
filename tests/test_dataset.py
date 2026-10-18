import math

import pytest

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
