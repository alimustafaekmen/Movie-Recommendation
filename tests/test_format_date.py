import pytest

from movierec.web import format_date_filter


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2023-10-26", "26 Ekim 2023"),
        ("1974-01-30", "30 Ocak 1974"),
        ("1980-03-05", "05 Mart 1980"),
        ("1999-12-31", "31 Aralık 1999"),
        (None, "Bilinmiyor"),
        ("", "Bilinmiyor"),
        ("  ", "Bilinmiyor"),
        ("not a date", "not a date"),
    ],
)
def test_format_date_filter(value, expected):
    assert format_date_filter(value) == expected
