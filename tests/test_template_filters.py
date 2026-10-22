"""The Jinja filters that movierec/web/__init__.py registers for the templates."""

import pytest
from flask import render_template_string

from movierec.web import format_date_filter, format_number_filter, format_rating_filter


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


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12002, "12.002"),
        (999, "999"),
        (1000, "1.000"),
        (1234567, "1.234.567"),
        (12002.0, "12.002"),
        (0, "0"),
        (None, ""),
    ],
)
def test_format_number_filter(value, expected):
    assert format_number_filter(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (8.2, "8,2"),
        (7, "7,0"),
        (10, "10,0"),
        (6.66, "6,7"),
        # Halves of the shortest decimal round up, as toLocaleString does in the browser.
        (8.25, "8,3"),
        (8.35, "8,4"),
        (None, ""),
        (0, ""),
        (0.0, ""),
    ],
)
def test_format_rating_filter(value, expected):
    assert format_rating_filter(value) == expected


def test_number_filters_are_registered_for_templates(app_module):
    with app_module.app.test_request_context():
        rendered = render_template_string("{{ 12002 | format_number }} {{ 8.2 | format_rating }}")

    assert rendered == "12.002 8,2"
