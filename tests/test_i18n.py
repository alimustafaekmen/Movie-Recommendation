"""movierec/i18n.py: the catalogue lookups and the date, number and runtime formats."""

import math

import pytest
from flask import render_template_string

from movierec import i18n
from movierec.genres import genre_badge, genre_label
from movierec.i18n import (
    decimal_text,
    format_date,
    format_number,
    format_rating,
    format_runtime,
    message,
    translate,
    translate_count,
)


@pytest.fixture
def catalogue(monkeypatch):
    """A small catalogue in place of movierec/messages.json."""
    monkeypatch.setattr(
        i18n,
        "MESSAGES",
        {
            "test.greeting": {"en": "Hello, {name}!", "tr": "Merhaba {name}!"},
            "test.films.one": {"en": "{count} film by {name}", "tr": "{name}: {count} film"},
            "test.films.other": {"en": "{count} films by {name}", "tr": "{name}: {count} film"},
        },
    )


@pytest.mark.parametrize(("lang", "expected"), [("en", "Hello, Ada!"), ("tr", "Merhaba Ada!")])
def test_translate_fills_the_placeholders_in_the_language(catalogue, lang, expected):
    assert translate(lang, "test.greeting", name="Ada") == expected


def test_a_parameter_is_text_never_a_placeholder(catalogue):
    assert translate("en", "test.greeting", name="{count} $1") == "Hello, {count} $1!"


def test_an_unknown_key_or_a_missing_parameter_raises(catalogue):
    with pytest.raises(KeyError):
        translate("en", "test.missing")
    with pytest.raises(KeyError):
        translate("en", "test.greeting")


@pytest.mark.parametrize(
    ("lang", "count", "expected"),
    [
        ("en", 1, "1 film by Ada"),
        ("en", 2, "2 films by Ada"),
        ("en", 0, "0 films by Ada"),
        ("en", 12002, "12,002 films by Ada"),
        ("tr", 1, "Ada: 1 film"),
        ("tr", 12002, "Ada: 12.002 film"),
    ],
)
def test_translate_count_picks_the_form_and_formats_the_count(catalogue, lang, count, expected):
    assert translate_count(lang, "test.films", count, name="Ada") == expected


def test_a_message_is_a_key_with_its_parameters():
    assert message("warning.not_exact", query="inceptoin", title="Inception") == {
        "key": "warning.not_exact",
        "params": {"query": "inceptoin", "title": "Inception"},
    }
    assert message("warning.no_match") == {"key": "warning.no_match", "params": {}}


@pytest.mark.parametrize(
    ("value", "tr", "en"),
    [
        ("2023-10-26", "26 Ekim 2023", "October 26, 2023"),
        ("1974-01-30", "30 Ocak 1974", "January 30, 1974"),
        ("1980-03-05", "05 Mart 1980", "March 5, 1980"),
        ("1999-12-31", "31 Aralık 1999", "December 31, 1999"),
        (None, "Bilinmiyor", "Unknown"),
        ("", "Bilinmiyor", "Unknown"),
        ("  ", "Bilinmiyor", "Unknown"),
        ("not a date", "not a date", "not a date"),
    ],
)
def test_format_date(value, tr, en):
    assert (format_date(value, "tr"), format_date(value, "en")) == (tr, en)


@pytest.mark.parametrize(
    ("value", "tr", "en"),
    [
        (12002, "12.002", "12,002"),
        (999, "999", "999"),
        (1000, "1.000", "1,000"),
        (1234567, "1.234.567", "1,234,567"),
        (12002.0, "12.002", "12,002"),
        (0, "0", "0"),
        (None, "", ""),
    ],
)
def test_format_number(value, tr, en):
    assert (format_number(value, "tr"), format_number(value, "en")) == (tr, en)


@pytest.mark.parametrize(
    ("value", "tr", "en"),
    [
        (8.2, "8,2", "8.2"),
        (7, "7,0", "7.0"),
        (10, "10,0", "10.0"),
        (6.66, "6,7", "6.7"),
        # Halves of the shortest decimal round up, as toLocaleString does in the browser.
        (8.25, "8,3", "8.3"),
        (8.35, "8,4", "8.4"),
        (None, "", ""),
        (0, "", ""),
        (0.0, "", ""),
    ],
)
def test_format_rating(value, tr, en):
    assert (format_rating(value, "tr"), format_rating(value, "en")) == (tr, en)


@pytest.mark.parametrize(("text", "tr", "en"), [("7.5", "7,5", "7.5"), ("8", "8", "8")])
def test_decimal_text_uses_the_languages_decimal_mark(text, tr, en):
    assert (decimal_text(text, "tr"), decimal_text(text, "en")) == (tr, en)


@pytest.mark.parametrize(
    ("minutes", "tr", "en"),
    [
        (0, None, None),
        (45, "45 dk", "45 min"),
        (60, "1 sa", "1 h"),
        (152, "2 sa 32 dk", "2 h 32 min"),
        ("152", "2 sa 32 dk", "2 h 32 min"),
        (152.0, "2 sa 32 dk", "2 h 32 min"),
        (math.nan, None, None),
        (None, None, None),
        ("", None, None),
        ("abc", None, None),
        (-5, None, None),
    ],
)
def test_format_runtime(minutes, tr, en):
    assert (format_runtime(minutes, "tr"), format_runtime(minutes, "en")) == (tr, en)


def test_genre_labels_are_turkish_or_the_english_name_itself():
    assert genre_label("Science Fiction", "tr") == "Bilim Kurgu"
    assert genre_label("Science Fiction", "en") == "Science Fiction"
    # A genre without a Turkish label keeps its name.
    assert genre_label("Mumblecore", "tr") == "Mumblecore"
    assert genre_badge("Science Fiction", "tr") == {
        "display_name": "Bilim Kurgu",
        "link_name": "Science Fiction",
    }
    assert genre_badge("Science Fiction", "en")["display_name"] == "Science Fiction"


def test_number_filters_are_registered_for_templates(app_module):
    with app_module.app.test_request_context():
        rendered = render_template_string("{{ 12002 | format_number }} {{ 8.2 | format_rating }}")

    assert rendered == "12.002 8,2"
