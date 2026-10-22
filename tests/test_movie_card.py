"""The shared movie card macro, templates/_movie_card.html."""

from urllib.parse import urlsplit

import pytest
from conftest import TURKISH_BROWSER
from flask import render_template_string
from pages import Page, texts

MOVIE = {
    "id": 27205,
    "title": "Inception",
    "year": 2010,
    "vote_average": 8.25,
    "poster_url": "https://image.tmdb.org/t/p/w342/inception.jpg",
}


@pytest.fixture
def render_card(app_module):
    def render(movie, **kwargs):
        with app_module.app.test_request_context(headers={"Accept-Language": TURKISH_BROWSER}):
            return render_template_string(
                '{% from "_movie_card.html" import movie_card %}{{ movie_card(movie, **kwargs) }}',
                movie=movie,
                kwargs=kwargs,
            )

    return render


def _text(html, class_name):
    """The text of the element with this class (among others), or None without one."""
    found = texts(html, class_=class_name)
    return found[0] if found else None


def _hidden_labels(html):
    return texts(html, "span", "sr-only")


def _icon_names(markup):
    """The sprite symbol each icon in the markup (HTML or an element) draws, in order."""
    page = Page(markup) if isinstance(markup, str) else markup
    return [urlsplit(use.attrs["href"]).fragment for use in page.find("use")]


def test_card_links_to_the_movie_with_a_lazy_decorative_poster(render_card):
    page = Page(render_card(MOVIE))

    assert [link["href"] for link in page.all("a", **{"class": "movie-card"})] == ["/movie/27205"]
    [poster] = page.all("img")
    assert poster["src"] == MOVIE["poster_url"]
    assert poster["loading"] == "lazy"
    # The title is the link text, so the poster adds nothing for screen readers.
    assert poster["alt"] == ""


def test_card_shows_the_title_year_and_a_labelled_rating(render_card):
    html = render_card(MOVIE)

    assert texts(html, "h3", "movie-card-title") == ["Inception"]
    # format_rating gives the decimal comma; screen readers hear "Puan:" before it.
    assert _text(html, "movie-card-meta") == "2010 Puan: 8,3"
    assert _hidden_labels(html) == ["Puan:"]
    assert not Page(html).find(class_="movie-card-note")


def test_card_without_a_poster_shows_the_placeholder(render_card):
    html = render_card({**MOVIE, "poster_url": None})

    assert not Page(html).all("img")
    assert Page(html).find(class_="movie-card-placeholder")


@pytest.mark.parametrize(("note", "label"), [("Christopher Nolan", "Yönetmen"), ("Cobb", "Rol")])
def test_card_note_is_muted_text_with_a_hidden_label(render_card, note, label):
    html = render_card(MOVIE, note=note, note_label=label)

    assert _text(html, "movie-card-note") == f"{label}: {note}"
    assert _hidden_labels(html) == ["Puan:", f"{label}:"]
    # A plain line of text like the meta line above it: no icon names the note.
    [note_element] = Page(html).find(class_="movie-card-note")
    assert _icon_names(note_element) == []


def test_card_icons_are_the_rating_star_and_the_film_placeholder(render_card):
    assert _icon_names(render_card(MOVIE)) == ["star"]
    assert _icon_names(render_card({**MOVIE, "poster_url": None})) == ["film", "star"]
    assert _icon_names(render_card(MOVIE, note="Cobb", note_label="Rol")) == ["star"]


def test_card_meta_is_a_meta_line_of_the_year_and_the_rating(render_card):
    page = Page(render_card(MOVIE))

    # base.css (.meta-line) draws the middle dot between its items: the year, the rating.
    [meta] = page.all("p", **{"class": "movie-card-meta meta-line"})
    assert page.all("span", **{"class": "movie-card-year"})
    assert page.all("span", **{"class": "movie-card-rating rating"})
    # "★ 8,3": the star in the rating colour, the number in ink.
    [star] = page.all("svg")
    assert star["class"].split() == ["icon", "rating-star"]


def test_card_note_without_a_label_is_plain_text(render_card):
    html = render_card(MOVIE, note="Bruce Wayne")

    assert _text(html, "movie-card-note") == "Bruce Wayne"
    assert _hidden_labels(html) == ["Puan:"]


# Year 0 is how the dataset marks an unknown release date.
@pytest.mark.parametrize(
    "movie", [{**MOVIE, "year": 0}, {k: v for k, v in MOVIE.items() if k != "year"}]
)
def test_card_leaves_out_an_unknown_year_and_a_missing_rating(render_card, movie):
    rated = render_card(movie)
    unrated = render_card({**movie, "vote_average": 0})

    assert _text(rated, "movie-card-meta") == "Puan: 8,3"
    assert _text(unrated, "movie-card-meta") == ""
    assert "Puan" not in unrated
