"""The pages in English: the texts the other page tests check in Turkish."""

import pytest
from conftest import PERSON, SEARCHES
from pages import Page, body_text, texts

LONG_BIOGRAPHY = "Christian Bale is an English actor. " * 30


@pytest.fixture
def english(client):
    """A visitor who chose English."""
    client.set_cookie("lang", "en")
    return client


def _page(client, url):
    return Page(body_text(client.get(url)))


def _patch_movie(details, monkeypatch, **fields):
    original = details.get_movie_details

    def patched(*args):
        movie = original(*args)
        movie.update(fields)
        return movie

    monkeypatch.setattr(details, "get_movie_details", patched)


def test_home_page(english):
    page = _page(english, "/")

    assert texts(page, "title") == ["MovieRec · Discover similar films"]
    assert texts(page, "p", "page-title") == ["Discover the films that suit you best"]
    assert texts(page, "p", "home-intro-text") == ["Find similar films or filter in detail."]
    assert texts(page, "h2") == ["Browse genres", "Popular films"]
    [panel] = page.find("section", "search-panel")
    assert panel.attrs["aria-label"] == "Film search"
    [tablist] = page.find(role="tablist")
    assert tablist.attrs["aria-label"] == "Search type"
    assert texts(page, "button", role="tab") == ["Similar film", "Filter"]
    buttons = page.find("button", "btn-primary", type="submit")
    assert [button.text for button in buttons] == ["Find similar films", "Filter"]
    assert [button.attrs["data-loading-text"] for button in buttons] == ["Searching..."] * 2


def test_home_page_search_fields(english):
    page = _page(english, "/")

    labels = {
        label.attrs["for"]: label.text for label in page.find("label") if "for" in label.attrs
    }
    assert [labels[name] for name in ("film", "yonetmen", "min_yil", "max_yil", "min_puan")] == [
        "A film you like",
        "Director",
        "Earliest",
        "Latest",
        "Minimum rating (TMDB)",
    ]
    assert texts(page, "legend") == ["Genres", "Release year"]
    assert [
        page.find("input", name=name)[0].attrs["placeholder"] for name in ("film", "yonetmen")
    ] == [
        "E.g. Inception",
        "E.g. Christopher Nolan",
    ]
    assert [
        page.find("input", name=name)[0].attrs["placeholder"] for name in ("min_yil", "max_yil")
    ] == [
        "e.g. 1990",
        "e.g. 2010",
    ]
    # The rating options keep their values; only the shown number follows the language.
    assert texts(page, "option") == ["Any", "5+", "6+", "7+", "7.5+", "8+", "8.5+"]
    [box] = page.find("input", name="tur", value="Science Fiction")
    assert texts(page, "label", for_=box.attrs["id"]) == ["Science Fiction"]
    assert texts(page, "a", href="/?tur=Science+Fiction") == ["Science Fiction"]


def test_turkish_rating_options_use_the_decimal_comma(client):
    assert "7,5+" in texts(_page(client, "/"), "option")


def test_similar_results(english):
    page = _page(english, "/?film=Inception&n=5")

    [heading] = page.find("h2", "results-title")
    assert heading.text == "Films similar to “Inception” 5 films"
    notes = texts(page, class_="movie-card-note")
    assert notes and all(note.startswith("Director: ") for note in notes)
    assert set(texts(page, "span", "sr-only")) >= {"Rating:", "Director:"}
    [button] = page.find("button", "search-toggle")
    assert button.text == "Edit search"
    assert texts(_page(english, "/?film=Avatar"), "a", "more-results") == ["Show more"]


def test_one_result_is_counted_in_the_singular(english):
    [count] = texts(_page(english, "/?film=Inception&n=1"), class_="results-count")

    assert count == "1 film"


def test_filter_results(english):
    [heading] = _page(english, "/?tur=Action&n=3").find("h2", "results-title")

    assert heading.text == "Filter results 3 films"


@pytest.mark.parametrize(
    ("url", "hint"),
    [
        (SEARCHES["empty"], "Try again with fewer filters or other values."),
        (
            "/?film=xyzzyplugh",
            "Try another film title, or pick one of the suggestions as you type.",
        ),
    ],
)
def test_empty_results_hint(english, url, hint):
    assert texts(_page(english, url), class_="empty-state-hint") == [hint]


def test_movie_page(english, details, monkeypatch):
    _patch_movie(
        details,
        monkeypatch,
        poster_url_large="https://image.tmdb.org/t/p/w500/poster.jpg",
        trailer_key="abc123",
        vote_average=8.25,
        vote_count=12002,
    )
    page = _page(english, "/movie/155")

    [poster] = page.find("img", "detail-image")
    assert poster.attrs["alt"] == "Poster of The Dark Knight"
    [meta] = page.find("ul", "movie-detail-meta")
    assert meta.text.endswith("Rating: 8.3 /10 12,002 votes")
    assert texts(page, "h2") == [
        "Overview",
        "Genres",
        "Director",
        "Main cast",
        "Trailer",
        "Similar films",
    ]
    [trailer] = page.find("button", "trailer-play")
    assert trailer.attrs["aria-label"] == "Play trailer"
    assert trailer.attrs["data-trailer-title"] == "The Dark Knight trailer"
    assert "Watch the trailer on YouTube" in body_text(english.get("/movie/155"))


def test_movie_page_list_buttons_and_status(english):
    page = _page(english, "/movie/155")

    [like, watchlist] = page.find("button", "action-btn")
    # movie_detail.js shows data-active-label while the film is on the list.
    assert (like.attrs["title"], like.attrs["data-label"], like.attrs["data-active-label"]) == (
        "Like",
        "Like",
        "Liked",
    )
    assert (watchlist.attrs["data-label"], watchlist.attrs["data-active-label"]) == (
        "Add to watchlist",
        "On your watchlist",
    )
    assert (texts(like, "span"), texts(watchlist, "span")) == (["Like"], ["Add to watchlist"])
    [status] = page.find("p", "action-status", role="status")
    assert status.attrs["data-load-error"] == (
        "The lists are unavailable right now. Please reload the page."
    )
    assert status.attrs["data-not-saved"] == (
        "This browser cannot save the lists; your choices are lost when you leave the page."
    )
    body = body_text(english.get("/movie/155"))
    noscript = body[body.index("<noscript>") : body.index("</noscript>")]
    assert "Adding films to the lists needs JavaScript." in noscript


def test_movie_page_without_a_trailer_or_poster(english, details, monkeypatch):
    _patch_movie(details, monkeypatch, poster_url_large=None, trailer_key=None)
    page = _page(english, "/movie/155")

    assert texts(page, "p", "detail-note") == ["No trailer was found for this film."]
    assert texts(page, class_="detail-image-placeholder") == ["No poster"]


def test_movie_page_names_two_directors_in_the_plural(english):
    # The Matrix has two directors; Turkish says "Yönetmen" either way.
    assert "Directors" in texts(_page(english, "/movie/603"), "h2")
    assert "Yönetmen" in texts(_page(english, "/movie/603?lang=tr"), "h2")


def test_one_vote_is_counted_in_the_singular(english, details, monkeypatch):
    _patch_movie(details, monkeypatch, vote_count=1)

    [meta] = _page(english, "/movie/155").find("ul", "movie-detail-meta")

    assert meta.text.endswith("1 vote")


def test_person_page(english, monkeypatch):
    person = {**PERSON, "biography": LONG_BIOGRAPHY, "deathday": "2050-02-01"}
    person["homepage"] = "https://example.com/bale"
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: person)
    page = _page(english, "/actor/Christian+Bale")

    [photo] = page.find("img", "detail-image")
    assert photo.attrs["alt"] == "Profile photo of Christian Bale"
    assert texts(page, "h2") == ["Biography", "Filmography"]
    assert texts(page, "dt") == ["Born", "Died", "Website"]
    assert texts(page, "dd")[:2] == [
        "January 30, 1974 (Haverfordwest, Wales, UK)",
        "February 1, 2050",
    ]
    assert texts(page, "summary") == ["Read more"]
    assert texts(page, "p", "movie-card-note") == ["Role: Bruce Wayne"]


@pytest.mark.parametrize(
    ("path", "role", "no_films"),
    [
        ("/actor/Christian+Bale", "Actor", "No filmography was found for this actor."),
        ("/director/Christian+Bale", "Director", "No filmography was found for this director."),
    ],
)
def test_person_page_role_and_empty_filmography(english, monkeypatch, path, role, no_films):
    person = {**PERSON, "movies": [], "profile_image_url": None}
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: person)
    page = _page(english, path)

    assert texts(page, "p", "person-role") == [role]
    assert texts(page, "p", "detail-note") == [no_films]
    assert texts(page, class_="detail-image-placeholder") == ["No photo"]


def test_my_lists_page(english):
    body = body_text(english.get("/my_lists"))
    page = Page(body)

    assert texts(page, "title") == ["My lists · MovieRec"]
    assert texts(page, "h1") == ["My film lists"]
    assert texts(page, "h2") == ["Films I like", "To watch later", "Based on your likes"]
    assert texts(page, "p", "empty-list-message") == [
        "You have not liked any films yet.",
        "Your watchlist has no films yet.",
    ]
    assert texts(page, "p", "empty-list-hint") == [
        "Add films with the “Like” button on their pages. The lists are kept in this browser.",
        "Add films with the “Add to watchlist” button on their pages. The lists are kept in this"
        " browser.",
    ]
    assert texts(page, "a", "empty-list-link") == ["Start discovering films"] * 2
    assert texts(page, "p", "recommendations-hint") == [
        "The films most like the ones you like; films on your lists are left out."
    ]
    noscript = body[body.index("<noscript>") : body.index("</noscript>")]
    assert "Your lists are kept in this browser; showing them needs JavaScript." in noscript


def test_my_lists_page_gives_its_script_the_messages(english):
    page = _page(english, "/my_lists")

    [lists] = page.find("div", "my-lists-page")
    assert lists.attrs["data-not-saved"] == (
        "This browser cannot save the lists; your choices are lost when you leave the page."
    )
    assert lists.attrs["data-load-failed"] == (
        "The films could not be loaded right now. Please reload the page."
    )
    assert lists.attrs["data-some-failed"] == "Some films could not be loaded right now."
    [status] = page.find("p", "list-status", role="status")
    assert status.attrs["data-load-error"] == (
        "Your lists cannot be shown right now. Please reload the page."
    )
    # The card template's hidden rating label.
    [template] = page.find("template", id="movie-card-template")
    assert texts(template, "span", "sr-only") == ["Rating:"]
