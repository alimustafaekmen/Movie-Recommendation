"""Rendered-page checks for templates/ and static/: layout, assets, flashes and accessibility."""

import html
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from conftest import PERSON

from movierec.web import build_asset_versions

REPO = Path(__file__).resolve().parents[1]
TEMPLATES = REPO / "templates"
STATIC = REPO / "static"
FLASH_TEXT = "Deneme bildirimi"
POSTER = "https://image.tmdb.org/t/p/w500/poster.jpg"
TRAILER_KEY = "abc123XYZ_-"
BIOGRAPHY_TAIL = "He is known for intense roles."
LONG_BIOGRAPHY = "Christian Bale is an English actor. " * 30 + BIOGRAPHY_TAIL

FORM_ACTIONS = {
    "results": {"film_ismi": "Avatar", "submit_similarity": "1", "num_recommendations": "5"},
    "warning": {"film_ismi": "Avatar", "submit_similarity": "1", "num_recommendations": "abc"},
    "empty": {
        "submit_features": "1",
        "tur": "Documentary",
        "min_puan": "8.5",
        "min_yil": "1900",
        "max_yil": "1905",
        "num_recommendations": "5",
    },
}


class Page(HTMLParser):
    """The start tags of a rendered page, for attribute-level assertions."""

    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    handle_startendtag = handle_starttag

    def all(self, tag, **attrs):
        """Return the attribute dicts of every `tag` whose attributes include `attrs`."""
        wanted = {key.replace("_", "-"): value for key, value in attrs.items()}
        return [
            found
            for name, found in self.tags
            if name == tag and all(found.get(key) == value for key, value in wanted.items())
        ]

    def with_id(self, element_id):
        """Return the attribute dicts of every start tag whose id is `element_id`."""
        return [found for _, found in self.tags if found.get("id") == element_id]


def _body(response):
    return response.get_data(as_text=True)


@pytest.fixture
def person_client(client, monkeypatch):
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: dict(PERSON))
    return client


@pytest.fixture
def movie_with_media(details, monkeypatch):
    original = details.get_movie_details

    def with_media(movie_id):
        movie = original(movie_id)
        movie.update(poster_url_large=POSTER, trailer_key=TRAILER_KEY)
        return movie

    monkeypatch.setattr(details, "get_movie_details", with_media)


def _pages(client, person_client):
    """Every page type once."""
    return {
        "home": client.get("/"),
        "genre": client.get("/?genre=Action"),
        "results": client.post("/", data=FORM_ACTIONS["results"]),
        "movie": client.get("/movie/155"),
        "person": person_client.get("/actor/Christian+Bale"),
        "my_lists": client.get("/my_lists"),
        "not_found": client.get("/does-not-exist"),
    }


@pytest.fixture
def pages(client, person_client):
    return {name: _body(response) for name, response in _pages(client, person_client).items()}


def test_every_page_has_one_h1_and_one_main_landmark(pages):
    for name, body in pages.items():
        page = Page(body)
        assert len(page.all("h1")) == 1, name
        assert [main.get("id") for main in page.all("main")] == ["main-content"], name


def test_footer_carries_the_tmdb_notice_and_logo_link(client):
    body = _body(client.get("/"))
    footer = body[body.index("<footer") :]

    assert "TMDB tarafından onaylanmamış veya sertifikalandırılmamıştır" in html.unescape(footer)
    assert Page(footer).all("a", href="https://www.themoviedb.org/")


def _static_urls(body):
    page = Page(body)
    urls = [tag["href"] for tag in page.all("link") if tag.get("href", "").startswith("/static/")]
    urls += [tag["src"] for tag in page.all("script") if tag.get("src", "").startswith("/static/")]
    urls += [tag["src"] for tag in page.all("img") if tag.get("src", "").startswith("/static/")]
    return urls


def test_static_urls_carry_a_stable_content_version(client, pages):
    for name, body in pages.items():
        urls = _static_urls(body)
        assert any("/static/styles.css?" in url for url in urls), name
        for url in urls:
            assert re.fullmatch(r"[0-9a-f]{12}", parse_qs(urlsplit(url).query).get("v", [""])[0]), (
                name,
                url,
            )
    assert any("/static/movie_detail.js?" in url for url in _static_urls(pages["movie"]))
    assert any("/static/my_lists.js?" in url for url in _static_urls(pages["my_lists"]))
    assert _static_urls(_body(client.get("/"))) == _static_urls(pages["home"])


def test_versioned_static_files_are_cached_for_a_year(client):
    [stylesheet] = [url for url in _static_urls(_body(client.get("/"))) if "styles.css" in url]

    response = client.get(stylesheet)

    assert response.status_code == 200
    assert response.cache_control.max_age == 31536000
    assert response.cache_control.public


def test_unversioned_static_urls_are_revalidated(client):
    # Module imports such as "./lists.js" cannot carry asset_url()'s ?v=, so a year-long
    # cache would keep an old module next to a new page after a deploy.
    response = client.get("/static/lists.js")

    assert response.status_code == 200
    assert response.cache_control.max_age == 0


def test_asset_version_follows_the_file_content(tmp_path):
    (tmp_path / "a.css").write_text("body{}")
    (tmp_path / "js").mkdir()
    (tmp_path / "js" / "b.js").write_text("let x;")
    before = build_asset_versions(tmp_path)
    (tmp_path / "a.css").write_text("body{color:red}")
    after = build_asset_versions(tmp_path)

    assert set(before) == {"a.css", "js/b.js"}
    assert before["a.css"] != after["a.css"]
    assert before["js/b.js"] == after["js/b.js"]


def test_every_page_loads_the_shared_script_once(pages):
    for name, body in pages.items():
        scripts = [
            tag
            for tag in Page(body).all("script")
            if tag.get("src", "").startswith("/static/scripts.js")
        ]
        assert len(scripts) == 1, name


def test_templates_have_no_inline_styles_scripts_or_handlers():
    for template in TEMPLATES.glob("*.html"):
        source = template.read_text(encoding="utf-8")
        assert "<style" not in source, template.name
        assert not re.search(r"<script(?![^>]*\bsrc=)", source), template.name
        assert " onclick=" not in source, template.name
        assert ' style="' not in source, template.name


# The categories movierec/web/movies.py flashes.
@pytest.mark.parametrize("category", ["warning", "danger"])
def test_every_flash_category_has_a_visible_style(category):
    css = (STATIC / "styles.css").read_text(encoding="utf-8")

    assert re.search(rf"\.alert-{category}\s*\{{[^}}]*background-color", css)


@pytest.mark.parametrize("path", ["/", "/movie/155", "/actor/Christian+Bale", "/my_lists", "/nope"])
def test_flashed_message_is_shown_once_on_every_page(person_client, path):
    with person_client.session_transaction() as session:
        session["_flashes"] = [("warning", FLASH_TEXT)]

    body = _body(person_client.get(path))

    assert body.count(FLASH_TEXT) == 1
    assert 'data-dismiss="alert"' in body


def test_header_search_is_on_every_page(pages):
    for name, body in pages.items():
        assert 'name="search_query"' in body, name


def test_every_page_links_to_my_lists_and_marks_it_current_there(pages):
    for name, body in pages.items():
        [link] = Page(body).all("a", href="/my_lists")
        assert link["class"] == "nav-link", name
        assert (link.get("aria-current") == "page") == (name == "my_lists"), name
        assert re.search(r'href="/my_lists"[^>]*>.*?Listelerim\s*</a>', body, re.S), name


def test_home_page_sets_no_session_cookie(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Set-Cookie" not in response.headers


def test_home_page_stays_small_and_loads_suggestions_lazily(client):
    body = _body(client.get("/"))
    page = Page(body)

    assert len(body.encode()) < 60_000
    assert page.all("datalist", id="movieTitlesDatalist", data_src="/api/titles")
    assert page.all("datalist", id="directorNamesDatalist", data_src="/api/directors")
    assert body.count("<option") < 200


def test_home_forms_point_at_the_results_anchor(client):
    page = Page(_body(client.get("/")))

    assert page.all("form", id="similarity-form", action="/#results")
    assert page.all("form", id="feature-form", action="/#results")


@pytest.mark.parametrize("case", sorted(FORM_ACTIONS))
def test_results_anchor_exists_for_results_warnings_and_empty_state(client, case):
    page = Page(_body(client.post("/", data=FORM_ACTIONS[case])))

    assert len(page.with_id("results")) == 1


def test_results_anchor_exists_for_an_empty_genre_listing(client):
    page = Page(_body(client.get("/?genre=Nonexistent")))

    assert len(page.with_id("results")) == 1


def test_home_page_copy_names_tmdb_and_groups_form_controls(client):
    body = _body(client.get("/"))

    assert "TMDB" in body
    assert body.count("<fieldset") == 2
    assert body.count("<legend") == 2


def test_genre_filters_show_turkish_labels_and_submit_english_values(client):
    body = _body(client.get("/"))

    assert Page(body).all("input", name="tur", value="Science Fiction")
    assert re.search(r">\s*Bilim Kurgu\s*</label>", body)


def test_custom_film_name_still_searches(client):
    response = client.post(
        "/",
        data={
            "toggle_film": "on",
            "film_ismi_custom": "Inception",
            "submit_similarity": "",
            "num_recommendations": "5",
        },
    )

    body = _body(response)
    assert response.status_code == 200
    assert '"Inception" için Benzer Öneriler' in html.unescape(body)
    assert 1 <= body.count("film-card-link") <= 5


@pytest.mark.parametrize(
    ("path", "attribute"),
    [("/api/titles", "all_movie_titles"), ("/api/directors", "all_director_names")],
)
def test_suggestion_endpoints_return_cacheable_lists(client, engine, path, attribute):
    response = client.get(path)

    assert response.status_code == 200
    assert response.get_json() == getattr(engine, attribute)
    assert response.cache_control.public
    assert response.cache_control.max_age == 86400
    assert response.headers["ETag"]

    again = client.get(path, headers={"If-None-Match": response.headers["ETag"]})
    assert again.status_code == 304


def test_suggestion_endpoint_without_engine_returns_503_json(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "engine", None)

    response = client.get("/api/titles")

    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "error": "unavailable"}


def test_movie_page_with_media_renders_poster_and_trailer(client, movie_with_media):
    page = Page(_body(client.get("/movie/155")))

    [poster] = page.all("img", src=POSTER)
    assert poster.get("loading") != "lazy"
    [iframe] = page.all("iframe")
    assert iframe["src"].endswith(f"/embed/{TRAILER_KEY}")
    assert iframe["loading"] == "lazy"
    assert iframe["title"] == "The Dark Knight fragmanı"


@pytest.mark.parametrize("path", ["/movie/155", "/actor/Christian+Bale"])
def test_detail_pages_link_back_with_a_plain_anchor(person_client, path):
    body = _body(person_client.get(path))

    assert [link for link in Page(body).all("a", href="/") if "data-back" in link]
    assert "history.back" not in body


def test_movie_page_links_every_director_and_turkish_genre(client):
    body = _body(client.get("/movie/603"))

    assert 'href="/director/Lana%20Wachowski"' in body
    assert 'href="/director/Lilly%20Wachowski"' in body
    assert 'href="/actor/Keanu%20Reeves"' in body
    assert re.search(r'href="/\?genre=Action"[^>]*>\s*Aksiyon\s*<', body)
    assert re.search(r'href="/\?genre=Science\+Fiction"[^>]*>\s*Bilim Kurgu\s*<', body)


def test_person_and_genre_links_keep_plus_signs_and_spaces(client, engine, details, monkeypatch):
    original = details.get_movie_details

    def with_odd_names(movie_id):
        movie = original(movie_id)
        movie.update(
            directors=["Jean+Pierre Jeunet"],
            cast="Zoë Saldaña, C+D 100%",
            genre_details_for_display=[{"display_name": "Bilim+Kurgu", "link_name": "Sci+Fi X"}],
        )
        return movie

    monkeypatch.setattr(details, "get_movie_details", with_odd_names)
    asked = []
    monkeypatch.setattr(
        "movierec.web.movies.find_person",
        lambda name, person_type: asked.append((person_type, name)),
    )
    monkeypatch.setattr(
        engine, "recommend_by_features", lambda genres, num: (asked.append(genres), ([], []))[1]
    )
    links = Page(_body(client.get("/movie/155"))).all("a")
    hrefs = [
        link["href"]
        for link in links
        if link.get("class") in ("director-link", "cast-item", "tag-item")
    ]

    # Encoded once by url_for, decoded once by Flask.
    assert hrefs == [
        "/?genre=Sci%2BFi+X",
        "/director/Jean+Pierre%20Jeunet",
        "/actor/Zo%C3%AB%20Salda%C3%B1a",
        "/actor/C+D%20100%25",
    ]
    for href in hrefs:
        client.get(href)
    assert asked == [
        ["Sci+Fi X"],
        ("director", "Jean+Pierre Jeunet"),
        ("actor", "Zoë Saldaña"),
        ("actor", "C+D 100%"),
    ]


def test_person_page_shows_a_long_biography_in_full_behind_a_details_element(
    person_client, monkeypatch
):
    monkeypatch.setattr(
        "movierec.web.movies.find_person",
        lambda name, person_type: {**PERSON, "biography": LONG_BIOGRAPHY},
    )

    body = _body(person_client.get("/actor/Christian+Bale"))
    page = Page(body)

    assert BIOGRAPHY_TAIL in body
    assert len(page.all("details")) == 1
    assert "Devamını Oku" in body[body.index("<summary") :]
    assert "Bruce Wayne" in body


def test_person_page_shows_a_short_biography_without_a_toggle(person_client):
    body = _body(person_client.get("/actor/Christian+Bale"))

    assert PERSON["biography"] in body
    assert "<details" not in body


@pytest.mark.parametrize("path", ["/movie/155", "/my_lists"])
def test_list_pages_say_that_the_lists_need_javascript(client, path):
    body = _body(client.get(path))

    assert body.count("<noscript>") == 1
    assert "JavaScript gerekir" in body[body.index("<noscript>") : body.index("</noscript>")]


def test_movie_page_renders_both_list_buttons_inactive(client):
    body = _body(client.get("/movie/155"))
    page = Page(body)

    [like] = [tag for tag in page.all("button") if "btn-like" in tag.get("class", "")]
    [watch] = [tag for tag in page.all("button") if "btn-watchlist" in tag.get("class", "")]
    for button in (like, watch):
        assert button["type"] == "button"
        assert button["data-movie-id"] == "155"
        assert button["aria-pressed"] == "false"
    assert re.search(r"<span>Beğen</span>", body)
    assert re.search(r"<span>Daha Sonra İzle</span>", body)
    assert [tag for tag in page.all("p") if tag.get("class") == "action-status"]
    [script] = [
        tag for tag in page.all("script") if "/static/movie_detail.js?" in tag.get("src", "")
    ]
    assert script["type"] == "module"
