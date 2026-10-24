"""The page language: ?lang=, then the stored choice, then the browser's; English otherwise."""

import pytest
from conftest import TURKISH_BROWSER
from flask import render_template_string
from pages import Page, body_text, texts

from movierec.web.language import COOKIE_NAME

ONE_YEAR_SECONDS = 365 * 24 * 60 * 60


@pytest.fixture
def browser(client):
    """A client whose browser asks for no language."""
    client.environ_base.pop("HTTP_ACCEPT_LANGUAGE")
    return client


def _lang(response):
    [html] = Page(body_text(response)).all("html")
    return html["lang"]


def _switch(response):
    """The links of the header's language switch, by language."""
    [switch] = Page(body_text(response)).find("nav", "lang-switch")
    return {link.attrs["hreflang"]: link for link in switch.find("a")}


def _language_cookies(response):
    return [
        header
        for header in response.headers.getlist("Set-Cookie")
        if header.startswith(f"{COOKIE_NAME}=")
    ]


def test_a_browser_that_asks_for_no_language_gets_english(browser):
    assert _lang(browser.get("/")) == "en"


# The first of Turkish or English in the browser's list wins; Werkzeug reads the list in
# falling q order.
@pytest.mark.parametrize(
    ("accept_language", "lang"),
    [
        (TURKISH_BROWSER, "tr"),
        ("tr", "tr"),
        ("tr-CY", "tr"),
        ("en-GB", "en"),
        ("de-DE", "en"),
        ("de-DE,tr;q=0.5", "tr"),
        ("de-DE,en;q=0.7,tr;q=0.5", "en"),
        ("tr;q=0.5,en;q=0.8", "en"),
        # A wildcard is no choice of Turkish.
        ("*", "en"),
        ("", "en"),
    ],
)
def test_the_browsers_language_is_used_without_a_choice(browser, accept_language, lang):
    response = browser.get("/", headers={"Accept-Language": accept_language})

    assert _lang(response) == lang
    # Worked out again on every visit, so nothing is stored.
    assert _language_cookies(response) == []


def test_a_stored_choice_beats_the_browser(client):
    client.set_cookie(COOKIE_NAME, "en")

    assert _lang(client.get("/")) == "en"


def test_lang_in_the_url_beats_the_stored_choice_and_is_stored(client):
    client.set_cookie(COOKIE_NAME, "tr")

    response = client.get("/?lang=en")

    assert _lang(response) == "en"
    [cookie] = _language_cookies(response)
    attributes = [part.strip() for part in cookie.split(";")]
    assert attributes[0] == "lang=en"
    assert f"Max-Age={ONE_YEAR_SECONDS}" in attributes
    assert {"HttpOnly", "SameSite=Lax", "Path=/"} <= set(attributes)
    assert "Secure" not in attributes
    # The next page follows the choice without ?lang=.
    assert _lang(client.get("/my_lists")) == "en"


def test_the_choice_is_a_secure_cookie_on_an_https_site(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.config, "SESSION_COOKIE_SECURE", True)

    [cookie] = _language_cookies(client.get("/?lang=en"))

    assert "Secure" in [part.strip() for part in cookie.split(";")]


def test_a_choice_already_stored_is_not_stored_again(client):
    client.set_cookie(COOKIE_NAME, "en")

    assert _language_cookies(client.get("/?lang=en")) == []


@pytest.mark.parametrize("value", ["de", "TR", "", "tr-TR"])
def test_an_unknown_language_is_ignored(client, value):
    response = client.get("/", query_string={"lang": value})
    assert _lang(response) == "tr"
    assert _language_cookies(response) == []

    client.set_cookie(COOKIE_NAME, value)
    assert _lang(client.get("/", headers={"Accept-Language": "en"})) == "en"


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/?tur=Action",
        "/movie/155",
        "/my_lists",
        "/does-not-exist",
        "/movie/0",
        # Their posters and the titles of films outside the dataset come from TMDb.
        "/api/movies?ids=155",
        "/api/recommendations?favorites=155",
    ],
)
def test_answers_that_read_the_language_tell_caches_so(client, path):
    response = client.get(path)

    assert {"Accept-Language", "Cookie"} <= set(response.vary)


def test_a_flash_redirect_reads_the_language_too(client):
    response = client.post("/search_movie_redirect", data={"search_query": ""})

    assert response.status_code == 302
    assert {"Accept-Language", "Cookie"} <= set(response.vary)


@pytest.mark.parametrize("path", ["/api/titles", "/static/css/base.css", "/api/nope", "/health"])
def test_answers_that_do_not_read_the_language_stay_one_copy_for_caches(client, path):
    response = client.get(path)

    assert "Accept-Language" not in response.vary


def test_json_errors_are_codes_in_every_language(client):
    for lang in ("tr", "en"):
        response = client.get(f"/api/nope?lang={lang}")

        assert response.get_json() == {"status": "error", "error": "not_found"}


@pytest.mark.parametrize(
    ("path", "query"),
    [
        ("/", ""),
        ("/?tur=Science+Fiction&tur=Drama&n=24", "tur=Science+Fiction&tur=Drama&n=24&"),
        ("/?lang=en&film=Heat", "film=Heat&"),
        ("/does-not-exist?x=1", "x=1&"),
    ],
)
def test_the_switch_links_to_this_page_in_each_language(client, path, query):
    links = _switch(client.get(path))

    base = path.split("?")[0]
    assert {lang: link.attrs["href"] for lang, link in links.items()} == {
        "tr": f"{base}?{query}lang=tr",
        "en": f"{base}?{query}lang=en",
    }


def test_the_switch_keeps_a_persons_name_in_the_path(person_client):
    links = _switch(person_client.get("/actor/Zo%C3%AB%20Salda%C3%B1a"))

    assert links["en"].attrs["href"] == "/actor/Zo%C3%AB%20Salda%C3%B1a?lang=en"


def test_the_switch_on_an_error_page_after_a_post_leads_home(client):
    response = client.post("/search_movie_redirect", data={"search_query": "q" * 20_000})

    assert response.status_code == 413
    assert {lang: link.attrs["href"] for lang, link in _switch(response).items()} == {
        "tr": "/?lang=tr",
        "en": "/?lang=en",
    }


@pytest.mark.parametrize("lang", ["tr", "en"])
def test_the_switch_names_each_language_in_itself_and_marks_the_current_one(client, lang):
    response = client.get(f"/?lang={lang}")
    [switch] = Page(body_text(response)).find("nav", "lang-switch")
    links = _switch(response)

    assert switch.attrs["aria-label"] == {"tr": "Dil", "en": "Language"}[lang]
    assert {code: link.text for code, link in links.items()} == {
        "tr": "TR – Türkçe",
        "en": "EN – English",
    }
    assert {code: link.attrs["lang"] for code, link in links.items()} == {"tr": "tr", "en": "en"}
    # The visible code starts the accessible name; the language's name is for screen readers.
    assert [texts(link, "span", "sr-only") for link in links.values()] == [
        ["– Türkçe"],
        ["– English"],
    ]
    current = [code for code, link in links.items() if link.attrs.get("aria-current") == "true"]
    assert current == [lang]


def test_the_header_is_english_on_an_english_page(client):
    page = Page(body_text(client.get("/movie/155?lang=en")))

    [skip] = page.find("a", "skip-link")
    assert skip.text == "Skip to content"
    assert texts(page, "a", href="/my_lists") == ["My lists"]
    [nav] = page.find("nav", "header-nav")
    assert nav.attrs["aria-label"] == "Main menu"
    [theme] = page.find("button", "theme-toggle")
    assert theme.attrs["aria-label"] == theme.attrs["title"] == "Switch to dark theme"
    # static/js/theme.js takes the label for each theme from the button.
    assert (theme.attrs["data-label-light"], theme.attrs["data-label-dark"]) == (
        "Switch to dark theme",
        "Switch to light theme",
    )
    [search] = page.find("input", name="search_query")
    assert (search.attrs["placeholder"], search.attrs["aria-label"]) == (
        "Search films...",
        "Search films",
    )
    [button] = page.find("form", "header-search-form")[0].find("button")
    assert button.attrs["aria-label"] == "Search"


def test_the_header_is_turkish_on_a_turkish_page(client):
    page = Page(body_text(client.get("/movie/155")))

    [theme] = page.find("button", "theme-toggle")
    assert (theme.attrs["data-label-light"], theme.attrs["data-label-dark"]) == (
        "Koyu temaya geç",
        "Açık temaya geç",
    )
    [search] = page.find("input", name="search_query")
    assert search.attrs["placeholder"] == "Film ara..."


def test_a_flash_message_closes_with_a_button_named_in_the_page_language(client):
    for lang, label in (("tr", "Kapat"), ("en", "Close")):
        with client.session_transaction() as session:
            session["_flashes"] = [("warning", "x")]

        [close] = Page(body_text(client.get(f"/my_lists?lang={lang}"))).find(
            "button", "close-button"
        )

        assert close.attrs["aria-label"] == label


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Accept-Language": "tr"}, "12.002 8,2 Bilim Kurgu Puan:"),
        ({}, "12,002 8.2 Science Fiction Rating:"),
    ],
)
def test_template_helpers_follow_the_page_language(app_module, headers, expected):
    # The card macro is imported with {% from %}, which gets no context: t must be a global.
    template = (
        '{% from "_movie_card.html" import movie_card %}'
        "{{ 12002 | format_number }} {{ 8.2 | format_rating }} {{ genre_label('Science Fiction') }}"
        "{{ movie_card({'id': 1, 'title': 'T', 'vote_average': 7.5}) }}"
    )
    with app_module.app.test_request_context(headers=headers):
        rendered = render_template_string(template)

    page = Page(rendered)
    numbers = rendered.split("<a", 1)[0].split()
    assert " ".join([*numbers, *texts(page, "span", "sr-only")]) == expected
