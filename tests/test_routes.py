import html
import logging
import re
from urllib.parse import urlparse

import pytest
from conftest import PERSON
from pages import Page, texts


def _location_path(response):
    return urlparse(response.headers["Location"]).path


@pytest.mark.parametrize(("movie_id", "title"), [(19995, "Avatar"), (155, "The Dark Knight")])
def test_movie_detail_page_shows_title(client, movie_id, title):
    response = client.get(f"/movie/{movie_id}")

    assert response.status_code == 200
    assert title in response.get_data(as_text=True)


@pytest.mark.parametrize("movie_id", [0, 999999999, 99999999999])
def test_unknown_movie_id_returns_turkish_404(client, movie_id):
    response = client.get(f"/movie/{movie_id}")

    assert response.status_code == 404
    assert "Aradığınız sayfa bulunamadı." in response.get_data(as_text=True)


def test_unknown_path_returns_turkish_404(client):
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    page = Page(response.get_data(as_text=True))
    assert "Aradığınız sayfa bulunamadı." in page.text
    assert texts(page, class_="error-code") == ["404"]


@pytest.mark.parametrize(
    ("lang", "heading", "message", "home"),
    [
        ("tr", "Sayfa bulunamadı", "Aradığınız sayfa bulunamadı.", "Ana Sayfaya Dön"),
        ("en", "Page not found", "The page you are looking for was not found.", "Back to home"),
    ],
)
def test_the_404_page_speaks_the_page_language(client, lang, heading, message, home):
    page = Page(client.get(f"/does-not-exist?lang={lang}").get_data(as_text=True))

    assert texts(page, "h1") == [heading]
    assert texts(page, "title") == [f"{heading} · MovieRec"]
    assert texts(page, class_="error-page-message") == [message]
    assert texts(page, "a", "btn-primary") == [home]


@pytest.mark.parametrize(
    ("method", "url", "status", "message"),
    [
        ("get", "/search_movie_redirect", 405, "This address cannot be used this way."),
        ("post", "/search_movie_redirect", 413, "The request sent is too large."),
    ],
)
def test_error_pages_speak_english_to_an_english_visitor(client, method, url, status, message):
    client.set_cookie("lang", "en")

    response = getattr(client, method)(url, data={"search_query": "q" * 20_000})

    assert response.status_code == status
    page = Page(response.get_data(as_text=True))
    assert texts(page, "h1") == ["Something went wrong"]
    assert texts(page, class_="error-page-message") == [message]


def test_pages_that_need_the_engine_say_so_in_english(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "engine", None)

    home = client.get("/?lang=en")
    movie = client.get("/movie/155?lang=en")

    assert home.status_code == movie.status_code == 503
    assert texts(home.get_data(as_text=True), class_="error-page-message") == [
        "The recommendation engine is unavailable right now. Please try again later."
    ]
    assert texts(movie.get_data(as_text=True), class_="error-page-message") == [
        "The service is unavailable right now. Please try again later."
    ]


def test_unknown_api_path_returns_json_404(client):
    response = client.get("/api/nope")

    assert response.status_code == 404
    assert response.get_json() == {"status": "error", "error": "not_found"}


def test_movie_detail_failure_hides_exception_text(client, details, monkeypatch):
    def broken(movie_id):
        raise RuntimeError("secret internals")

    monkeypatch.setattr(details, "get_movie_details", broken)

    response = client.get("/movie/155")

    assert response.status_code == 500
    body = response.get_data(as_text=True)
    assert "Film detayı şu anda alınamıyor." in body
    assert "secret internals" not in body
    english = client.get("/movie/155?lang=en").get_data(as_text=True)
    assert texts(english, class_="error-page-message") == [
        "The film details cannot be loaded right now."
    ]


@pytest.mark.parametrize(
    ("method", "url"),
    [("recommend_by_features", "/?tur=Action"), ("recommend_by_similarity", "/?film=Avatar")],
)
def test_failed_recommendation_shows_a_generic_warning(client, engine, monkeypatch, method, url):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, method, broken)

    response = client.get(url)

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Öneriler sayfaya yüklenirken bir hata oluştu." in body
    assert "boom" not in body
    english = client.get(f"{url}&lang=en").get_data(as_text=True)
    assert "Something went wrong while loading the recommendations." in english


def test_unhandled_error_renders_turkish_500(client, app_module, engine, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "popular_cards", broken)
    monkeypatch.setitem(app_module.app.config, "PROPAGATE_EXCEPTIONS", False)

    response = client.get("/")

    assert response.status_code == 500
    body = response.get_data(as_text=True)
    assert "Beklenmedik bir hata oluştu." in body
    assert "boom" not in body
    english = client.get("/?lang=en").get_data(as_text=True)
    assert texts(english, class_="error-page-message") == [
        "An unexpected error occurred. Please try again later."
    ]


def test_person_page_does_not_need_the_engine(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "engine", None)
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: dict(PERSON))

    response = client.get("/director/Christopher+Nolan")

    assert response.status_code == 200
    assert "Christian Bale" in response.get_data(as_text=True)


def test_wrong_method_returns_turkish_405(client):
    response = client.get("/search_movie_redirect")

    assert response.status_code == 405
    assert "Bu adres bu şekilde kullanılamaz." in response.get_data(as_text=True)


def test_health_endpoint_reports_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["engine_loaded"] is True
    assert payload["movies"] > 0


def test_health_endpoint_reports_degraded_without_engine(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "engine", None)

    response = client.get("/health")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "degraded"
    assert payload["engine_loaded"] is False


def test_header_search_redirects_to_closest_title(client):
    response = client.post("/search_movie_redirect", data={"search_query": "avatr"})

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/movie/19995")


def test_header_search_without_match_returns_to_the_previous_page(client):
    response = client.post(
        "/search_movie_redirect",
        data={"search_query": "q" * 5000},
        headers={"Referer": "http://localhost/movie/19995"},
    )

    assert response.headers["Location"] == "/movie/19995"
    with client.session_transaction() as session:
        [(category, message)] = session["_flashes"]
    assert category == "warning"
    assert message == f"'{'q' * 100}' ismine benzer hiçbir film bulamadık."


def _flashes(client):
    with client.session_transaction() as session:
        return session["_flashes"]


def test_header_search_messages_follow_the_visitors_language(client, app_module, monkeypatch):
    # The form carries no ?lang=; the stored choice decides.
    client.set_cookie("lang", "en")

    client.post("/search_movie_redirect", data={"search_query": "q" * 5000})
    assert _flashes(client) == [("warning", f"We found no film with a title like '{'q' * 100}'.")]
    client.get("/")

    client.post("/search_movie_redirect", data={"search_query": " "})
    assert _flashes(client) == [("warning", "Please enter a film title.")]
    client.get("/")

    monkeypatch.setitem(app_module.app.extensions, "engine", None)
    client.post("/search_movie_redirect", data={"search_query": "Heat"})
    assert _flashes(client) == [("danger", "The recommendation engine is unavailable.")]


def test_header_search_with_empty_query_redirects(client):
    response = client.post("/search_movie_redirect", data={"search_query": ""})

    assert response.status_code == 302


@pytest.mark.parametrize(
    ("referrer", "expected"),
    [
        ("https://evil.example/x", "/"),
        ("http://localhost//evil.example/x", "/"),
        ("http://localhost/\\evil.example/x", "/"),
        ("http://localhost/api/movies?ids=155", "/"),
        ("http://localhost/movie/155?tab=cast", "/movie/155?tab=cast"),
        # urlsplit raises on a broken IPv6 host; a header like that must not fail the search.
        ("http://[", "/"),
        ("http://[::1/movie/155", "/"),
    ],
)
def test_header_search_returns_only_to_same_site_referrer(client, referrer, expected):
    response = client.post(
        "/search_movie_redirect", data={"search_query": ""}, headers={"Referer": referrer}
    )

    assert response.status_code == 302
    assert response.headers["Location"] == expected


def test_director_route_without_tmdb_key_redirects_home(client):
    response = client.get("/director/Christopher+Nolan")

    assert response.status_code == 302
    assert _location_path(response) == "/"


def test_person_page_shows_the_name_filmography_and_birthday(client, monkeypatch):
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: dict(PERSON))

    response = client.get("/actor/Christian+Bale")

    assert response.status_code == 200
    page = Page(response.get_data(as_text=True))
    assert "Christian Bale" in texts(page, "h1")
    assert page.find("a", href="/movie/155")
    assert "30 Ocak 1974" in page.text


def test_person_lookup_gets_the_decoded_name_and_role(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "movierec.web.movies.find_person",
        lambda name, person_type: calls.append((name, person_type)),
    )

    client.get("/director/Christopher%20Nolan")
    client.get("/actor/Zo%C3%AB%20Salda%C3%B1a")

    assert calls == [("Christopher Nolan", "director"), ("Zoë Saldaña", "actor")]


def test_unknown_person_flashes_not_found(client, monkeypatch):
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: None)

    response = client.get("/director/Nobody%20Here", follow_redirects=True)

    assert response.request.path == "/"
    assert "'Nobody Here' adlı yönetmen bulunamadı." in html.unescape(
        response.get_data(as_text=True)
    )


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/director/Nobody%20Here", "No director named 'Nobody Here' was found."),
        ("/actor/Nobody%20Here", "No actor named 'Nobody Here' was found."),
    ],
)
def test_unknown_person_flashes_in_english(client, monkeypatch, path, message):
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: None)
    client.set_cookie("lang", "en")

    response = client.get(path, follow_redirects=True)

    assert message in html.unescape(response.get_data(as_text=True))


def test_tmdb_outage_on_person_page_flashes_a_warning(client, monkeypatch):
    from movierec.tmdb import TmdbUnavailableError

    def unavailable(name, person_type):
        raise TmdbUnavailableError("down")

    monkeypatch.setattr("movierec.web.movies.find_person", unavailable)

    response = client.get("/actor/Christian+Bale", follow_redirects=True)

    assert response.request.path == "/"
    assert "Kişi bilgileri şu anda TMDb'den alınamıyor." in html.unescape(
        response.get_data(as_text=True)
    )
    english = client.get("/actor/Christian+Bale?lang=en", follow_redirects=True)
    assert "Person details cannot be loaded from TMDb right now." in html.unescape(
        english.get_data(as_text=True)
    )


@pytest.mark.parametrize("field", ["film", "yonetmen", "min_yil", "max_yil", "min_puan"])
def test_echoed_search_values_are_cut_to_the_fields_100_characters(client, field):
    # The film and director inputs have maxlength="100" and the engine reads no further, so
    # a longer value from a pasted URL is cut before the page shows it again. The language
    # switch only links to the visitor's own URL, so it is left out.
    body = client.get("/", query_string={"tur": "Action", field: "7" + "x" * 3000}).get_data(
        as_text=True
    )
    body = re.sub(r'<nav class="lang-switch".*?</nav>', "", body, flags=re.S)

    assert max(len(run) for run in re.findall("x+", body)) == 99


def test_a_query_without_letters_or_digits_finds_nothing_quietly(client, engine, caplog):
    # thefuzz warns for every query its processor reduces to nothing; any visitor could fill
    # the log with them.
    with caplog.at_level(logging.WARNING):
        assert engine.find_movie_id("?!") is None
        response = client.get("/", query_string={"film": "“…”", "n": 3})
        client.post("/search_movie_redirect", data={"search_query": "😀😀"})

    assert response.status_code == 200
    assert caplog.records == []
