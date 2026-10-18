import html
from urllib.parse import urlparse

import pytest
from conftest import PERSON


def _location_path(response):
    return urlparse(response.headers["Location"]).path


def test_home_page_renders_both_search_forms(client):
    response = client.get("/")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert 'id="similarity-form"' in body
    assert 'id="feature-form"' in body


def test_genre_query_parameter_lists_movies(client):
    response = client.get("/?genre=Action")

    assert response.status_code == 200
    assert "film-card-link" in response.get_data(as_text=True)


def test_similarity_search_returns_at_most_requested_count(client):
    response = client.post(
        "/",
        data={"film_ismi": "Avatar", "submit_similarity": "1", "num_recommendations": "5"},
    )

    assert response.status_code == 200
    assert 1 <= response.get_data(as_text=True).count("film-card-link") <= 5


def test_non_numeric_recommendation_count_shows_validation_message(client):
    response = client.post(
        "/",
        data={"film_ismi": "Avatar", "submit_similarity": "1", "num_recommendations": "abc"},
    )

    assert response.status_code == 200
    assert "Lütfen sayısal alanlara sadece geçerli sayılar girin." in response.get_data(
        as_text=True
    )


def test_feature_search_by_genre_returns_results(client):
    response = client.post(
        "/", data={"submit_features": "1", "tur": "Action", "num_recommendations": "10"}
    )

    assert response.status_code == 200
    assert 1 <= response.get_data(as_text=True).count("film-card-link") <= 10


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
    body = response.get_data(as_text=True)
    assert "Aradığınız sayfa bulunamadı." in body
    assert '<p class="error-code">404</p>' in body


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


def test_unhandled_error_renders_turkish_500(client, app_module, engine, monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine, "recommend_by_features", broken)
    monkeypatch.setitem(app_module.app.config, "PROPAGATE_EXCEPTIONS", False)

    response = client.get("/?genre=Action")

    assert response.status_code == 500
    body = response.get_data(as_text=True)
    assert "Beklenmedik bir hata oluştu." in body
    assert "boom" not in body


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
    ],
)
def test_header_search_returns_only_to_same_site_referrer(client, referrer, expected):
    response = client.post(
        "/search_movie_redirect", data={"search_query": ""}, headers={"Referer": referrer}
    )

    assert response.status_code == 302
    assert response.headers["Location"] == expected


def test_genre_route_redirects_to_filtered_home(client):
    response = client.get("/genre/Action")

    assert response.status_code == 302
    assert "genre=Action" in response.headers["Location"]


def test_director_route_without_tmdb_key_redirects_home(client):
    response = client.get("/director/Christopher+Nolan")

    assert response.status_code == 302
    assert _location_path(response) == "/"


def test_person_page_renders(client, monkeypatch):
    monkeypatch.setattr("movierec.web.movies.find_person", lambda name, person_type: dict(PERSON))

    response = client.get("/actor/Christian+Bale")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Christian Bale" in body
    assert 'href="/movie/155"' in body
    assert "30 Ocak 1974" in body


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


def test_tmdb_outage_on_person_page_flashes_a_warning(client, monkeypatch):
    from movierec.tmdb import TmdbUnavailable

    def unavailable(name, person_type):
        raise TmdbUnavailable("down")

    monkeypatch.setattr("movierec.web.movies.find_person", unavailable)

    response = client.get("/actor/Christian+Bale", follow_redirects=True)

    assert response.request.path == "/"
    assert "Kişi bilgileri şu anda TMDb'den alınamıyor." in html.unescape(
        response.get_data(as_text=True)
    )


def test_genre_listing_title_uses_the_turkish_label(client):
    body = client.get("/?genre=Science%20Fiction").get_data(as_text=True)

    assert "Tür: Bilim Kurgu" in body


def test_feature_results_title_lists_turkish_genre_labels(client):
    response = client.post(
        "/",
        data={
            "submit_features": "1",
            "tur": ["Action", "Science Fiction"],
            "num_recommendations": "3",
        },
    )

    assert "Türler: Aksiyon, Bilim Kurgu" in response.get_data(as_text=True)
