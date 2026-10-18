import logging
import traceback

import pytest
import requests
from conftest import DATASET, FAKE_TMDB_KEY, FakeResponse

from movierec import people, tmdb
from movierec.engine import RecommendationEngine

DARK_KNIGHT_TR = {
    "id": 155,
    "title": "Kara Şövalye",
    "overview": "Batman suçla savaşını bir adım öteye taşıyor.",
    "tagline": "Kaosa hoş geldin.",
    "runtime": 152,
    "poster_path": "/dk.jpg",
    "videos": {
        "results": [
            {"site": "YouTube", "type": "Teaser", "official": True, "key": "teaser"},
            {"site": "YouTube", "type": "Trailer", "official": False, "key": "fan"},
            {"site": "YouTube", "type": "Trailer", "official": True, "key": "official"},
        ]
    },
}

TURKISH_VIDEOS = [
    {"site": "YouTube", "type": "Trailer", "official": True, "key": "en", "iso_639_1": "en"},
    {"site": "YouTube", "type": "Trailer", "official": False, "key": "tr-fan", "iso_639_1": "tr"},
    {"site": "YouTube", "type": "Trailer", "official": True, "key": "tr", "iso_639_1": "tr"},
]

PERSON_SEARCH = {
    "results": [
        {"id": 1, "name": "Christian Bale", "known_for_department": "Writing"},
        {"id": 3894, "name": "Christian Bale", "known_for_department": "Acting"},
    ]
}
PERSON_DETAILS = {
    "id": 3894,
    "name": "Christian Bale",
    "biography": "Christian Charles Philip Bale is an English actor.",
    "birthday": "1974-01-30",
    "profile_path": "/bale.jpg",
    "known_for_department": "Acting",
    "movie_credits": {
        "cast": [
            {
                "id": 155,
                "title": "The Dark Knight",
                "poster_path": "/dk.jpg",
                "release_date": "2008-07-16",
                "overview": "Short overview.",
                "vote_average": 8.2,
                "character": "Bruce Wayne",
            },
            {"id": 1, "title": "No Poster", "poster_path": None, "release_date": "2010-01-01"},
        ]
    },
}


def test_unconfigured_client_raises_without_calling(fake, monkeypatch):
    monkeypatch.setattr(tmdb, "TMDB_API_KEY", None)

    assert not tmdb.is_configured()
    with pytest.raises(tmdb.TmdbUnavailable):
        tmdb.get("/movie/155")
    assert fake.calls == []


def test_key_is_sent_as_a_param(fake):
    fake.routes["/movie/155"] = FakeResponse({"id": 155})

    assert tmdb.get("/movie/155", language="tr-TR") == {"id": 155}
    assert fake.calls == [("/movie/155", {"api_key": FAKE_TMDB_KEY, "language": "tr-TR"})]


def test_connection_error_does_not_leak_the_key(fake, caplog):
    fake.routes["/movie/1"] = requests.ConnectionError(
        f"Max retries exceeded with url: /3/movie/1?api_key={FAKE_TMDB_KEY}&language=tr-TR"
    )

    with caplog.at_level(logging.DEBUG), pytest.raises(tmdb.TmdbUnavailable) as excinfo:
        tmdb.get("/movie/1")

    assert "/movie/1" in caplog.text
    assert FAKE_TMDB_KEY not in caplog.text
    # urllib3 would log the full request URL, api_key included, at DEBUG.
    assert not logging.getLogger("urllib3").isEnabledFor(logging.DEBUG)
    assert FAKE_TMDB_KEY not in "".join(traceback.format_exception(excinfo.value))


def test_not_found_returns_none_and_server_error_raises(fake):
    fake.routes["/movie/2"] = FakeResponse(status_code=500)

    assert tmdb.get("/movie/1") is None
    with pytest.raises(tmdb.TmdbUnavailable):
        tmdb.get("/movie/2")


def test_failure_pauses_further_calls(fake):
    fake.routes["/movie/1"] = FakeResponse(status_code=503)

    for _ in range(3):
        with pytest.raises(tmdb.TmdbUnavailable):
            tmdb.get("/movie/1")

    assert fake.paths == ["/movie/1"]


@pytest.mark.parametrize(
    ("videos", "expected"),
    [
        (DARK_KNIGHT_TR["videos"]["results"], "official"),
        ([{"site": "YouTube", "type": "Trailer", "key": "any"}], "any"),
        ([{"site": "Vimeo", "type": "Trailer", "official": True, "key": "v"}], None),
        ([{"site": "YouTube", "type": "Teaser", "key": "t"}], "t"),
        ([], None),
        (TURKISH_VIDEOS, "tr"),
        (TURKISH_VIDEOS[:2], "en"),
    ],
)
def test_pick_trailer(videos, expected):
    assert tmdb.pick_trailer(videos) == expected


def test_cold_movie_details_make_one_details_call(fake, cold_details):
    fake.routes["/movie/155"] = FakeResponse(DARK_KNIGHT_TR)

    movie = cold_details.get_movie_details(155)

    assert fake.paths.count("/movie/155") == 1
    assert not any(path.endswith("/videos") for path in fake.paths)
    assert dict(fake.calls)["/movie/155"]["append_to_response"] == "videos"
    assert movie["overview"] == DARK_KNIGHT_TR["overview"]
    assert movie["trailer_key"] == "official"
    assert movie["poster_url_large"] == "https://image.tmdb.org/t/p/w500/dk.jpg"
    assert cold_details.get_movie_poster_url(155) == "https://image.tmdb.org/t/p/w342/dk.jpg"
    assert fake.paths.count("/movie/155") == 1


def test_movie_details_during_an_outage_degrade_with_a_single_call(fake, cold_details):
    fake.routes["/movie/155"] = requests.Timeout()

    movie = cold_details.get_movie_details(155)

    assert movie["title"] == "The Dark Knight"
    assert movie["poster_url_large"] is None
    assert movie["trailer_key"] is None
    assert len(movie["similar_movies"]) == 8
    assert fake.paths == ["/movie/155"]


def test_poster_is_cached_only_after_an_answer(fake, cold_details, monkeypatch):
    fake.routes["/movie/155"] = [requests.ConnectionError(), FakeResponse(DARK_KNIGHT_TR)]

    assert cold_details.get_movie_poster_url(155) is None

    monkeypatch.setattr(tmdb, "_cooldown_until", 0.0)
    expected = "https://image.tmdb.org/t/p/w342/dk.jpg"
    assert cold_details.get_movie_poster_url(155) == expected
    assert cold_details.get_movie_poster_url(155) == expected
    assert fake.paths == ["/movie/155", "/movie/155"]


def test_missing_poster_is_cached_after_not_found(fake, cold_details):
    assert cold_details.get_movie_poster_url(424242) is None
    assert cold_details.get_movie_poster_url(424242) is None
    assert fake.paths == ["/movie/424242"]


def test_poster_cache_is_emptied_once_it_holds_5000_entries(fake, cold_details):
    fake.routes["/movie/155"] = FakeResponse(DARK_KNIGHT_TR)
    cold_details.poster_cache.update((f"filler-{n}", None) for n in range(5000))

    assert cold_details.get_movie_poster_url(155) == "https://image.tmdb.org/t/p/w342/dk.jpg"
    assert len(cold_details.poster_cache) == 1
    assert cold_details.get_movie_poster_url(155, size="w500").endswith("/w500/dk.jpg")
    assert fake.paths == ["/movie/155"]


def test_find_person_prefers_the_requested_department(fake):
    fake.routes["/search/person"] = FakeResponse(PERSON_SEARCH)
    fake.routes["/person/3894"] = FakeResponse(PERSON_DETAILS)

    person = people.find_person("Christian Bale", "actor")

    assert dict(fake.calls)["/search/person"]["query"] == "Christian Bale"
    assert person["id"] == 3894
    assert person["profile_image_url"] == "https://image.tmdb.org/t/p/h632/bale.jpg"
    assert [movie["title"] for movie in person["movies"]] == ["The Dark Knight"]
    assert person["movies"][0]["character"] == "Bruce Wayne"


@pytest.mark.parametrize(
    ("homepage", "expected"),
    [
        ("https://bale.example/", "https://bale.example/"),
        ("http://bale.example", "http://bale.example"),
        ("javascript:alert(1)", None),
        ("JavaScript:alert(1)", None),
        ("data:text/html,hi", None),
        ("//bale.example", None),
        ("", None),
        (None, None),
    ],
)
def test_person_homepage_is_kept_only_for_web_addresses(fake, homepage, expected):
    fake.routes["/search/person"] = FakeResponse(PERSON_SEARCH)
    fake.routes["/person/3894"] = FakeResponse({**PERSON_DETAILS, "homepage": homepage})

    assert people.find_person("Christian Bale", "actor")["homepage"] == expected


def test_find_person_returns_none_when_nobody_matches(fake):
    fake.routes["/search/person"] = FakeResponse({"results": []})

    assert people.find_person("Zzqx Qwerty", "director") is None


def test_find_person_propagates_unavailable(fake):
    fake.routes["/search/person"] = FakeResponse(status_code=502)

    with pytest.raises(tmdb.TmdbUnavailable):
        people.find_person("Christopher Nolan", "director")


def test_overlong_person_name_makes_no_call(fake):
    assert people.find_person("a" * 150, "director") is None
    assert fake.calls == []


@pytest.mark.parametrize(
    ("overview", "expected"),
    [
        ("Short overview.", "Short overview."),
        ("x" * 120, "x" * 120),
        ("x" * 121, "x" * 119 + "…"),
        ("word " * 30, ("word " * 30)[:119] + "…"),
        ("", "Özet bulunmuyor."),
    ],
)
def test_credit_overview_is_shortened_only_when_long(overview, expected):
    assert people.short_overview(overview) == expected


def test_person_page_renders_from_canned_tmdb_json(fake, client):
    fake.routes["/search/person"] = FakeResponse(PERSON_SEARCH)
    fake.routes["/person/3894"] = FakeResponse(PERSON_DETAILS)

    response = client.get("/actor/Christian+Bale")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Christian Bale" in body
    assert 'href="/movie/155"' in body
    assert "30 Ocak 1974" in body


def test_tmdb_only_movie_renders_from_canned_json(fake, client, cold_details):
    fake.routes["/movie/999001"] = FakeResponse(
        {
            "id": 999001,
            "title": "Yalnızca TMDb Filmi",
            "overview": "Veri setinde olmayan bir film.",
            "release_date": "2024-05-01",
            "runtime": 95,
            "vote_average": 7.04,
            "vote_count": 321,
            "popularity": 12.5,
            "poster_path": "/only.jpg",
            "genres": [{"id": 878, "name": "Bilim-Kurgu"}],
            "credits": {
                "cast": [{"name": "Oyuncu Bir"}, {"name": "Oyuncu İki"}],
                "crew": [{"job": "Director", "name": "Yönetmen Kişi"}],
            },
            "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "abc123"}]},
        }
    )
    fake.routes["/movie/999001/similar"] = FakeResponse(
        {"results": [{"id": 155, "title": "The Dark Knight", "poster_path": "/dk.jpg"}]}
    )

    response = client.get("/movie/999001")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Yalnızca TMDb Filmi" in body
    assert "Yönetmen Kişi" in body
    assert "abc123" in body
    assert 'href="/movie/155"' in body
    assert fake.paths.count("/movie/999001") == 1


def test_engine_construction_makes_no_tmdb_call(fake):
    RecommendationEngine(
        str(DATASET / "tmdb_5000_movies.csv"), str(DATASET / "tmdb_5000_credits.csv")
    )

    assert fake.calls == []


def test_tmdb_only_genres_are_mapped_by_id(fake, cold_details):
    fake.routes["/movie/999002"] = FakeResponse(
        {
            "id": 999002,
            "title": "Uzay",
            "genres": [{"id": 878, "name": "Bilim-Kurgu"}, {"id": 10770, "name": "TV filmi"}],
        }
    )

    movie = cold_details.get_movie_details(999002)

    assert [(b["display_name"], b["link_name"]) for b in movie["genre_details_for_display"]] == [
        ("Bilim Kurgu", "Science Fiction")
    ]


@pytest.mark.parametrize("movie_id", [155, 999003])
def test_detail_request_includes_non_turkish_videos(fake, cold_details, movie_id):
    fake.routes[f"/movie/{movie_id}"] = FakeResponse({"id": movie_id, "title": "X"})

    cold_details.get_movie_details(movie_id)

    params = dict(fake.calls)[f"/movie/{movie_id}"]
    assert params["include_video_language"] == "tr,en,null"
    assert "videos" in params["append_to_response"]


def test_result_cards_and_similar_movies_get_posters(fake, cold_details, client, monkeypatch):
    monkeypatch.setattr(tmdb, "get", lambda path, **params: {"poster_path": "/p.jpg"})
    poster = "https://image.tmdb.org/t/p/w342/p.jpg"

    page = client.post(
        "/", data={"film_ismi": "Inception", "submit_similarity": "1", "num_recommendations": "3"}
    )
    assert page.get_data(as_text=True).count(poster) == 3

    movie = cold_details.get_movie_details(155)
    assert [card["poster_url"] for card in movie["similar_movies"]] == [poster] * 8
