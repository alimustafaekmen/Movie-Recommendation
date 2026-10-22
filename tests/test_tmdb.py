import logging
import threading
import time
import traceback

import pytest
import requests
from conftest import DATASET_DIR, FAKE_TMDB_KEY, FakeResponse, FakeSession
from pages import Page, body_text, texts

from movierec import people, tmdb
from movierec.engine import DETAIL_SIMILAR_COUNT, RecommendationEngine
from movierec.movie_details import POSTER_CACHE_LIMIT, POSTER_WORKERS

# Seconds each answer of the slow fake TMDb takes, about a real round trip.
DELAY = 0.2
# gunicorn's --threads in the Dockerfile; each thread can be in a TMDb call too.
GUNICORN_THREADS = 4

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


class SlowSession(FakeSession):
    """A FakeSession whose answers take `delay` seconds; it records how many overlapped."""

    def __init__(self, delay):
        super().__init__()
        self.delay = delay
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()

    def get(self, url, params=None, timeout=None):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay)
            return super().get(url, params, timeout)
        finally:
            with self._lock:
                self.in_flight -= 1


class InvalidJsonResponse(FakeResponse):
    def json(self):
        raise ValueError("not JSON")


@pytest.fixture(autouse=True)
def empty_person_cache(monkeypatch):
    """Each test starts without remembered person lookups."""
    monkeypatch.setattr(people, "_person_cache", {})


@pytest.fixture
def slow(fake, monkeypatch):
    """Like `fake`, but every answer takes DELAY seconds."""
    session = SlowSession(DELAY)
    monkeypatch.setattr(tmdb, "_session", session)
    return session


def _poster_routes(session, movie_ids):
    for movie_id in movie_ids:
        session.routes[f"/movie/{movie_id}"] = FakeResponse(
            {"id": movie_id, "title": f"Film {movie_id}", "poster_path": f"/{movie_id}.jpg"}
        )


def _poster(movie_id):
    return f"https://image.tmdb.org/t/p/w342/{movie_id}.jpg"


def _credit(movie_id, **fields):
    return {"id": movie_id, "title": f"Film {movie_id}", "poster_path": f"/{movie_id}.jpg"} | fields


def _filmography(fake, person_type="actor", **movie_credits):
    """The filmography find_person builds from these cast/crew credits."""
    fake.routes["/search/person"] = FakeResponse({"results": [{"id": 3894}]})
    fake.routes["/person/3894"] = FakeResponse({**PERSON_DETAILS, "movie_credits": movie_credits})
    return people.find_person("Christian Bale", person_type)["movies"]


def test_unconfigured_client_raises_without_calling(fake, monkeypatch):
    monkeypatch.setattr(tmdb, "TMDB_API_KEY", None)

    assert not tmdb.is_configured()
    with pytest.raises(tmdb.TmdbUnavailableError):
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

    with caplog.at_level(logging.DEBUG), pytest.raises(tmdb.TmdbUnavailableError) as excinfo:
        tmdb.get("/movie/1")

    assert "/movie/1" in caplog.text
    assert FAKE_TMDB_KEY not in caplog.text
    # urllib3 would log the full request URL, api_key included, at DEBUG.
    assert not logging.getLogger("urllib3").isEnabledFor(logging.DEBUG)
    assert FAKE_TMDB_KEY not in "".join(traceback.format_exception(excinfo.value))


def test_not_found_returns_none_and_server_error_raises(fake):
    fake.routes["/movie/2"] = FakeResponse(status_code=500)

    assert tmdb.get("/movie/1") is None
    with pytest.raises(tmdb.TmdbUnavailableError):
        tmdb.get("/movie/2")


def test_failure_pauses_further_calls(fake):
    fake.routes["/movie/1"] = FakeResponse(status_code=503)

    for _ in range(3):
        with pytest.raises(tmdb.TmdbUnavailableError):
            tmdb.get("/movie/1")

    assert fake.paths == ["/movie/1"]


@pytest.mark.parametrize("payload", [["not", "an", "object"], "text", 42])
def test_json_that_is_not_an_object_counts_as_a_failed_call(fake, cold_details, payload):
    fake.routes["/movie/7"] = FakeResponse(payload)
    fake.routes["/search/person"] = FakeResponse(payload)

    with pytest.raises(tmdb.TmdbUnavailableError):
        tmdb.get("/movie/7")
    with pytest.raises(tmdb.TmdbUnavailableError):
        people.find_person("Christian Bale", "actor")
    cards = [{"id": 7}]
    cold_details.add_posters(cards)

    assert cards[0]["poster_url"] is None
    assert cold_details.get_movie_poster_url(7) is None
    # Like invalid JSON, a broken answer is not remembered as "no poster".
    assert 7 not in cold_details.poster_cache


def test_session_keeps_a_connection_for_every_concurrent_caller():
    adapter = tmdb._session.get_adapter(tmdb.TMDB_API_BASE_URL)

    # Poster workers and request threads share the session; past maxsize, urllib3 opens a
    # fresh connection for each extra call and throws it away afterwards.
    assert adapter.poolmanager.connection_pool_kw["maxsize"] >= POSTER_WORKERS + GUNICORN_THREADS


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
    assert len(movie["similar_movies"]) == DETAIL_SIMILAR_COUNT
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


def test_poster_cache_holds_the_whole_refreshed_catalogue(fake, cold_details):
    # The refreshed catalogue has about 9,000 films.
    _poster_routes(fake, [1])
    cold_details.poster_cache.update((f"filler-{n}", None) for n in range(9999))

    assert cold_details.get_movie_poster_url(1) == _poster(1)
    assert len(cold_details.poster_cache) == 10000


def test_poster_cache_is_emptied_once_it_holds_10000_entries(fake, cold_details):
    fake.routes["/movie/155"] = FakeResponse(DARK_KNIGHT_TR)
    cold_details.poster_cache.update((f"filler-{n}", None) for n in range(10000))

    assert cold_details.get_movie_poster_url(155) == "https://image.tmdb.org/t/p/w342/dk.jpg"
    assert len(cold_details.poster_cache) == 1
    assert cold_details.get_movie_poster_url(155, size="w500").endswith("/w500/dk.jpg")
    assert fake.paths == ["/movie/155"]


def test_result_posters_are_fetched_in_parallel_in_card_order(slow, cold_details):
    ids = list(range(900015, 900000, -1))
    _poster_routes(slow, ids)
    cards = [{"id": movie_id} for movie_id in ids]

    cold_details.add_posters(cards)

    assert 1 < slow.max_in_flight <= POSTER_WORKERS
    assert [card["poster_url"] for card in cards] == [_poster(movie_id) for movie_id in ids]
    assert sorted(slow.paths) == sorted(f"/movie/{movie_id}" for movie_id in ids)


def test_add_posters_fetches_each_uncached_id_once(fake, cold_details):
    _poster_routes(fake, [1, 2])
    cold_details.poster_cache.update({3: "/cached.jpg", 4: None})
    cards = [{"id": 1}, {"id": 3}, {"id": 1}, {"id": 4}, {"id": 2}]

    cold_details.add_posters(cards)

    assert [card["poster_url"] for card in cards] == [
        _poster(1),
        "https://image.tmdb.org/t/p/w342/cached.jpg",
        _poster(1),
        None,
        _poster(2),
    ]
    assert sorted(fake.paths) == ["/movie/1", "/movie/2"]
    assert cold_details.poster_cache == {1: "/1.jpg", 2: "/2.jpg", 3: "/cached.jpg", 4: None}


def test_one_failed_poster_lookup_leaves_the_others(fake, cold_details):
    ids = list(range(1, 16))
    _poster_routes(fake, ids)
    fake.routes["/movie/7"] = InvalidJsonResponse()
    del fake.routes["/movie/9"]
    cards = [{"id": movie_id} for movie_id in ids]

    cold_details.add_posters(cards)

    assert [card["poster_url"] for card in cards] == [
        None if movie_id in (7, 9) else _poster(movie_id) for movie_id in ids
    ]
    # A 404 is an answer (no poster); the broken one is tried again next time.
    assert cold_details.poster_cache[9] is None
    assert 7 not in cold_details.poster_cache


def test_poster_lookups_stop_once_tmdb_fails(fake, cold_details):
    ids = list(range(1, 16))
    for movie_id in ids:
        fake.routes[f"/movie/{movie_id}"] = requests.Timeout()
    cards = [{"id": movie_id} for movie_id in ids]

    cold_details.add_posters(cards)
    calls = len(fake.calls)
    cold_details.add_posters(cards)

    assert [card["poster_url"] for card in cards] == [None] * 15
    # Each worker can have one lookup under way when the first failure starts the cooldown;
    # the queued ones, and the next page, never reach TMDb.
    assert 1 <= calls <= POSTER_WORKERS
    assert len(fake.calls) == calls
    assert cold_details.poster_cache == {}


def test_posters_fill_in_when_the_cache_is_emptied_during_a_batch(fake, cold_details):
    ids = list(range(1, 16))
    _poster_routes(fake, ids)
    cold_details.poster_cache.update((f"filler-{n}", None) for n in range(POSTER_CACHE_LIMIT))
    cards = [{"id": movie_id} for movie_id in ids]

    cold_details.add_posters(cards)

    assert [card["poster_url"] for card in cards] == [_poster(movie_id) for movie_id in ids]
    assert set(cold_details.poster_cache) <= set(ids)


def test_unconfigured_add_posters_makes_no_call(fake, cold_details, monkeypatch):
    monkeypatch.setattr(tmdb, "TMDB_API_KEY", None)
    cards = [{"id": 155}, {"id": 19995}]

    cold_details.add_posters(cards)

    assert [card["poster_url"] for card in cards] == [None, None]
    assert fake.calls == []


def test_list_cards_fetch_dataset_posters_in_parallel(slow, cold_details, client, engine):
    ids = list(engine.row_of)[:10]
    _poster_routes(slow, ids)
    query = {"ids": ",".join(map(str, ids))}

    cards = client.get("/api/movies", query_string=query).get_json()

    assert 1 < slow.max_in_flight <= POSTER_WORKERS
    assert [card["id"] for card in cards] == ids
    assert [card["poster_url"] for card in cards] == [_poster(movie_id) for movie_id in ids]
    assert sorted(slow.paths) == sorted(f"/movie/{movie_id}" for movie_id in ids)


def test_list_cards_use_the_prefetched_posters_when_the_cache_empties(
    fake, cold_details, client, engine
):
    ids = list(engine.row_of)[:10]
    _poster_routes(fake, ids)
    # Five lookups fill the cache, the sixth empties it and drops the posters fetched before.
    cold_details.poster_cache.update((f"filler-{n}", None) for n in range(POSTER_CACHE_LIMIT - 5))

    cards = client.get("/api/movies", query_string={"ids": ",".join(map(str, ids))}).get_json()

    assert [card["poster_url"] for card in cards] == [_poster(movie_id) for movie_id in ids]
    assert sorted(fake.paths) == sorted(f"/movie/{movie_id}" for movie_id in ids)


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
    ("department", "label"),
    [("Directing", "Yönetmen"), ("Costume & Make-Up", "Kostüm ve Makyaj")],
)
def test_person_known_for_department_gets_a_turkish_label(fake, department, label):
    fake.routes["/search/person"] = FakeResponse(PERSON_SEARCH)
    fake.routes["/person/3894"] = FakeResponse(
        {**PERSON_DETAILS, "known_for_department": department}
    )

    person = people.find_person("Christian Bale", "actor")

    assert person["known_for_label"] == label
    assert person["known_for_department"] == department


@pytest.mark.parametrize("department", ["Creator", "directing", "", None])
def test_unknown_department_has_no_known_for_label(fake, department):
    fake.routes["/search/person"] = FakeResponse(PERSON_SEARCH)
    fake.routes["/person/3894"] = FakeResponse(
        {**PERSON_DETAILS, "known_for_department": department}
    )

    assert "known_for_label" not in people.find_person("Christian Bale", "actor")


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

    with pytest.raises(tmdb.TmdbUnavailableError):
        people.find_person("Christopher Nolan", "director")


def test_overlong_person_name_makes_no_call(fake):
    assert people.find_person("a" * 150, "director") is None
    assert fake.calls == []


def test_filmography_leaves_out_appearances_as_oneself_and_archive_footage(fake):
    not_roles = [
        "Self",
        "Himself",
        "herself",
        "Self - Host",
        "Self / Allen Bauer",
        "Self (archive footage)",
        "Woody (voice) (archive footage)",
        "(archival footage)",
    ]
    roles = ["Bruce Wayne", "Selfie Guy", "Woody (voice)", ""]
    credits = [
        _credit(movie_id, character=character)
        for movie_id, character in enumerate([*not_roles, *roles], start=1)
    ]

    movies = _filmography(fake, cast=credits)

    assert [movie["character"] for movie in movies] == roles


def test_filmography_lists_the_most_voted_films_first_up_to_the_limit(fake):
    count = people.MOVIE_LIMIT + 5
    # The newest films have the fewest votes, so release order and vote order disagree.
    credits = [
        _credit(n, vote_count=n * 10, release_date=f"{2030 - n}-01-01") for n in range(1, count + 1)
    ]

    movies = _filmography(fake, cast=credits)

    assert [movie["id"] for movie in movies] == list(range(count, 5, -1))


def test_filmography_breaks_vote_ties_by_popularity(fake):
    credits = [
        _credit(1, vote_count=0, popularity=2.0),
        _credit(2, vote_count=0, popularity=90.0),
        _credit(3, vote_count=500, popularity=1.0),
    ]

    assert [movie["id"] for movie in _filmography(fake, cast=credits)] == [3, 2, 1]


@pytest.mark.parametrize(
    ("person_type", "key", "role"),
    [("actor", "cast", {"character": "Rol"}), ("director", "crew", {"job": "Director"})],
)
def test_filmography_cards_carry_the_release_year(fake, person_type, key, role):
    credits = [
        _credit(1, release_date="2008-07-16", vote_count=3, **role),
        _credit(2, release_date="", vote_count=2, **role),
        _credit(3, vote_count=1, **role),
    ]

    movies = _filmography(fake, person_type, **{key: credits})

    assert [(movie["id"], movie["year"]) for movie in movies] == [(1, 2008), (2, 0), (3, 0)]


def test_director_filmography_lists_directed_films_with_a_poster(fake):
    crew = [
        _credit(1, job="Director"),
        _credit(2, job="Producer"),
        _credit(3, job="Director", poster_path=None),
        _credit(1, job="Director"),
    ]

    movies = _filmography(fake, "director", crew=crew, cast=[_credit(4, character="Cameo")])

    assert [movie["id"] for movie in movies] == [1]


def test_repeat_person_pages_are_served_from_memory(fake):
    fake.routes["/search/person"] = FakeResponse(PERSON_SEARCH)
    fake.routes["/person/3894"] = FakeResponse(PERSON_DETAILS)
    fake.routes["/person/1"] = FakeResponse({**PERSON_DETAILS, "id": 1})

    actor = people.find_person("Christian Bale", "actor")
    assert people.find_person(" Christian Bale ", "actor") == actor
    assert len(fake.calls) == 2

    # The director page for the same name picks the person and credits differently.
    assert people.find_person("Christian Bale", "director")["id"] == 1
    assert len(fake.calls) == 4


def test_failed_person_lookups_are_not_remembered(fake, monkeypatch):
    fake.routes["/search/person"] = [
        FakeResponse({"results": []}),
        FakeResponse(status_code=502),
        FakeResponse(PERSON_SEARCH),
    ]
    fake.routes["/person/3894"] = FakeResponse(PERSON_DETAILS)

    assert people.find_person("Christian Bale", "actor") is None
    with pytest.raises(tmdb.TmdbUnavailableError):
        people.find_person("Christian Bale", "actor")
    monkeypatch.setattr(tmdb, "_cooldown_until", 0.0)

    assert people.find_person("Christian Bale", "actor")["name"] == "Christian Bale"


def test_person_cache_is_emptied_once_full(fake):
    fake.routes["/search/person"] = FakeResponse(PERSON_SEARCH)
    fake.routes["/person/3894"] = FakeResponse(PERSON_DETAILS)
    people._person_cache.update(
        ((f"filler-{n}", "actor"), {}) for n in range(people.PERSON_CACHE_LIMIT)
    )

    person = people.find_person("Christian Bale", "actor")

    assert people._person_cache == {("Christian Bale", "actor"): person}


def test_person_page_renders_from_canned_tmdb_json(fake, client):
    fake.routes["/search/person"] = FakeResponse(PERSON_SEARCH)
    fake.routes["/person/3894"] = FakeResponse(PERSON_DETAILS)

    response = client.get("/actor/Christian+Bale")

    assert response.status_code == 200
    page = Page(response.get_data(as_text=True))
    assert texts(page, "h1") == ["Christian Bale"]
    assert page.find("a", href="/movie/155")
    assert "30 Ocak 1974" in page.text


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
    page = Page(response.get_data(as_text=True))
    assert texts(page, "h1") == ["Yalnızca TMDb Filmi"]
    assert page.find("a", href="/director/Y%C3%B6netmen%20Ki%C5%9Fi")
    assert [button.attrs["data-trailer-key"] for button in page.find("button", "trailer-play")] == [
        "abc123"
    ]
    assert page.find("a", href="/movie/155")
    assert fake.paths.count("/movie/999001") == 1


def test_tmdb_only_movie_page_is_unavailable_during_an_outage_not_missing(
    fake, client, cold_details
):
    assert client.get("/movie/999009").status_code == 404

    fake.routes["/movie/999001"] = requests.Timeout()
    response = client.get("/movie/999001")

    assert response.status_code == 503
    assert "Film detayı şu anda alınamıyor." in response.get_data(as_text=True)


def test_dataset_movie_details_list_the_first_12_cast_names(cold_details):
    movie = cold_details.get_movie_details(155)

    names = [name.strip() for name in movie["cast"].split(",")]
    assert len(names) == 15
    assert movie["cast_list"] == names[:12]
    assert movie["cast_list"][:2] == ["Christian Bale", "Heath Ledger"]


def test_tmdb_only_movie_details_list_12_cast_names_from_the_credits(fake, cold_details):
    cast = [{"name": f"Oyuncu {number}"} for number in range(1, 17)]
    fake.routes["/movie/999004"] = FakeResponse(
        {
            "id": 999004,
            "title": "Kalabalık",
            "credits": {"cast": [cast[0], {"character": "Adsız"}, {"name": ""}, *cast[1:]]},
        }
    )

    movie = cold_details.get_movie_details(999004)

    assert movie["cast_list"] == [actor["name"] for actor in cast[:12]]
    assert movie["cast"] == ", ".join(actor["name"] for actor in cast[:15])


def test_movie_without_cast_has_an_empty_cast_list(fake, cold_details):
    fake.routes["/movie/999005"] = FakeResponse({"id": 999005, "title": "Sessiz"})

    assert cold_details.get_movie_details(999005)["cast_list"] == []


def test_engine_construction_makes_no_tmdb_call(fake):
    RecommendationEngine(
        str(DATASET_DIR / "tmdb_5000_movies.csv"), str(DATASET_DIR / "tmdb_5000_credits.csv")
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

    assert [(b["display_name"], b["link_name"]) for b in movie["genre_badges"]] == [
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

    page = client.get("/", query_string={"film": "Inception", "n": 3})
    assert page.get_data(as_text=True).count(poster) == 3

    similar = cold_details.get_movie_details(155)["similar_movies"]
    assert [card["poster_url"] for card in similar] == [poster] * DETAIL_SIMILAR_COUNT


# --- Answers of an unexpected shape: one odd field must never break a page ---


@pytest.mark.parametrize(
    ("data", "year"),
    [
        ({"release_date": "2008-07-16"}, 2008),
        ({"release_date": "2008"}, 2008),
        ({"release_date": ""}, 0),
        ({"release_date": None}, 0),
        ({}, 0),
        ({"release_date": "TBA"}, 0),
        ({"release_date": "20x8-01-01"}, 0),
        ({"release_date": 2008}, 0),
        ({"release_date": ["2008-07-16"]}, 0),
    ],
)
def test_release_year_of_an_odd_date_is_unknown(data, year):
    assert tmdb.release_year(data) == year


@pytest.mark.parametrize(
    "videos",
    [
        None,
        "Trailer",
        {"results": []},
        ["junk", 7, None],
        [{"site": "YouTube", "type": "Trailer", "key": 123}],
        [{"site": "YouTube", "type": "Trailer", "key": ""}],
        # The key goes into i.ytimg.com and youtube-nocookie.com URLs.
        [{"site": "YouTube", "type": "Trailer", "key": "../../watch?v=x"}],
        [{"site": "YouTube", "type": "Trailer", "key": 'a" onerror="x'}],
    ],
)
def test_pick_trailer_ignores_videos_of_an_odd_shape(videos):
    assert tmdb.pick_trailer(videos) is None


def test_pick_trailer_skips_odd_videos_before_a_good_one():
    videos = ["junk", {"site": "YouTube", "type": "Trailer", "key": None}, *TURKISH_VIDEOS]

    assert tmdb.pick_trailer(videos) == "tr"


@pytest.mark.parametrize(
    ("path", "url"),
    [
        ("/p.jpg", "https://image.tmdb.org/t/p/w342/p.jpg"),
        (None, None),
        ("", None),
        (7, None),
        (["/p.jpg"], None),
        ("p.jpg", None),
    ],
)
def test_image_url_needs_a_path_starting_with_a_slash(path, url):
    assert tmdb.image_url(path, "w342") == url


ODD_DETAILS = {
    "id": 155,
    "title": ["Kara Şövalye"],
    "overview": 5,
    "tagline": {"tr": "x"},
    "runtime": "uzun",
    "poster_path": 7,
    "videos": ["junk"],
}


def test_dataset_movie_page_survives_an_odd_tmdb_answer(fake, cold_details, client, engine):
    fake.routes["/movie/155"] = FakeResponse(ODD_DETAILS)
    row = engine.movies_df.iloc[engine.row_of[155]]

    movie = cold_details.get_movie_details(155)
    response = client.get("/movie/155")

    # The dataset's own texts stay; nothing of the odd answer is shown.
    assert (movie["title"], movie["overview"], movie["tagline"]) == (
        "The Dark Knight",
        row["overview"],
        row["tagline"],
    )
    assert movie["runtime_formatted"] == "2 sa 32 dk"
    assert movie["trailer_key"] is None
    assert movie["poster_url_large"] is None
    assert response.status_code == 200


ODD_TMDB_ONLY = {
    "id": 999010,
    "title": "Bozuk Film",
    "overview": None,
    "genres": ["Dram", {"id": [18], "name": "Dram"}, {"id": 18, "name": "Dram"}, {"id": 99}],
    "credits": {
        "crew": ["junk", {"job": "Director", "name": 5}, {"job": "Director", "name": "Yön Etmen"}],
        "cast": [{"name": None}, {"name": ["Bir"]}, {"name": "Oyuncu Bir"}, 7],
    },
    "vote_average": "çok iyi",
    "vote_count": [3],
    "popularity": "NaN",
    "release_date": "TBA",
    "runtime": {"minutes": 90},
    "videos": {"results": {"key": "x"}},
}


def test_tmdb_only_movie_page_survives_odd_fields(fake, cold_details, client):
    fake.routes["/movie/999010"] = FakeResponse(ODD_TMDB_ONLY)
    fake.routes["/movie/999010/similar"] = FakeResponse(
        {
            "results": [
                "junk",
                {"id": "155", "title": "Text id"},
                {"id": 155, "title": "The Dark Knight", "release_date": "TBA", "poster_path": 3},
                {"id": 13, "title": ["Forrest Gump"]},
            ]
        }
    )

    movie = cold_details.get_movie_details(999010)
    response = client.get("/movie/999010")

    assert response.status_code == 200
    assert "Bozuk Film" in body_text(response)
    assert [badge["display_name"] for badge in movie["genre_badges"]] == ["Dram"]
    assert movie["directors"] == ["Yön Etmen"]
    assert movie["cast_list"] == ["Oyuncu Bir"]
    assert (movie["year"], movie["vote_average"], movie["vote_count"]) == (0, 0.0, 0)
    assert movie["popularity"] == 0.0
    assert movie["runtime_formatted"] is None
    assert movie["trailer_key"] is None
    assert movie["similar_movies"] == [
        {"id": 155, "title": "The Dark Knight", "poster_url": None, "year": 0}
    ]


@pytest.mark.parametrize("title", [None, "", 5, ["Film"]])
def test_tmdb_only_movie_without_a_usable_title_is_not_found(fake, cold_details, client, title):
    fake.routes["/movie/999012"] = FakeResponse({"id": 999012, "title": title, "overview": "x"})

    assert client.get("/movie/999012").status_code == 404


def test_tmdb_only_movie_without_texts_shows_no_none(fake, cold_details):
    fake.routes["/movie/999015"] = FakeResponse(
        {"id": 999015, "title": "Sessiz", "overview": None, "tagline": None}
    )

    movie = cold_details.get_movie_details(999015)

    assert (movie["overview"], movie["tagline"]) == ("", "")


def test_list_cards_survive_an_odd_tmdb_answer(fake, cold_details, client):
    fake.routes["/movie/999011"] = FakeResponse(
        {"id": None, "title": "Bozuk", "vote_average": "abc", "release_date": "TBA"}
    )
    fake.routes["/movie/999013"] = FakeResponse({"id": 999013, "title": ["x"]})
    fake.routes["/movie/999014"] = FakeResponse({"id": 999014, "title": "İyi", "vote_average": 7.5})

    response = client.get("/api/movies", query_string={"ids": "999011,999013,999014,155"})

    assert response.status_code == 200
    assert [(card["id"], card["year"], card["vote_average"]) for card in response.get_json()] == [
        (999011, 0, 0.0),
        (999014, 0, 7.5),
        (155, 2008, 8.2),
    ]


ODD_PERSON = {
    **PERSON_DETAILS,
    "name": "Christian Bale",
    "biography": None,
    "homepage": 5,
    "birthday": 1974,
    "known_for_department": ["Acting"],
    "profile_path": ["x"],
    "movie_credits": {
        "cast": [
            "junk",
            {"id": None, "title": "No id", "poster_path": "/a.jpg"},
            {"id": [1], "title": "List id", "poster_path": "/b.jpg"},
            {"id": "13", "title": "Text id", "poster_path": "/c.jpg"},
            {
                "id": 155,
                "title": "The Dark Knight",
                "poster_path": "/dk.jpg",
                "release_date": "TBA",
                "vote_count": "many",
                "vote_average": "NaN",
                "character": ["Bruce Wayne"],
            },
            {"id": 272, "title": "Batman Begins", "poster_path": "/bb.jpg", "vote_count": 9000},
        ]
    },
}


def test_person_page_survives_an_odd_tmdb_answer(fake, client):
    fake.routes["/search/person"] = FakeResponse({"results": [{"id": 3894}]})
    fake.routes["/person/3894"] = FakeResponse(ODD_PERSON)

    person = people.find_person("Christian Bale", "actor")
    response = client.get("/actor/Christian+Bale")

    assert response.status_code == 200
    assert [(m["id"], m["year"], m["vote_average"]) for m in person["movies"]] == [
        (272, 0, 0.0),
        (155, 0, 0.0),
    ]
    assert person["movies"][1]["character"] == ""
    assert person["homepage"] is None
    assert person["profile_image_url"] is None
    assert "known_for_label" not in person
    assert person["biography"] == "Biyografi bilgisi bulunmamaktadır."


@pytest.mark.parametrize(
    "results",
    [
        {"id": 3894},
        ["junk", None],
        [{"id": "3894"}],
        [{"id": None, "known_for_department": "Acting"}],
    ],
)
def test_person_search_without_a_usable_id_finds_nobody(fake, results):
    fake.routes["/search/person"] = FakeResponse({"results": results})

    assert people.find_person("Christian Bale", "actor") is None
    assert fake.paths == ["/search/person"]


def test_person_search_skips_results_without_an_id(fake):
    fake.routes["/search/person"] = FakeResponse(
        {
            "results": [
                {"known_for_department": "Acting"},
                {"id": 3894, "known_for_department": "Acting"},
            ]
        }
    )
    fake.routes["/person/3894"] = FakeResponse(PERSON_DETAILS)

    assert people.find_person("Christian Bale", "actor")["id"] == 3894


def test_person_without_movie_credits_has_an_empty_filmography(fake):
    fake.routes["/search/person"] = FakeResponse({"results": [{"id": 3894}]})
    fake.routes["/person/3894"] = FakeResponse({**PERSON_DETAILS, "movie_credits": ["junk"]})

    assert people.find_person("Christian Bale", "actor")["movies"] == []


def test_an_unexpected_error_in_one_poster_lookup_leaves_the_page(
    fake, cold_details, client, caplog
):
    ids = list(range(1, 6))
    _poster_routes(fake, ids)
    fake.routes["/movie/3"] = RuntimeError("bug")
    cards = [{"id": movie_id} for movie_id in ids]

    with caplog.at_level(logging.ERROR):
        cold_details.add_posters(cards)

    assert [card["poster_url"] for card in cards] == [
        None if movie_id == 3 else _poster(movie_id) for movie_id in ids
    ]
    assert "Poster lookup for movie 3 failed" in caplog.text
    assert 3 not in cold_details.poster_cache


def test_home_page_renders_when_every_poster_lookup_fails_unexpectedly(
    fake, cold_details, client, engine
):
    fake.routes.update(
        {f"/movie/{card['id']}": RuntimeError("bug") for card in engine.popular_cards()}
    )

    assert client.get("/").status_code == 200


class ForgetfulCache(dict):
    """A poster cache that another request empties right after every write."""

    def __setitem__(self, key, value):
        pass


def test_a_fetched_poster_is_used_even_when_the_cache_is_emptied_meanwhile(
    fake, details, monkeypatch
):
    fake.routes["/movie/155"] = FakeResponse(DARK_KNIGHT_TR)
    monkeypatch.setattr(details, "poster_cache", ForgetfulCache())

    assert details.get_movie_poster_url(155) == "https://image.tmdb.org/t/p/w342/dk.jpg"
    assert details.get_movie_details(155)["poster_url_large"] == (
        "https://image.tmdb.org/t/p/w500/dk.jpg"
    )
    assert details.get_list_card(155)["poster_url"] == "https://image.tmdb.org/t/p/w342/dk.jpg"


def test_a_slow_tmdb_delays_the_posters_not_the_page(fake, cold_details, monkeypatch):
    from movierec import movie_details as details_module

    slow = SlowSession(1.0)
    monkeypatch.setattr(tmdb, "_session", slow)
    monkeypatch.setattr(details_module, "POSTER_WAIT_SECONDS", 0.1)
    ids = list(range(1, 13))
    _poster_routes(slow, ids)
    cards = [{"id": movie_id} for movie_id in ids]

    started = time.perf_counter()
    cold_details.add_posters(cards)

    assert time.perf_counter() - started < 0.8
    assert [card["poster_url"] for card in cards] == [None] * 12
    # The lookups under way still fill the cache for the next page; the queued ones are
    # dropped instead of piling more work on a slow TMDb.
    time.sleep(1.5)
    assert sorted(cold_details.poster_cache) == ids[:POSTER_WORKERS]
    assert len(slow.calls) == POSTER_WORKERS
