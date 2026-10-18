"""The lists page and /api/movies; the favourites and watchlist themselves live in the browser."""

import re

import pytest
from conftest import FakeResponse

DARK_KNIGHT = {
    "id": 155,
    "title": "The Dark Knight",
    "year": 2008,
    "vote_average": 8.2,
    "poster_url": None,
}
AVATAR = {"id": 19995, "title": "Avatar", "year": 2009, "vote_average": 7.2, "poster_url": None}
TMDB_ONLY = {
    "id": 999001,
    "title": "Yalnızca TMDb Filmi",
    "release_date": "2024-05-01",
    "vote_average": 7.04,
    "poster_path": "/only.jpg",
}


def _cards(client, query):
    response = client.get("/api/movies", query_string={"ids": query})
    assert response.status_code == 200
    return response.get_json()


def test_my_lists_renders_both_sections_empty(client):
    response = client.get("/my_lists")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Beğendiğim Filmler" in body
    assert "Daha Sonra İzleyeceklerim" in body
    assert "Henüz beğendiğiniz bir film bulunmuyor." in body
    assert "İzleme listenizde henüz bir film bulunmuyor." in body
    for list_name, grid_id in [
        ("favorites", "liked-movies-grid"),
        ("watchlist", "watchlist-movies-grid"),
    ]:
        assert re.search(rf'<section [^>]*data-list="{list_name}"', body)
        assert re.search(rf'<div class="movies-grid" id="{grid_id}"[^>]*>\s*</div>', body)
    # The collapse toggles are added by my_lists.js once a list has movies.
    assert "aria-controls" not in body


def test_my_lists_carries_the_card_template_and_loads_its_module(client):
    body = client.get("/my_lists").get_data(as_text=True)

    template = body[body.index("<template") : body.index("</template>")]
    assert 'class="movie-card-personal"' in template
    assert "movie-card-personal-title" in template
    assert "movie-card-personal-rating" in template
    assert "poster-placeholder" in template
    assert re.search(r'<script type="module" src="/static/my_lists\.js\?v=[0-9a-f]{12}">', body)


def test_my_lists_renders_without_the_engine(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "engine", None)

    assert client.get("/my_lists").status_code == 200


def test_api_movies_returns_cards_in_request_order(client):
    assert _cards(client, "155,19995") == [DARK_KNIGHT, AVATAR]
    assert _cards(client, "19995,155") == [AVATAR, DARK_KNIGHT]


def test_api_movies_ignores_tokens_that_are_not_movie_ids(client):
    query = "abc,155,,0,-5,1.5,+7,2147483648,99999999999999999999,١٥٥, 19995 ,x19995"

    assert _cards(client, query) == [DARK_KNIGHT, AVATAR]


def test_api_movies_drops_duplicates(client):
    assert _cards(client, "155,155,19995,155") == [DARK_KNIGHT, AVATAR]


def test_api_movies_skips_unknown_ids_and_looks_up_others_on_tmdb(client, fake, cold_details):
    fake.routes["/movie/999001"] = FakeResponse(TMDB_ONLY)

    cards = _cards(client, "999001,424242,155")

    assert cards == [
        {
            "id": 999001,
            "title": "Yalnızca TMDb Filmi",
            "year": 2024,
            "vote_average": 7.0,
            "poster_url": "https://image.tmdb.org/t/p/w342/only.jpg",
        },
        DARK_KNIGHT,
    ]
    assert "/movie/424242" in fake.paths


def test_api_movies_looks_up_at_most_20_ids_outside_the_dataset(client, fake, cold_details):
    unknown = list(range(900001, 900031))
    for movie_id in unknown:
        fake.routes[f"/movie/{movie_id}"] = FakeResponse({**TMDB_ONLY, "id": movie_id})

    cards = _cards(client, ",".join(map(str, [*unknown, 155])))

    # Dataset ids do not count against the limit, even after it is reached.
    assert [card["id"] for card in cards] == [*unknown[:20], 155]
    assert [path for path in fake.paths if path != "/movie/155"] == [
        f"/movie/{movie_id}" for movie_id in unknown[:20]
    ]


@pytest.mark.parametrize("query_string", [{}, {"ids": ""}, {"ids": "abc,0"}])
def test_api_movies_without_ids_returns_an_empty_list(client, query_string):
    response = client.get("/api/movies", query_string=query_string)

    assert response.status_code == 200
    assert response.get_json() == []


def test_api_movies_accepts_100_ids(client):
    response = client.get("/api/movies", query_string={"ids": ",".join(map(str, range(1, 101)))})

    assert response.status_code == 200
    assert all(card["id"] <= 100 for card in response.get_json())


def test_api_movies_counts_distinct_ids_against_the_limit(client):
    assert _cards(client, ",".join(["155"] * 150)) == [DARK_KNIGHT]


def test_api_movies_rejects_more_than_100_ids(client):
    response = client.get("/api/movies", query_string={"ids": ",".join(map(str, range(1, 102)))})

    assert response.status_code == 400
    assert response.get_json() == {"status": "error", "error": "bad_request"}


def test_api_movies_without_engine_returns_503_json(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "engine", None)

    response = client.get("/api/movies", query_string={"ids": "155"})

    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "error": "unavailable"}
