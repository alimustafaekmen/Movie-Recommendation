"""The lists page and /api/movies; the favourites and watchlist themselves live in the browser."""

import re
from pathlib import Path

import pytest
from conftest import FakeResponse
from flask import render_template_string, url_for
from pages import Element, Page, body_text

STATIC = Path(__file__).resolve().parents[1] / "static"

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
    assert "Beğendiğim filmler" in body
    assert "Daha sonra izleyeceklerim" in body
    assert "Henüz beğendiğiniz bir film bulunmuyor." in body
    assert "İzleme listenizde henüz bir film bulunmuyor." in body
    page = Page(body)
    for list_name, grid_id in [("favorites", "favorites-grid"), ("watchlist", "watchlist-grid")]:
        [section] = page.find("section", data_list=list_name)
        [grid] = section.find("ul", id=grid_id)
        # Empty and hidden until my_lists.js fills it.
        assert "movie-grid" in grid.classes
        assert "hidden" in grid.attrs
        assert grid.text == "" and not grid.find()
    # The collapse toggles are added by my_lists.js once a list has movies.
    assert not [tag for _, tag in page.tags if "aria-controls" in tag]


def test_my_lists_empty_messages_say_where_lists_live_and_link_home(client):
    body = body_text(client.get("/my_lists"))

    # Plain panels, hidden until my_lists.js finds a list empty: the movie page's button labels
    # in quotes, and a link home.
    messages = Page(body).find("div", "empty-list")
    assert len(messages) == 2
    for message, button in zip(messages, ("Beğen", "İzleme listesine ekle"), strict=True):
        assert "panel" in message.classes
        assert "hidden" in message.attrs
        assert f"Film sayfalarındaki “{button}” düğmesiyle ekleyebilirsiniz." in message.text
        assert "Listeler bu tarayıcıda saklanır." in message.text
        [link] = message.find("a")
        assert link.attrs["href"] == "/"
        assert message.text.endswith("Film keşfetmeye başla")


def test_my_lists_headings_are_one_h1_and_an_h2_per_list(client):
    page = Page(body_text(client.get("/my_lists")))
    headings = [element for element in page.root.descendants() if element.tag in ("h1", "h2")]

    [h1] = page.find("h1")
    assert "page-title" in h1.classes
    assert [(heading.tag, heading.text) for heading in headings] == [
        ("h1", "Kişisel film listelerim"),
        ("h2", "Beğendiğim filmler"),
        ("h2", "Daha sonra izleyeceklerim"),
        ("h2", "Beğendiklerinize göre"),
    ]
    # Plain headings; my_lists.js adds only the collapse chevron.
    assert not any(heading.find("svg") for heading in headings)


def test_my_lists_recommendations_wait_hidden_after_the_lists(client):
    page = Page(body_text(client.get("/my_lists")))

    # my_lists.js shows the section once the favourites give it films to show.
    [section] = page.find("section", "recommendations")
    assert "hidden" in section.attrs
    assert "data-list" not in section.attrs
    [hint] = section.find("p", "recommendations-hint")
    assert "Beğendiğiniz filmlere" in hint.text
    [grid] = section.find("ul", id="recommendations-grid")
    assert "movie-grid" in grid.classes
    assert not grid.find()
    sections = page.find("section")
    assert sections.index(section) == len(sections) - 1


def _markup(root, dropped=()):
    """(tag, attributes) of every element in `root` (HTML or an element), in order, without
    the per-movie attributes and without the elements of a `dropped` class and their content."""
    skip = {"href", "src", "alt"}
    found = []

    def walk(element):
        for child in element.children:
            if isinstance(child, Element) and not set(child.classes) & set(dropped):
                found.append((child.tag, {k: v for k, v in child.attrs.items() if k not in skip}))
                walk(child)

    walk(Page(root).root if isinstance(root, str) else root)
    return found


def _macro_card(app_module, poster_url):
    movie = {"id": 1, "title": "T", "year": 2000, "vote_average": 7.5, "poster_url": poster_url}
    with app_module.app.test_request_context():
        return render_template_string(
            '{% from "_movie_card.html" import movie_card %}{{ movie_card(movie) }}', movie=movie
        )


def test_my_lists_card_template_has_the_movie_card_markup(client, app_module):
    [template] = Page(body_text(client.get("/my_lists"))).find("template", id="movie-card-template")

    # One list item per card; my_lists.js keeps either the poster or the placeholder with its
    # icon, as the macro does.
    [card] = [child for child in template.children if isinstance(child, Element)]
    assert card.tag == "li"
    assert _markup(card, dropped=("movie-card-placeholder",)) == _markup(
        _macro_card(app_module, "https://image.tmdb.org/t/p/w342/poster.jpg")
    )
    assert _markup(card, dropped=("movie-card-poster",)) == _markup(_macro_card(app_module, None))


def test_my_lists_toggle_chevron_waits_in_a_template(client):
    page = Page(body_text(client.get("/my_lists")))

    # my_lists.js adds this chevron to the collapse button of each list with movies.
    [template] = page.find("template", id="list-toggle-chevron-template")
    span, svg, use = template.find()
    assert (span.tag, svg.tag, use.tag) == ("span", "svg", "use")
    assert "list-toggle-chevron" in span.classes
    assert svg.attrs["aria-hidden"] == "true"
    assert use.attrs["href"].endswith("#chevron-down")


def test_my_lists_takes_its_urls_from_the_server(client, app_module):
    body = body_text(client.get("/my_lists"))

    # my_lists.js fetches the cards from data-cards-url and links each card to data-movie-url
    # with the movie's id in place of the 0, so no script knows the site's routes.
    [page] = [tag for tag in Page(body).all("div") if "my-lists-page" in tag.get("class", "")]
    with app_module.app.test_request_context():
        assert page["data-cards-url"] == url_for("lists.api_movies")
        assert page["data-recommendations-url"] == url_for("lists.api_recommendations")
        assert page["data-movie-url"] == url_for("movies.movie_detail", movie_id=0)


def test_scripts_hard_code_no_path_of_this_site():
    scripts = list((STATIC / "js").glob("*.js"))
    assert scripts
    for script in scripts:
        code = script.read_text(encoding="utf-8")
        # Comments may name paths; "https://" is not a comment.
        code = re.sub(r"/\*.*?\*/|(?<![:\w])//[^\n]*", "", code, flags=re.S)
        assert not re.findall(r"""["'`]/(?![/*])""", code), script.name


def test_my_lists_loads_its_module_before_the_first_frame(client):
    body = body_text(client.get("/my_lists"))
    head = body[: body.index("</head>")]

    # Render-blocking, so the placeholders or empty messages are in the first frame.
    [script] = [tag for tag in Page(head).all("script") if "my_lists.js" in tag.get("src", "")]
    assert script["type"] == "module"
    assert script["blocking"] == "render"
    assert re.fullmatch(r"/static/js/my_lists\.js\?v=[0-9a-f]{12}", script["src"])
    assert body.count("my_lists.js") == 1


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


def test_api_movies_remembers_cards_and_unknown_ids_outside_the_dataset(client, fake, cold_details):
    fake.routes["/movie/999001"] = FakeResponse({**TMDB_ONLY, "poster_path": None})

    first = _cards(client, "999001,424242")
    second = _cards(client, "999001,424242")

    assert first == second
    assert [(card["id"], card["poster_url"]) for card in first] == [(999001, None)]
    # A film without a poster is still a film; only TMDb's 404 marks an id as unknown.
    assert fake.paths == ["/movie/999001", "/movie/424242"]


def test_api_movies_remembers_cards_per_language(client, fake, cold_details):
    fake.routes["/movie/999001"] = FakeResponse(TMDB_ONLY)

    _cards(client, "999001")
    client.get("/api/movies", query_string={"ids": "999001", "lang": "en"})

    assert [params["language"] for _, params in fake.calls] == ["tr-TR", "en-US"]


def test_api_movies_looks_up_an_id_again_after_tmdb_failed(client, fake, cold_details):
    # An answer that is not a JSON object is a failure, like a timeout.
    fake.routes["/movie/999001"] = [FakeResponse(["broken"]), FakeResponse(TMDB_ONLY)]

    assert _cards(client, "999001") == []
    assert [card["id"] for card in _cards(client, "999001")] == [999001]


def test_the_list_card_cache_is_emptied_when_full(fake, cold_details, monkeypatch):
    monkeypatch.setattr("movierec.movie_details.LIST_CARD_CACHE_LIMIT", 2)
    for movie_id in (999001, 999002, 999003):
        fake.routes[f"/movie/{movie_id}"] = FakeResponse({**TMDB_ONLY, "id": movie_id})
        cold_details.get_list_card(movie_id, "tr")

    assert list(cold_details.list_card_cache) == [(999003, "tr")]


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


def test_api_movies_stops_looking_up_ids_on_tmdb_after_its_time_budget(
    client, fake, cold_details, monkeypatch
):
    from movierec.web import lists

    # Each TMDb answer takes 3 s on this clock; the budget is 8 s.
    clock = [0.0]
    monkeypatch.setattr(lists, "monotonic", lambda: clock[0])
    monkeypatch.setattr(lists, "TMDB_LOOKUP_SECONDS", 8.0)
    answer = fake.get

    def slow_get(url, params=None, timeout=None):
        clock[0] += 3
        return answer(url, params, timeout)

    monkeypatch.setattr(fake, "get", slow_get)
    unknown = list(range(900001, 900007))
    for movie_id in unknown:
        fake.routes[f"/movie/{movie_id}"] = FakeResponse({**TMDB_ONLY, "id": movie_id})
    # The dataset film's poster is known, so only the lookups below take time.
    cold_details.poster_cache[(155, "tr")] = None

    cards = _cards(client, ",".join(map(str, [*unknown, 155])))

    # Lookups start at 0, 3 and 6 s; the fourth would start at 9 s. Dataset ids still come.
    assert [card["id"] for card in cards] == [*unknown[:3], 155]


def _recommended(client, **query):
    response = client.get("/api/recommendations", query_string=query)
    assert response.status_code == 200
    return response.get_json()


def test_api_recommendations_are_list_cards_of_the_engine_ranking(client, engine, details):
    cards = _recommended(client, favorites="155,19995", watchlist="")

    ids = engine.recommend_from_favourites([155, 19995], [])
    assert len(ids) == 12
    assert cards == [details.get_list_card(movie_id, "tr") for movie_id in ids]


def test_api_recommendations_leave_out_the_watchlist(client, engine):
    top = engine.recommend_from_favourites([155], [])[0]

    cards = _recommended(client, favorites="155", watchlist=f"abc,{top}")

    assert [card["id"] for card in cards] == engine.recommend_from_favourites([155], [top])


@pytest.mark.parametrize(
    "query", [{}, {"favorites": ""}, {"favorites": "abc,0,-5"}, {"favorites": "424242"}]
)
def test_api_recommendations_without_dataset_favourites_are_empty(client, query):
    assert _recommended(client, **query, watchlist="155") == []


def test_api_recommendations_read_at_most_100_ids_per_list(client, engine):
    unknown = ",".join(map(str, range(900001, 900101)))
    top = engine.recommend_from_favourites([155], [])[0]

    # Ids after the 100th of a list are left out, not refused.
    assert _recommended(client, favorites=f"{unknown},155") == []
    cards = _recommended(client, favorites="155", watchlist=f"{unknown},{top}")
    assert cards[0]["id"] == top


def test_api_recommendations_without_engine_return_503_json(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "engine", None)

    response = client.get("/api/recommendations", query_string={"favorites": "155"})

    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "error": "unavailable"}
