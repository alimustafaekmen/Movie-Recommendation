"""The movie and person pages: media, meta, links, sections, cards and the list buttons."""

import re
from urllib.parse import urlsplit

import pytest
from conftest import PERSON
from flask import request
from pages import Page, body_text, texts

POSTER = "https://image.tmdb.org/t/p/w500/poster.jpg"
TRAILER_KEY = "abc123XYZ_-"
BIOGRAPHY_TAIL = "He is known for intense roles."
LONG_BIOGRAPHY = "Christian Bale is an English actor. " * 30 + BIOGRAPHY_TAIL
CAST = [f"Oyuncu {number}" for number in range(1, 16)]
# Page.all() keyword arguments for the shared card grid and its cards.
GRID = {"class": "movie-grid"}
CARD = {"class": "movie-card"}
# Filmography cards as movierec.people builds them: a year, and the character for actors.
ACTOR = {**PERSON, "movies": [{**PERSON["movies"][0], "year": 2008}]}
DIRECTOR = {
    **PERSON,
    "name": "Christopher Nolan",
    "known_for_department": "Directing",
    "movies": [
        {"id": 27205, "title": "Inception", "poster_url": None, "year": 2010, "vote_average": 8.369}
    ],
}


def _section(body, heading):
    """The markup from the h2 with this text up to the next h2."""
    start = re.search(rf"<h2[^>]*>(?:(?!</h2>).)*{heading}", body, re.S).start()
    end = body.find("<h2", start + 1)
    return body[start : end if end != -1 else len(body)]


def _classes(attrs):
    return attrs.get("class", "").split()


def _icon_names(markup):
    """The sprite symbol each icon in the markup (HTML or an element) draws, in order."""
    page = Page(markup) if isinstance(markup, str) else markup
    return [urlsplit(use.attrs["href"]).fragment for use in page.find("use")]


def _patch_movie(details, monkeypatch, **fields):
    original = details.get_movie_details

    def patched(movie_id, lang):
        movie = original(movie_id, lang)
        movie.update(fields)
        return movie

    monkeypatch.setattr(details, "get_movie_details", patched)


def _patch_person(monkeypatch, person):
    monkeypatch.setattr(
        "movierec.web.movies.find_person", lambda name, person_type, lang: dict(person)
    )


def test_movie_page_renders_the_poster_eagerly(client, details, monkeypatch):
    _patch_movie(details, monkeypatch, poster_url_large=POSTER)
    page = Page(body_text(client.get("/movie/155")))

    [poster] = page.all("img", src=POSTER)
    assert poster.get("loading") != "lazy"
    assert poster["alt"] == "The Dark Knight posteri"


def test_movie_page_shows_a_trailer_thumbnail_instead_of_the_player(client, details, monkeypatch):
    _patch_movie(details, monkeypatch, trailer_key=TRAILER_KEY)
    body = body_text(client.get("/movie/155"))
    section = _section(body, "Fragman")
    page = Page(section)

    # No YouTube player until the visitor asks for it; movie_detail.js inserts it.
    assert "<iframe" not in body
    [button] = page.all("button", **{"class": "trailer-play"})
    assert button["type"] == "button"
    assert button["data-trailer-key"] == TRAILER_KEY
    assert button["data-trailer-title"] == "The Dark Knight fragmanı"
    assert button["aria-label"] == "Fragmanı oynat"
    [thumbnail] = page.all("img")
    assert thumbnail["src"] == f"https://i.ytimg.com/vi/{TRAILER_KEY}/hqdefault.jpg"
    assert thumbnail["alt"] == ""
    assert thumbnail["loading"] == "lazy"
    noscript = section[section.index("<noscript>") : section.index("</noscript>")]
    [link] = Page(noscript).all("a")
    assert link["href"] == f"https://www.youtube.com/watch?v={TRAILER_KEY}"


def test_movie_page_meta_uses_turkish_number_formats(client, details, monkeypatch):
    _patch_movie(details, monkeypatch, vote_average=8.25, vote_count=12002)

    [meta] = texts(body_text(client.get("/movie/155")), "ul", "movie-detail-meta")

    # "2008 · 2 sa 32 dk · ★ 8,3/10 · 12.002 oy": the dots are CSS, so not in the text.
    assert meta == "2008 2 sa 32 dk Puan: 8,3 /10 12.002 oy"


def test_movie_page_meta_is_a_meta_line_whose_only_icon_is_the_rating_star(client):
    [meta] = Page(body_text(client.get("/movie/155"))).find("ul", "movie-detail-meta")

    assert "meta-line" in meta.classes
    assert _icon_names(meta) == ["star"]
    [star] = meta.find("svg")
    assert "rating-star" in star.classes
    assert [item.classes for item in meta.find("li")] == [[], [], ["rating"], []]


def test_movie_page_meta_leaves_out_what_is_unknown(client, details, monkeypatch):
    _patch_movie(details, monkeypatch, year=0, runtime_formatted=None, vote_average=0, vote_count=0)

    page = Page(body_text(client.get("/movie/155")))

    assert not page.find(class_="movie-detail-meta")


def test_movie_page_title_is_the_display_page_title_with_the_tagline(client, details, monkeypatch):
    _patch_movie(details, monkeypatch, tagline="Welcome to a world without rules.")
    body = body_text(client.get("/movie/155"))

    [h1] = Page(body).all("h1")
    assert "page-title" in _classes(h1)
    # The tagline is a plain line under the title, not a quotation.
    assert texts(body, "p", "movie-tagline") == ["Welcome to a world without rules."]


def test_movie_page_list_buttons_follow_the_title_and_meta(client):
    page = Page(body_text(client.get("/movie/155")))
    order = list(page.root.descendants())

    # Next to the content, not under the poster: after the meta line, before the first section.
    positions = [
        order.index(page.find(class_="detail-media")[0]),
        order.index(page.find("h1")[0]),
        order.index(page.find(class_="movie-detail-meta")[0]),
        order.index(page.find(class_="movie-actions")[0]),
        order.index(page.find("h2")[0]),
    ]
    assert positions == sorted(positions)


def test_movie_page_sections_are_h2_and_only_cards_have_h3(client, details, monkeypatch):
    _patch_movie(details, monkeypatch, trailer_key=TRAILER_KEY)
    body = body_text(client.get("/movie/155"))

    assert texts(body, "h1") == ["The Dark Knight"]
    # Plain headings: no icon leads a section.
    assert all(not _icon_names(h2) for h2 in Page(body).find("h2"))
    assert texts(body, "h2") == [
        "Özet",
        "Türler",
        "Yönetmen",
        "Başlıca oyuncular",
        "Fragman",
        "Benzer filmler",
    ]
    assert {tag.get("class") for tag in Page(body).all("h3")} == {"movie-card-title"}


def test_movie_page_says_when_there_is_no_trailer(client):
    body = body_text(client.get("/movie/155"))
    section = _section(body, "Fragman")

    assert texts(section, "p", "detail-note") == ["Bu film için fragman bulunamadı."]
    assert not _icon_names(section)
    assert not Page(body).find(class_="trailer-play")


def test_movie_page_links_every_director_and_turkish_genre(client):
    body = body_text(client.get("/movie/603"))
    page = Page(body)

    for href in (
        "/director/Lana%20Wachowski",
        "/director/Lilly%20Wachowski",
        "/actor/Keanu%20Reeves",
    ):
        assert page.find("a", href=href), href
    assert texts(page, "a", href="/?tur=Action") == ["Aksiyon"]
    assert texts(page, "a", href="/?tur=Science+Fiction") == ["Bilim Kurgu"]
    assert "genre=" not in body


def test_person_and_genre_links_keep_plus_signs_and_spaces(
    app_module, client, details, monkeypatch
):
    _patch_movie(
        details,
        monkeypatch,
        directors=["Jean+Pierre Jeunet"],
        cast_list=["Zoë Saldaña", "C+D 100%"],
        genre_badges=[{"display_name": "Bilim+Kurgu", "link_name": "Sci+Fi X"}],
    )
    asked = []
    monkeypatch.setattr(
        "movierec.web.movies.find_person",
        lambda name, person_type, lang: asked.append((person_type, name)),
    )
    links = Page(body_text(client.get("/movie/155"))).all("a")
    hrefs = [link["href"] for link in links if {"tag", "link"} & set(_classes(link))]

    # Encoded once by url_for, decoded once by Flask.
    assert hrefs == [
        "/?tur=Sci%2BFi+X",
        "/director/Jean+Pierre%20Jeunet",
        "/actor/Zo%C3%AB%20Salda%C3%B1a",
        "/actor/C+D%20100%25",
    ]
    with app_module.app.test_request_context(hrefs[0]):
        assert request.args.getlist("tur") == ["Sci+Fi X"]
    for href in hrefs[1:]:
        client.get(href)
    assert asked == [
        ("director", "Jean+Pierre Jeunet"),
        ("actor", "Zoë Saldaña"),
        ("actor", "C+D 100%"),
    ]


def test_movie_page_links_the_cast_list(client, details, monkeypatch):
    _patch_movie(details, monkeypatch, cast_list=CAST[:12], cast="Somebody Else")

    section = _section(body_text(client.get("/movie/155")), "Başlıca oyuncular")

    assert texts(section, "a", "link") == CAST[:12]


def test_movie_page_genres_are_tags_and_people_are_plain_name_links(client, details):
    movie = details.get_movie_details(155, "tr")

    body = body_text(client.get("/movie/155"))

    genres = [genre["display_name"] for genre in movie["genre_badges"]]
    assert texts(_section(body, "Türler"), "a", "tag") == genres
    assert Page(_section(body, "Türler")).all("ul", **{"class": "tag-list"})
    for heading, names in (
        ("Yönetmen", movie["directors"]),
        ("Başlıca oyuncular", movie["cast_list"]),
    ):
        section = _section(body, heading)
        # Names in a row (base.css underlines them), not tags or pills.
        assert texts(section, "a", "link") == names
        assert all(_classes(link) == ["link"] for link in Page(section).all("a"))


def test_movie_page_without_cast_has_no_cast_section(client, details, monkeypatch):
    _patch_movie(details, monkeypatch, cast_list=[], cast="")

    page = Page(body_text(client.get("/movie/155")))

    assert "Başlıca oyuncular" not in texts(page, "h2")


def test_similar_movies_use_the_shared_card_grid(client, details):
    similar = details.get_movie_details(155, "tr")["similar_movies"]

    section = _section(body_text(client.get("/movie/155")), "Benzer filmler")
    page = Page(section)

    assert page.all("ul", **GRID)
    assert [card["href"] for card in page.all("a", **CARD)] == [
        f"/movie/{movie['id']}" for movie in similar
    ]
    assert texts(section, "h3", "movie-card-title") == [movie["title"] for movie in similar]


def test_person_page_title_names_the_person_and_the_site(person_client):
    body = body_text(person_client.get("/actor/Christian+Bale"))

    assert texts(body, "title") == ["Christian Bale · MovieRec"]


def test_person_page_role_comes_from_the_page_type(client, monkeypatch):
    _patch_person(monkeypatch, DIRECTOR)

    body = body_text(client.get("/director/Christopher+Nolan"))

    assert texts(body, "p", "person-role") == ["Yönetmen"]
    # The department TMDb knows them for says the same, so it is not repeated.
    assert not Page(body).find(class_="person-known-for")


def test_person_page_adds_a_different_known_for_label_as_a_second_line(client, monkeypatch):
    _patch_person(monkeypatch, {**DIRECTOR, "known_for_label": "Senaryo"})

    body = body_text(client.get("/director/Christopher+Nolan"))

    assert texts(body, "p", "person-role") == ["Yönetmen"]
    assert texts(body, "p", "person-known-for") == ["Tanındığı alan: Senaryo"]


def test_person_page_without_a_known_for_label_shows_only_the_role(person_client):
    body = body_text(person_client.get("/actor/Christian+Bale"))

    assert texts(body, "p", "person-role") == ["Oyuncu"]
    assert not Page(body).find(class_="person-known-for")
    assert "Acting" not in body


def test_person_page_sections_are_h2_and_only_cards_have_h3(person_client):
    body = body_text(person_client.get("/actor/Christian+Bale"))

    assert texts(body, "h1") == ["Christian Bale"]
    [h1] = Page(body).all("h1")
    assert "page-title" in _classes(h1)
    assert texts(body, "h2") == ["Biyografi", "Filmografi"]
    assert all(not _icon_names(h2) for h2 in Page(body).find("h2"))
    assert {tag.get("class") for tag in Page(body).all("h3")} == {"movie-card-title"}


def test_actor_filmography_uses_the_shared_card_with_the_character(client, monkeypatch):
    _patch_person(monkeypatch, ACTOR)

    section = _section(body_text(client.get("/actor/Christian+Bale")), "Filmografi")

    assert Page(section).all("ul", **GRID)
    assert [card["href"] for card in Page(section).all("a", **CARD)] == ["/movie/155"]
    assert texts(section, "h3", "movie-card-title") == ["The Dark Knight"]
    assert texts(section, "p", "movie-card-meta") == ["2008 Puan: 8,2"]
    assert texts(section, "p", "movie-card-note") == ["Rol: Bruce Wayne"]


def test_actor_filmography_card_without_a_character_has_no_note(client, monkeypatch):
    _patch_person(monkeypatch, {**ACTOR, "movies": [{**ACTOR["movies"][0], "character": ""}]})

    section = _section(body_text(client.get("/actor/Christian+Bale")), "Filmografi")

    assert "movie-card-note" not in section


def test_director_filmography_cards_have_no_note(client, monkeypatch):
    _patch_person(monkeypatch, DIRECTOR)

    section = _section(body_text(client.get("/director/Christopher+Nolan")), "Filmografi")

    assert texts(section, "p", "movie-card-meta") == ["2010 Puan: 8,4"]
    assert "movie-card-note" not in section
    assert "movie-card-placeholder" in section


def test_person_page_without_movies_says_so(person_client, monkeypatch):
    _patch_person(monkeypatch, {**PERSON, "movies": []})

    body = body_text(person_client.get("/actor/Christian+Bale"))

    assert "Bu oyuncu için filmografi bilgisi bulunamadı." in _section(body, "Filmografi")
    assert not Page(body).find(class_="movie-grid")


def test_person_page_lists_the_known_dates_and_homepage(person_client, monkeypatch):
    _patch_person(
        monkeypatch,
        {**PERSON, "deathday": "2050-02-01", "homepage": "https://example.com/bale"},
    )

    body = body_text(person_client.get("/actor/Christian+Bale"))

    assert texts(body, "dt") == ["Doğum tarihi", "Ölüm tarihi", "Web sitesi"]
    assert texts(body, "dd") == [
        "30 Ocak 1974 (Haverfordwest, Wales, UK)",
        "01 Şubat 2050",
        "https://example.com/bale",
    ]
    [link] = Page(body).all("a", href="https://example.com/bale")
    assert link["rel"] == "noopener noreferrer"


def test_person_page_shows_a_long_biography_in_full_behind_a_details_element(
    person_client, monkeypatch
):
    _patch_person(monkeypatch, {**PERSON, "biography": LONG_BIOGRAPHY})

    page = Page(body_text(person_client.get("/actor/Christian+Bale")))

    [details] = page.find("details")
    assert texts(details, "summary") == ["Devamını oku"]
    assert details.text.endswith(BIOGRAPHY_TAIL)
    assert "Bruce Wayne" in page.text


def test_person_page_shows_a_short_biography_without_a_toggle(person_client):
    page = Page(body_text(person_client.get("/actor/Christian+Bale")))

    assert PERSON["biography"] in texts(page, "p", "detail-text")
    assert not page.find("details")


@pytest.mark.parametrize("path", ["/movie/155", "/my_lists"])
def test_list_pages_say_that_the_lists_need_javascript(client, path):
    body = body_text(client.get(path))

    assert body.count("<noscript>") == 1
    assert "JavaScript gerekir" in body[body.index("<noscript>") : body.index("</noscript>")]


def test_movie_page_renders_both_list_buttons_inactive(client):
    body = body_text(client.get("/movie/155"))
    page = Page(body)

    [like] = [tag for tag in page.all("button") if "btn-like" in tag.get("class", "")]
    [watch] = [tag for tag in page.all("button") if "btn-watchlist" in tag.get("class", "")]
    for button, label in ((like, "Beğen"), (watch, "İzleme listesine ekle")):
        assert button["type"] == "button"
        assert {"btn", "btn-secondary"} <= set(_classes(button))
        assert button["data-movie-id"] == "155"
        # The label alone says the state (movie_detail.js swaps it), so no aria-pressed.
        assert "aria-pressed" not in button
        assert button["title"] == label
    # Inactive buttons draw the outline icons; movie_detail.js swaps in the solid ones.
    [like_button, watch_button] = page.find("button", "action-btn")
    assert (texts(like_button, "span"), texts(watch_button, "span")) == (
        ["Beğen"],
        ["İzleme listesine ekle"],
    )
    assert _icon_names(like_button) == ["heart-regular"]
    assert _icon_names(watch_button) == ["bookmark-regular"]
    assert [tag for tag in page.all("p") if tag.get("class") == "action-status"]
    # Render-blocking in the head, so a liked film never shows "Beğen" for a frame first.
    head = body[: body.index("</head>")]
    [script] = [
        tag
        for tag in Page(head).all("script")
        if "/static/js/movie_detail.js?" in tag.get("src", "")
    ]
    assert script["type"] == "module"
    assert script["blocking"] == "render"
    assert body.count("movie_detail.js") == 1
