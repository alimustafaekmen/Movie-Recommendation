"""The home page: popular movies, the tabbed search panel, shareable GET searches and results."""

import datetime
import re
from urllib.parse import urlsplit

import pytest
from conftest import SEARCHES
from pages import Page, body_text, texts

TAGLINE = "Size en uygun filmleri keşfedin"
# Searches that find nothing, in both modes, and the message that explains why.
NO_RESULTS = [
    (SEARCHES["empty"], "Seçtiğiniz kriterlere tam olarak uyan hiçbir film bulunamadı."),
    ("/?film=xyzzyplugh", "'xyzzyplugh' ismine benzer bir film bulunamadı."),
    ("/?film=", "Lütfen bir film adı girin."),
    ("/?yonetmen=&min_yil=&max_yil=&min_puan=", "en az bir kriter"),
]


def _get(client, url="/"):
    body = body_text(client.get(url))
    return body, Page(body)


def _card_ids(page):
    return [
        int(card["href"].removeprefix("/movie/"))
        for card in page.all("a", **{"class": "movie-card"})
    ]


def _with_role(page, role):
    return [attrs for _, attrs in page.tags if attrs.get("role") == role]


def _text(body):
    """The page text with tags dropped and spaces collapsed."""
    return Page(body).text


def _headings(body):
    """The text of every h2 on the page."""
    return texts(body, "h2")


def _classes(attrs):
    return attrs.get("class", "").split()


def _more_link(page):
    return [link for link in page.all("a") if "more-results" in _classes(link)]


def _toggles(page):
    return [button for button in page.all("button") if "search-toggle" in _classes(button)]


def _hint(body):
    return texts(body, class_="empty-state-hint")


def test_home_page_sets_no_session_cookie(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "Set-Cookie" not in response.headers


def test_home_page_answers_only_get(client):
    assert client.post("/", data={"film": "Avatar"}).status_code == 405


def test_home_page_stays_small_and_loads_suggestions_lazily(client):
    body, page = _get(client)

    assert len(body.encode()) < 60_000
    assert page.all("datalist", id="movie-titles", data_src="/api/titles")
    assert page.all("datalist", id="director-names", data_src="/api/directors")
    assert body.count("<option") < 20


def test_search_forms_are_get_forms_with_one_input_per_field(client):
    _, page = _get(client)

    for form_id in ("similarity-form", "feature-form"):
        [form] = page.all("form", id=form_id)
        assert form["action"] == "/", form_id
        assert form.get("method", "get").lower() == "get", form_id
    [film] = page.all("input", name="film")
    assert film["list"] == "movie-titles"
    assert "required" in film
    [director] = page.all("input", name="yonetmen")
    assert director["list"] == "director-names"
    # The rating threshold is the only select; films and directors are typed.
    assert [select["name"] for select in page.all("select")] == ["min_puan"]
    # The result count is the "more" link's business, not a form field.
    assert not [tag for _, tag in page.tags if tag.get("name") == "n"]


def test_years_and_rating_are_validated_by_the_browser(client):
    _, page = _get(client)
    year = str(datetime.date.today().year)

    for name in ("min_yil", "max_yil"):
        [field] = page.all("input", name=name)
        assert (field["type"], field["min"], field["max"]) == ("number", "1900", year), name
    assert [option["value"] for option in page.all("option")] == [
        "",
        "5",
        "6",
        "7",
        "7.5",
        "8",
        "8.5",
    ]


def test_every_search_field_has_a_label(client):
    _, page = _get(client)

    for field in page.all("input") + page.all("select"):
        if field.get("type") in ("checkbox", "hidden") or field.get("name") == "search_query":
            continue
        assert page.find("label", for_=field["id"]), field["id"]


def test_filter_fields_read_clearly(client):
    _, page = _get(client)

    [label] = texts(page, "label", for_="min_puan")
    assert label.endswith("(TMDB)")
    assert texts(page, "option", value="") == ["Fark etmez"]
    # Example placeholders must not look like years that are already filled in.
    for name in ("min_yil", "max_yil"):
        [field] = page.all("input", name=name)
        assert re.fullmatch(r"örn\. \d{4}", field["placeholder"]), name


def test_genre_filters_show_turkish_labels_and_submit_english_values(client):
    _, page = _get(client)

    [box] = page.all("input", name="tur", value="Science Fiction")
    assert texts(page, "label", for_=box["id"]) == ["Bilim Kurgu"]


@pytest.mark.parametrize(
    ("url", "selected"),
    [("/", 0), (SEARCHES["results"], 0), ("/?tur=Action", 1), (SEARCHES["empty"], 1)],
)
def test_search_tabs_follow_the_aria_tabs_pattern(client, url, selected):
    body, page = _get(client, url)
    tabs = _with_role(page, "tab")
    panels = _with_role(page, "tabpanel")

    assert len(_with_role(page, "tablist")) == 1
    assert [tab["aria-controls"] for tab in tabs] == [panel["id"] for panel in panels]
    assert [tab["id"] for tab in tabs] == [panel["aria-labelledby"] for panel in panels]
    # base.js shows the selected tab's form and hides the other one.
    assert [tab["aria-selected"] for tab in tabs] == [str(i == selected).lower() for i in range(2)]
    assert [tab["tabindex"] for tab in tabs] == ["0" if i == selected else "-1" for i in range(2)]
    # Underline tabs with plain labels: no icons.
    assert texts(page, "button", role="tab") == ["Benzer film", "Filtrele"]
    assert not any(tab.find("svg") for tab in page.find(role="tab"))


@pytest.mark.parametrize("url", ["/", SEARCHES["results"], SEARCHES["empty"]])
def test_both_searches_work_without_javascript(client, url):
    _, page = _get(client, url)

    # The tabs and the fold toggle only work with base.js, which reveals them; until
    # then both forms are on the page and nothing is folded.
    assert all("hidden" in tablist for tablist in _with_role(page, "tablist"))
    assert all("hidden" in toggle for toggle in _toggles(page))
    assert not any("hidden" in panel for panel in _with_role(page, "tabpanel"))
    assert not page.all("button", aria_expanded="false")


@pytest.mark.parametrize("url", [SEARCHES["results"], SEARCHES["warning"]])
def test_result_pages_can_fold_the_search_panel(client, url):
    _, page = _get(client, url)

    [toggle] = _toggles(page)
    [button] = page.find("button", "search-toggle")
    assert "Aramayı düzenle" in button.text
    assert toggle["aria-expanded"] == "true"
    # A quiet button: the chevron is its only icon and shows the folded state.
    assert {"btn", "btn-secondary"} <= set(_classes(toggle))
    assert [svg.classes for svg in button.find("svg")] == [["icon", "search-toggle-chevron"]]


@pytest.mark.parametrize("url", ["/", *(url for url, _ in NO_RESULTS)])
def test_pages_without_results_keep_the_search_panel_open(client, url):
    _, page = _get(client, url)

    assert not _toggles(page)


def test_popular_cards_are_the_most_voted_movies(engine):
    cards = engine.popular_cards()

    assert len(cards) == 12
    by_votes = engine.movies_df.sort_values("vote_count", ascending=False, kind="stable")
    assert [card["id"] for card in cards] == by_votes["id"].head(12).tolist()
    # A card carries what the card template shows, plus the director for result notes.
    assert set(cards[0]) == {"id", "title", "year", "vote_average", "poster_url", "director"}
    # Each call returns fresh cards, so adding posters to one page never leaks into the next.
    assert engine.popular_cards() is not cards
    assert engine.popular_cards()[0] is not cards[0]


def test_empty_home_shows_popular_movies_and_genre_shortcuts(client, engine):
    body, page = _get(client)

    assert _card_ids(page) == [card["id"] for card in engine.popular_cards()]
    assert "Popüler filmler" in _headings(body)
    assert "Türlere göz atın" in _headings(body)
    tags = [link["href"] for link in page.all("a") if link.get("href", "").startswith("/?tur=")]
    assert tags == [f"/?tur={genre.replace(' ', '+')}" for genre in engine.all_genres]
    assert texts(page, "a", href="/?tur=Science+Fiction") == ["Bilim Kurgu"]
    assert not _more_link(page)


def test_genre_shortcuts_are_tags(client, engine):
    _, page = _get(client)

    [tag_list] = page.all("ul", **{"class": "tag-list"})
    tags = [link for link in page.all("a") if "tag" in _classes(link)]
    assert len(tags) == len(engine.all_genres)
    assert all(link["href"].startswith("/?tur=") for link in tags)
    # Tags are the only genre style: no element has a chip class.
    assert not [c for _, attrs in page.tags for c in _classes(attrs) if "chip" in c]


def test_genre_filters_are_toggle_tags(client, engine):
    _, page = _get(client)

    # Each checkbox's label is drawn as a tag; home.css fills it when checked.
    boxes = page.all("input", name="tur")
    labels = {label["for"]: label for label in page.all("label") if "for" in label}
    assert len(boxes) == len(engine.all_genres)
    assert all("tag" in _classes(labels[box["id"]]) for box in boxes)


@pytest.mark.parametrize("url", ["/", SEARCHES["results"], "/?tur=Action"])
def test_headings_carry_no_icons(client, url):
    _, page = _get(client, url)

    headings = page.find("h2")
    assert headings
    assert not [heading for heading in headings if heading.find("svg")]


def test_results_heading_keeps_the_count_apart(client):
    _, page = _get(client, SEARCHES["results"])

    [heading] = page.find("h2", "results-title")
    assert "page-title" in heading.classes
    # The count closes the heading, in its own element.
    assert texts(heading, class_="results-count") == ["5 film"]
    assert heading.text.endswith("5 film")


def test_search_panel_is_a_panel(client):
    _, page = _get(client)

    [panel] = page.all("section", aria_label="Film arama")
    assert {"search-panel", "panel"} <= set(_classes(panel))


def test_search_buttons_are_text_only_primary_buttons(client):
    _, page = _get(client)

    buttons = page.find("button", "btn-primary", type="submit")
    assert [button.text for button in buttons] == ["Benzerlerini bul", "Filtrele"]
    assert not [button for button in buttons if button.find("svg")]


def test_tagline_opens_the_landing_page_content(client):
    body, _ = _get(client)
    header, main = body.split("<main", 1)

    assert TAGLINE not in header
    assert TAGLINE in main
    # A search page leads with its results instead.
    assert TAGLINE not in body_text(client.get(SEARCHES["results"]))


def test_home_page_marks_its_body_for_the_header_search_rule(client):
    _, page = _get(client)
    [body] = page.all("body")

    # Phones get the home page's own search panel, so base.css hides the header search.
    assert "page-home" in body.get("class", "").split()
    [other] = Page(body_text(client.get("/my_lists"))).find("body")
    assert "page-home" not in other.classes


def test_similarity_search_is_a_shareable_url(client):
    body, page = _get(client, "/?film=Inception&n=5")

    # Typographic quotes around the film name.
    assert "“Inception” için benzer filmler 5 film" in _headings(body)
    assert len(_card_ids(page)) == 5
    [film] = page.all("input", name="film")
    assert film["value"] == "Inception"
    assert "Popüler filmler" not in body


@pytest.mark.parametrize("typed", ["Avatr", " avatar "])
def test_similarity_results_name_the_film_they_matched(client, typed):
    body, page = _get(client, f"/?film={typed}&n=5")

    assert any(heading.startswith("“Avatar” için benzer filmler") for heading in _headings(body))
    [film] = page.all("input", name="film")
    assert film["value"] == "Avatar"


def test_a_fuzzy_match_keeps_its_warning(client):
    body, page = _get(client, SEARCHES["warning"])

    assert "'Avatr' adında tam bir eşleşme bulunamadı." in _text(body)
    assert page.find(class_="alert-warning")


def test_filter_search_is_a_shareable_url_and_keeps_its_values(client):
    url = "/?tur=Action&tur=Science+Fiction&yonetmen=&min_yil=1990&max_yil=2015&min_puan=7&n=3"
    body, page = _get(client, url)

    assert "Türler: Aksiyon, Bilim Kurgu · Yıl: 1990 - 2015 · Min. puan: 7" in _text(body)
    assert any(heading.startswith("Filtre sonuçları") for heading in _headings(body))
    assert [box["value"] for box in page.all("input", name="tur") if "checked" in box] == [
        "Action",
        "Science Fiction",
    ]
    assert [page.all("input", name=name)[0]["value"] for name in ("min_yil", "max_yil")] == [
        "1990",
        "2015",
    ]
    assert [option["value"] for option in page.all("option") if "selected" in option] == ["7"]
    assert 1 <= len(_card_ids(page)) <= 3


def test_a_reversed_year_range_is_shown_in_order(client):
    body, page = _get(client, "/?min_yil=2015&max_yil=1990")

    assert "Yıl: 1990 - 2015" in _text(body)
    assert [page.all("input", name=name)[0]["value"] for name in ("min_yil", "max_yil")] == [
        "1990",
        "2015",
    ]
    assert "Yıl aralığı ters girilmişti, düzeltildi." in _text(body)


@pytest.mark.parametrize(
    ("query", "lang", "summary"),
    [
        ("min_yil=2010", "tr", "Yıl: 2010 ve sonrası"),
        ("max_yil=2010", "tr", "Yıl: 2010 ve öncesi"),
        ("min_yil=2010", "en", "Years: 2010 onward"),
        ("max_yil=2010", "en", "Years: up to 2010"),
    ],
)
def test_a_year_range_open_at_one_end_says_so(client, query, lang, summary):
    _, page = _get(client, f"/?{query}&lang={lang}")

    assert texts(page, "p", "results-summary") == [summary]


def test_an_english_page_shows_the_warning_and_the_filters_in_english(client):
    url = "/?tur=Action&tur=Science+Fiction&yonetmen=nolan&min_yil=2015&max_yil=1990&min_puan=7.5"
    _, page = _get(client, f"{url}&lang=en")

    summary = (
        "Genres: Action, Science Fiction · Director: nolan · Years: 1990 - 2015 · Min. rating: 7.5"
    )
    assert texts(page, "p", "results-summary") == [summary]
    assert texts(page, class_="search-warnings") == [
        "Director 'nolan' not found. Using 'Christopher Nolan'. "
        "The year range was reversed and has been corrected."
    ]
    # The fields show the range in order, whatever the language.
    assert [page.all("input", name=name)[0]["value"] for name in ("min_yil", "max_yil")] == [
        "1990",
        "2015",
    ]


def test_a_fuzzy_match_warns_in_english(client):
    _, page = _get(client, f"{SEARCHES['warning']}&lang=en")

    assert texts(page, class_="search-warnings") == [
        "No exact match for 'Avatr'. Showing results for the closest title, 'Avatar'."
    ]


@pytest.mark.parametrize(
    ("url", "message"),
    [
        (SEARCHES["empty"], "No film matches all of the criteria you chose."),
        ("/?film=xyzzyplugh", "No film with a title like 'xyzzyplugh' was found."),
        ("/?film=", "Please enter a film title."),
        (
            "/?yonetmen=&min_yil=&max_yil=&min_puan=",
            "Please enter at least one criterion (genre, director, etc.) to search.",
        ),
    ],
)
def test_a_search_without_results_explains_it_in_english(client, url, message):
    _, page = _get(client, f"{url}&lang=en")

    assert texts(page, "p", "empty-state-message") == [message]


def test_one_genre_is_named_in_the_singular(client):
    body, _ = _get(client, "/?tur=Science%20Fiction")
    english, _ = _get(client, "/?tur=Science%20Fiction&lang=en")

    assert "Tür: Bilim Kurgu" in _text(body)
    assert "Genre: Science Fiction" in _text(english)


def test_a_repeated_genre_counts_once(client):
    body, page = _get(client, "/?tur=Action&tur=Action")

    assert "Tür: Aksiyon" in _text(body)
    [link] = _more_link(page)
    assert link["href"] == "/?tur=Action&n=24#sonuc-13"


@pytest.mark.parametrize(
    ("url", "more"),
    [
        ("/?film=Avatar", "/?film=Avatar&n=24#sonuc-13"),
        ("/?tur=Action", "/?tur=Action&n=24#sonuc-13"),
    ],
)
def test_results_show_twelve_and_link_to_twelve_more(client, url, more):
    body, first = _get(client, url)

    assert len(_card_ids(first)) == 12
    [link] = _more_link(first)
    assert link["href"] == more
    assert {"btn", "btn-secondary"} <= set(_classes(link))
    assert texts(first, "a", "more-results") == ["Daha fazla göster"]

    _, second = _get(client, more.split("#")[0])
    # Rankings are prefix-consistent, so the first twelve stay where they were.
    assert _card_ids(second)[:12] == _card_ids(first)
    assert len(_card_ids(second)) == 24
    assert second.with_id("sonuc-13")
    assert not _more_link(second)


def test_more_link_is_left_out_when_every_result_is_shown(client):
    _, page = _get(client, "/?yonetmen=Christopher+Nolan")

    assert 1 <= len(_card_ids(page)) < 12
    assert not _more_link(page)


def test_more_link_is_left_out_when_exactly_twelve_results_match(client):
    # Joel Coen has exactly 12 films with 100 or more votes, in the Kaggle CSVs and the fixture.
    _, page = _get(client, "/?yonetmen=Joel+Coen")

    assert len(_card_ids(page)) == 12
    assert not _more_link(page)


@pytest.mark.parametrize(
    ("n", "count"), [("5", 5), ("abc", 12), ("", 12), ("0", 1), ("30", 30), ("99", 30)]
)
def test_result_count_comes_from_n_and_is_clamped(client, n, count):
    _, page = _get(client, f"/?tur=Drama&n={n}")

    assert len(_card_ids(page)) == count


def test_results_use_the_shared_movie_card_with_the_director(client, engine):
    _, page = _get(client, "/?film=Inception&n=3")

    assert len(page.all("ul", **{"class": "movie-grid"})) == 1
    notes = texts(page, class_="movie-card-note")
    directors = engine.movies_df.set_index("id").loc[_card_ids(page), "director"].tolist()
    assert notes == [f"Yönetmen: {director}" for director in directors]


@pytest.mark.parametrize(("url", "message"), NO_RESULTS)
def test_search_without_results_shows_one_message(client, url, message):
    body, page = _get(client, url)

    # Without results the reason is the one message: no separate warning box repeats it.
    assert _text(body).count(message) == 1
    [empty] = [div for div in page.all("div") if "empty-state" in _classes(div)]
    assert "panel" in _classes(empty)
    assert not page.find(class_="alert-warning")
    assert not _card_ids(page)


@pytest.mark.parametrize(
    ("url", "fits", "other"),
    [(SEARCHES["empty"], "filtre", "film adı"), ("/?film=xyzzyplugh", "film adı", "filtre")],
)
def test_empty_state_hint_fits_the_search(client, url, fits, other):
    body, _ = _get(client, url)

    [hint] = _hint(body)
    assert fits in hint.casefold()
    assert other not in hint.casefold()


@pytest.mark.parametrize("url", ["/?film=", "/?yonetmen=&min_yil=&max_yil=&min_puan="])
def test_an_empty_search_needs_no_hint(client, url):
    body, _ = _get(client, url)

    assert not _hint(body)


def test_search_buttons_show_and_announce_a_loading_text(client):
    _, page = _get(client)

    buttons = page.all("button", type="submit")
    loading = [
        button.get("data-loading-text")
        for button in buttons
        if "btn-primary" in button.get("class", "")
    ]
    assert loading == ["Aranıyor...", "Aranıyor..."]
    # base.js writes the loading text here; focus stays on the button.
    [status] = page.all("p", role="status")
    assert status["aria-live"] == "polite"


def test_search_buttons_have_a_spinner_icon_ready(client):
    body, _ = _get(client)

    # base.js puts this spinning icon before the loading text.
    [template] = Page(body).find("template", id="spinner-icon-template")
    [svg] = template.find("svg")
    assert svg.classes == ["icon", "icon-spin"]
    assert svg.attrs["aria-hidden"] == "true"
    [use] = svg.find("use")
    href = urlsplit(use.attrs["href"])
    assert (href.path, href.fragment) == ("/static/icons.svg", "spinner")
    assert re.fullmatch(r"v=[0-9a-f]{12}", href.query)


def test_a_genre_link_lists_only_that_genre(client, engine):
    ids = _card_ids(Page(body_text(client.get("/?tur=Science+Fiction"))))

    assert ids
    genres = engine.movies_df.set_index("id").loc[ids, "genre_list"]
    assert all("Science Fiction" in names for names in genres)


def test_header_search_suggests_titles_on_every_page(pages):
    for name, body in pages.items():
        page = Page(body)
        [field] = page.all("input", name="search_query")
        assert field["list"] == "movie-titles", name
        assert field["autocomplete"] == "off", name
        assert len(page.all("datalist", id="movie-titles")) == 1, name


@pytest.mark.parametrize(
    ("path", "attribute"),
    [("/api/titles", "title_suggestions"), ("/api/directors", "all_director_names")],
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


def test_title_suggestions_carry_the_turkish_titles_when_the_catalogue_has_them(turkish_client):
    titles = turkish_client.get("/api/titles").get_json()

    assert ["Inception", "Başlangıç"] in titles
    assert "Avatar" in titles


def test_a_similar_search_by_turkish_title_shows_the_original_title(turkish_client):
    _, page = _get(turkish_client, "/?film=baslangic&n=3")

    assert page.all("input", name="film")[0]["value"] == "Inception"
    assert 1 <= len(_card_ids(page)) <= 3
    assert 27205 not in _card_ids(page)


def test_suggestion_endpoint_without_engine_returns_503_json(client, app_module, monkeypatch):
    monkeypatch.setitem(app_module.app.extensions, "engine", None)

    response = client.get("/api/titles")

    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "error": "unavailable"}
