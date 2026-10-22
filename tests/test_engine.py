import pytest

from movierec.engine import DETAIL_SIMILAR_COUNT, RecommendationEngine, clamp_count
from movierec.i18n import message


def test_similarity_excludes_the_query_movie(engine):
    results, _ = engine.recommend_by_similarity("Avatar", num=5)

    assert 1 <= len(results) <= 5
    assert 19995 not in {movie["id"] for movie in results}


def test_misspelled_title_falls_back_to_closest_match(engine):
    results, warnings = engine.recommend_by_similarity("inceptoin")

    assert results
    # A catalogue key and its parameters; the page translates it.
    assert message("warning.not_exact", query="inceptoin", title="Inception") in warnings


def test_feature_search_by_genre_returns_movies(engine):
    results, _ = engine.recommend_by_features(genres=["Drama"])

    assert results


def _similar_ids(details, movie_id):
    return [movie["id"] for movie in details.get_movie_details(movie_id)["similar_movies"]]


def test_duplicate_titles_get_their_own_neighbours(details):
    batman_1966 = _similar_ids(details, 2661)

    assert batman_1966 != _similar_ids(details, 268)
    assert 2661 not in batman_1966


def test_similar_movie_ids_match_the_detail_page(engine, details):
    assert engine.similar_movie_ids(155) == _similar_ids(details, 155)
    # Twelve fills grid rows of 2, 3, 4 or 6 cards.
    assert DETAIL_SIMILAR_COUNT == 12
    assert len(engine.similar_movie_ids(155, num=30)) == 30


@pytest.mark.parametrize(
    ("query", "movie_id"),
    [("Alien³", 8077), ("alien", 348), ("batman", 268), ("  The Host ", 72710), ("avatr", 19995)],
)
def test_find_movie_id(engine, query, movie_id):
    assert engine.find_movie_id(query) == movie_id


@pytest.mark.real_dataset
def test_typo_does_not_match_a_one_letter_title(engine):
    # WRatio scores "Bogs" against the film "O" (id 11065) as a partial match.
    assert engine.find_movie_id("Bogs") == 3587


@pytest.mark.real_dataset
@pytest.mark.parametrize(
    ("query", "movie_id"),
    [("Jaws 1975", 578), ("Thor movie", 10195), ("Saw 2004", 176), ("Juno 2007", 7326)],
)
def test_short_title_with_extra_words(engine, query, movie_id):
    assert engine.find_movie_id(query) == movie_id


@pytest.mark.real_dataset
@pytest.mark.parametrize(
    ("query", "movie_id"),
    [("Godfather 2", 238), ("Godfather II", 240), ("Hangover 2", 45243)],
)
def test_sequel_query_does_not_match_a_short_unrelated_title(engine, query, movie_id):
    # WRatio gave "Her" (inside "godfather") and "2 Fast 2 Furious" (first of the tied "2"
    # titles) 90 and 86.
    assert engine.find_movie_id(query) == movie_id


@pytest.mark.parametrize("query", ["a", " ", "", None, 5, "qqqqzzzzxx"])
def test_find_movie_id_rejects_junk(engine, query):
    assert engine.find_movie_id(query) is None


def test_overlong_query_is_cut_to_100_characters(engine):
    _, warnings = engine.recommend_by_similarity("x" * 5000)

    assert warnings == [message("warning.movie_not_found", query="x" * 100)]


def test_overlong_director_query_is_cut_to_100_characters(engine):
    _, warnings = engine.recommend_by_features(director_query="q" * 5000)

    assert warnings == [message("warning.director_not_found", query="q" * 100)]


def test_exact_title_has_no_fuzzy_warning(engine):
    _, warnings = engine.recommend_by_similarity("alien³")

    assert warnings == []


@pytest.mark.parametrize(
    ("num", "count"), [(-5, 1), (0, 1), ("5", 5), ("abc", 12), ("", 12), (None, 12), (999, 30)]
)
def test_result_count_is_clamped(engine, num, count):
    similar, _ = engine.recommend_by_similarity("Avatar", num=num)
    by_genre, _ = engine.recommend_by_features(genres=["Drama"], num=num)

    assert clamp_count(num) == count
    assert 1 <= len(similar) <= count
    assert 1 <= len(by_genre) <= count


@pytest.mark.parametrize("query", [5, None, "", "   "])
def test_missing_title_asks_for_one(engine, query):
    results, warnings = engine.recommend_by_similarity(query)

    assert results == []
    assert warnings == [message("warning.empty_query")]


def test_a_filter_search_without_criteria_asks_for_one(engine):
    assert engine.recommend_by_features() == ([], [message("warning.no_criteria")])
    # The warning about a bad number still says what went wrong.
    assert engine.recommend_by_features(min_rating="abc") == (
        [],
        [message("warning.no_criteria"), message("warning.invalid_number")],
    )


def test_single_genre_string_is_treated_as_one_genre(engine):
    assert engine.recommend_by_features(genres="Drama") == engine.recommend_by_features(
        genres=["Drama"]
    )


@pytest.mark.parametrize("movie_id", ["abc", None, 1.5j])
def test_list_card_for_invalid_id_is_none(details, movie_id):
    assert details.get_list_card(movie_id) is None


def test_missing_dataset_raises(tmp_path):
    missing = str(tmp_path / "missing.csv")

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        RecommendationEngine(missing, missing)


def test_low_vote_feature_results_fall_back_with_a_warning(engine):
    results, warnings = engine.recommend_by_features(genres=["Foreign"])

    assert results
    assert message("warning.less_known") in warnings


def _feature_ids(results):
    return [movie["id"] for movie in results]


def _rows(engine, results):
    ids = _feature_ids(results)
    return engine.movies_df.set_index("id").loc[ids]


def test_multi_word_genres_stay_whole(engine):
    assert "Science Fiction" in engine.all_genres
    assert not {"Science", "Fiction", "TV", "Movie", "TV Movie", "Foreign"} & set(engine.all_genres)


def test_genre_filter_needs_every_genre_exactly(engine):
    results, _ = engine.recommend_by_features(genres=["science fiction", "Animation"], num=30)

    assert results
    for genres in _rows(engine, results)["genre_list"]:
        assert {"Science Fiction", "Animation"} <= set(genres)


def test_genre_fragment_matches_nothing(engine):
    results, warnings = engine.recommend_by_features(genres=["a"])

    assert results == []
    assert warnings


def test_genre_with_no_matches_plus_director_and_year_finds_nothing(engine):
    results, warnings = engine.recommend_by_features(
        genres=["a"], director_query="Christopher Nolan", min_year="2000"
    )

    assert results == []
    assert warnings == [message("warning.no_match")]


def test_co_directors_are_searchable(engine):
    results, warnings = engine.recommend_by_features(director_query="Ethan Coen", num=30)

    assert len(results) >= 5
    assert warnings == []
    assert all("Ethan Coen" in directors for directors in _rows(engine, results)["directors"])


@pytest.mark.parametrize(
    ("query", "director"),
    [
        ("nolan", "Christopher Nolan"),
        ("spielberg", "Steven Spielberg"),
        ("tarantino", "Quentin Tarantino"),
        ("cristopher nolan", "Christopher Nolan"),
    ],
)
def test_fuzzy_director_resolves(engine, query, director):
    results, warnings = engine.recommend_by_features(director_query=query)

    assert results
    assert warnings == [message("warning.director_fuzzy", query=query, name=director)]


@pytest.mark.parametrize("query", ["Zzqx Qwerty", "John Smith"])
def test_unknown_director_returns_nothing(engine, query):
    results, warnings = engine.recommend_by_features(
        genres=["Drama"], director_query=query, min_rating="7"
    )

    assert results == []
    assert warnings == [message("warning.director_not_found", query=query)]


def test_turkish_decimal_comma_is_accepted(engine):
    results, warnings = engine.recommend_by_features(genres=["Drama"], min_rating="7,5", num=30)

    assert results
    assert warnings == []
    assert (_rows(engine, results)["vote_average"] >= 7.5).all()


def test_one_invalid_number_keeps_the_other_filters(engine):
    results, warnings = engine.recommend_by_features(
        min_rating="abc", min_year="2010.0", max_year="nan", num=30
    )

    assert warnings == [message("warning.invalid_number")]
    assert (_rows(engine, results)["year"] >= 2010).all()


@pytest.mark.parametrize(("field", "value"), [("min_rating", "11"), ("min_year", "1800")])
def test_out_of_range_number_is_rejected(engine, field, value):
    _, warnings = engine.recommend_by_features(genres=["Drama"], **{field: value})

    assert warnings == [message("warning.invalid_number")]


def test_reversed_year_range_is_swapped(engine):
    results, warnings = engine.recommend_by_features(min_year="2020", max_year="2000", num=30)

    assert warnings == [message("warning.years_swapped")]
    assert _rows(engine, results)["year"].between(2000, 2020).all()


def test_detail_badges_use_turkish_labels(details):
    movie = details.get_movie_details(19995)

    assert {"display_name": "Bilim Kurgu", "link_name": "Science Fiction"}.items() <= next(
        badge for badge in movie["genre_badges"] if badge["link_name"] == "Science Fiction"
    ).items()
    assert [badge["display_name"] for badge in movie["genre_badges"]] == [
        "Aksiyon",
        "Macera",
        "Fantastik",
        "Bilim Kurgu",
    ]


def test_director_names_are_stripped(engine):
    names = engine.movies_df["director"]

    assert (names == names.str.strip()).all()


def test_details_list_every_director(details):
    movie = details.get_movie_details(6977)  # No Country for Old Men

    assert movie["directors"] == ["Joel Coen", "Ethan Coen"]
    assert movie["director"] == "Joel Coen"


def test_close_franchise_match_outranks_a_popular_weak_match(engine):
    results, _ = engine.recommend_by_similarity("The Dark Knight", num=10)
    titles = [movie["title"] for movie in results]

    assert "Despicable Me 2" not in titles or titles.index("Batman Returns") < titles.index(
        "Despicable Me 2"
    )


@pytest.mark.parametrize("movie_id", [155, 19995, 862, 2661, 597, 238, 13, 680, 24428, 72710])
def test_every_count_is_a_prefix_of_the_full_list(engine, movie_id):
    # Asking for more results must never reorder the top ones, so one 30-item list per
    # movie serves every count.
    full = engine.similar_movie_ids(movie_id, 30)

    assert len(full) == 30
    for num in range(1, 30):
        assert engine.similar_movie_ids(movie_id, num) == full[:num]


@pytest.mark.real_dataset
def test_toy_story_neighbours_are_not_matched_on_a_character_name(engine):
    # The 40 Year Old Virgin used to rank fifth only because both overviews mention "Andy".
    assert 6957 not in engine.similar_movie_ids(862)


@pytest.mark.real_dataset
def test_dark_knight_keeps_the_earlier_batman_films_near_the_top(engine):
    # Too much director weight let Inception and Scarface push these down to #10-11.
    top = engine.similar_movie_ids(155, 6)

    assert {364, 414} <= set(top)  # Batman Returns, Batman Forever


def _fake_similar(engine, monkeypatch, rankings):
    """Give each favourite the ranked similar list in `rankings` instead of the real one."""
    monkeypatch.setattr(engine, "similar_movie_ids", lambda movie_id, num: rankings[movie_id])


def test_one_favourite_recommends_its_own_similar_list(engine):
    assert engine.recommend_from_favourites([155], []) == engine.similar_movie_ids(155)


def test_a_film_shared_by_three_favourites_ranks_first(engine, monkeypatch):
    a, b, c = engine.movies_df["id"].head(3).tolist()
    # 3 is second in every list: 3 * 1/2 beats the 1/1 of the tops, which keep the order
    # they first appear in, not the order of their ids.
    _fake_similar(engine, monkeypatch, {a: [6, 3], b: [2, 3], c: [9, 3]})

    assert engine.recommend_from_favourites([a, b, c], []) == [3, 6, 2, 9]


def test_listed_films_are_not_recommended(engine, monkeypatch):
    first, second = engine.movies_df["id"].head(2).tolist()
    _fake_similar(engine, monkeypatch, {first: [second, 1, 2], second: [first, 3]})

    assert engine.recommend_from_favourites([first, second], [2]) == [1, 3]


def test_unknown_favourites_are_ignored(engine):
    assert engine.recommend_from_favourites([424242, 155], []) == engine.similar_movie_ids(155)
    assert engine.recommend_from_favourites([424242], []) == []


def test_only_the_first_20_dataset_favourites_count(engine, monkeypatch):
    favourites = engine.movies_df["id"].head(21).tolist()
    rankings = {movie_id: [-movie_id] for movie_id in favourites}
    # The 21st favourite adds no list of its own but is still left out of the others.
    rankings[favourites[19]].append(favourites[20])
    _fake_similar(engine, monkeypatch, rankings)

    watchlist = [-movie_id for movie_id in favourites[:19]]

    # The unknown id takes no place among the 20.
    assert engine.recommend_from_favourites([424242, *favourites], watchlist) == [-favourites[19]]


def test_no_favourites_recommend_nothing(engine):
    assert engine.recommend_from_favourites([], [155]) == []
