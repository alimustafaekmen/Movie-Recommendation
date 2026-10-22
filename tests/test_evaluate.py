import pandas as pd
import pytest
from conftest import FIXTURE_DATASET_DIR

from movierec.engine import RecommendationEngine
from scripts.evaluate import METHODS, evaluate, jaccard, metrics, title_stem


def test_jaccard():
    assert jaccard({"Action", "Drama"}, {"Drama", "Crime"}) == pytest.approx(1 / 3)
    assert jaccard({"Action"}, {"Action"}) == 1.0
    assert jaccard({"Action"}, set()) == 0.0


@pytest.mark.parametrize(
    ("title", "stem"),
    [
        ("Star Trek II: The Wrath of Khan", "star trek"),
        ("Toy Story 3", "toy story"),
        ("The Dark Knight Rises", "dark knight"),
        ("Rocky IV", "rocky"),
        ("Harry Potter and the Goblet of Fire", "harry potter"),
        ("Pirates of the Caribbean: On Stranger Tides", None),  # "of" is too generic
        ("2012", None),
    ],
)
def test_title_stem(title, stem):
    assert title_stem(title) == stem


def test_metrics_on_hand_made_lists():
    movies = pd.DataFrame(
        {
            "genre_list": [["Action", "Drama"], ["Drama"], []],
            "directors": [["Ann"], ["Ann", "Bob"], ["Cy"]],
            "vote_count": [10, 300, 500],
            "cast": ["Louis Gossett, Jr., Ann, Cy, Dee", "Dee, Eve", "Cy"],
            "title": ["Jaws", "Jaws 2", "Jaws: The Revenge"],
        }
    )
    # Film 2 has no genres, so only film 0's list counts towards genre overlap.
    result = metrics(movies, {0: [1, 2], 2: [0]})

    assert result["Genre overlap"] == pytest.approx((0.5 + 0.0) / 2)
    assert result["Same director"] == pytest.approx(1 / 3)
    assert result["Catalog coverage"] == 1.0
    assert result["Median votes"] == 300
    # Cy is billed in the top three of films 0 and 2 ("Louis Gossett, Jr." is one name); Dee
    # is only fourth for film 0.
    assert result["Shared lead actor"] == pytest.approx(2 / 3)
    # Film 0 finds both of its siblings. Film 2's only sibling is film 1, since film 0 has
    # under 100 votes, and its list misses it.
    assert result["Series recall"] == pytest.approx((1.0 + 0.0) / 2)


def test_evaluate_on_the_fixture_dataset():
    engine = RecommendationEngine(
        str(FIXTURE_DATASET_DIR / "tmdb_5000_movies.csv"),
        str(FIXTURE_DATASET_DIR / "tmdb_5000_credits.csv"),
    )

    results = evaluate(engine)

    assert list(results) == list(METHODS)
    for row in results.values():
        assert 0 <= row["Genre overlap"] <= 1
        assert 0 <= row["Same director"] <= 1
        assert 0 <= row["Shared lead actor"] <= 1
        assert 0 <= row["Series recall"] <= 1
        assert 0 < row["Catalog coverage"] <= 1
        assert row["Median votes"] >= 0
    # Every seed gets the same 12 most-voted films, give or take itself.
    assert results["Most-voted baseline"]["Catalog coverage"] == 13 / len(engine.movies_df)
