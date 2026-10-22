import pandas as pd
import pytest
from conftest import FIXTURE_DATASET_DIR

from movierec.engine import RecommendationEngine
from scripts.evaluate import METHODS, evaluate, jaccard, metrics


def test_jaccard():
    assert jaccard({"Action", "Drama"}, {"Drama", "Crime"}) == pytest.approx(1 / 3)
    assert jaccard({"Action"}, {"Action"}) == 1.0
    assert jaccard({"Action"}, set()) == 0.0


def test_metrics_on_hand_made_lists():
    movies = pd.DataFrame(
        {
            "genre_list": [["Action", "Drama"], ["Drama"], []],
            "directors": [["Ann"], ["Ann", "Bob"], ["Cy"]],
            "vote_count": [10, 30, 50],
        }
    )
    # Film 2 has no genres, so only film 0's list counts towards genre overlap.
    result = metrics(movies, {0: [1, 2], 2: [1]})

    assert result["Genre overlap"] == pytest.approx((0.5 + 0.0) / 2)
    assert result["Same director"] == pytest.approx(1 / 3)
    assert result["Catalog coverage"] == pytest.approx(2 / 3)
    assert result["Median votes"] == 30


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
        assert 0 < row["Catalog coverage"] <= 1
        assert row["Median votes"] >= 0
    # Every seed gets the same 12 most-voted films, give or take itself.
    assert results["Most-voted baseline"]["Catalog coverage"] == 13 / len(engine.movies_df)
