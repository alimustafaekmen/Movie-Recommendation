"""TF-IDF content similarity and the formulas that rank candidate movies."""

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from movierec.config import (
    FEATURE_POPULARITY_WEIGHT,
    FEATURE_RATING_WEIGHT,
    FEATURE_VOTE_COUNT_WEIGHT,
    POPULARITY_WEIGHT,
    SIMILARITY_WEIGHT,
    VOTE_AVERAGE_WEIGHT,
    VOTE_COUNT_WEIGHT,
)

TFIDF_MAX_FEATURES = 20000
RATING_MAX = 10.0


def content_text(movies: pd.DataFrame) -> pd.Series:
    """Return the text each movie is vectorised from: genres, keywords, director, overview, tagline."""
    return (
        movies["genres"]
        + " "
        + movies["keywords"]
        + " "
        + movies["director"]
        + " "
        + movies["overview"]
        + " "
        + movies["tagline"]
    )


def build_tfidf_matrix(movies: pd.DataFrame) -> sparse.csr_matrix:
    """Return one L2-normalised TF-IDF row per movie (about 2 MB for the full dataset)."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
    # fit_transform rows are unit length only up to rounding; normalising once more makes
    # plain dot products equal sklearn's cosine_similarity exactly.
    return normalize(vectorizer.fit_transform(content_text(movies)))


def hybrid_score(
    similarity: np.ndarray,
    vote_average: np.ndarray,
    vote_count: np.ndarray,
    popularity: np.ndarray,
) -> np.ndarray:
    """Rank score for similarity candidates, with every signal scaled to 0-1 within the pool.

        SIMILARITY_WEIGHT * similarity / max(similarity)
        + VOTE_AVERAGE_WEIGHT * vote_average / RATING_MAX
        + VOTE_COUNT_WEIGHT * vote_count / max(vote_count, 1)
        + POPULARITY_WEIGHT * popularity / max(popularity, 1)

    The weights come from movierec.config; the arrays describe one candidate pool and the
    maxima are taken over it.
    """
    # Raw cosine scores in a pool span roughly 0.1-0.45; unscaled, the similarity weight
    # would lose to popularity and votes, and blockbusters would outrank close matches.
    similarity = similarity / max(similarity.max(initial=0.0), 1e-9)
    max_votes = max(vote_count.max(initial=0), 1.0)
    max_popularity = max(popularity.max(initial=0.0), 1.0)
    return (
        similarity * SIMILARITY_WEIGHT
        + vote_average / RATING_MAX * VOTE_AVERAGE_WEIGHT
        + vote_count / max_votes * VOTE_COUNT_WEIGHT
        + popularity / max_popularity * POPULARITY_WEIGHT
    )


def feature_score(movies: pd.DataFrame) -> pd.Series:
    """Rank score for filter results, evaluated left to right in this order:

        vote_average / RATING_MAX * FEATURE_RATING_WEIGHT
        + popularity / max(popularity, 1) * FEATURE_POPULARITY_WEIGHT
        + vote_count / max(vote_count, 1) * FEATURE_VOTE_COUNT_WEIGHT

    Maxima are taken over ``movies``.
    """
    max_votes = max(movies["vote_count"].max(), 1.0)
    max_popularity = max(movies["popularity"].max(), 1.0)
    return (
        movies["vote_average"] / RATING_MAX * FEATURE_RATING_WEIGHT
        + movies["popularity"] / max_popularity * FEATURE_POPULARITY_WEIGHT
        + movies["vote_count"] / max_votes * FEATURE_VOTE_COUNT_WEIGHT
    )
