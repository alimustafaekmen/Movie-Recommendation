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
    TFIDF_DIRECTOR_WEIGHT,
    TFIDF_GENRE_WEIGHT,
    TFIDF_KEYWORD_WEIGHT,
    TFIDF_TEXT_WEIGHT,
    VOTE_AVERAGE_WEIGHT,
    VOTE_COUNT_WEIGHT,
)

TFIDF_MAX_FEATURES = 20000
RATING_MAX = 10.0


def build_tfidf_matrix(df: pd.DataFrame) -> sparse.csr_matrix:
    """Return one L2-normalised, field-weighted TF-IDF row per movie (about 2.5 MB in all).

    Genres, directors, keywords and overview + tagline each get their own vectorizer, so a
    long overview cannot drown out the metadata and its character names carry less weight.

    df: the movies table built by dataset.load_movies, one row per film.
    """
    overview_and_tagline = df["overview"] + " " + df["tagline"]
    # (vectorizer, column, weight) for each block of features.
    blocks = (
        # genre_list and directors hold a list of names per film. analyzer=list hands that list
        # over unchanged as the film's tokens, so "Science Fiction" and "Christopher Nolan" are
        # one feature each and co-directors stay separate features.
        (TfidfVectorizer(analyzer=list), df["genre_list"], TFIDF_GENRE_WEIGHT),
        (TfidfVectorizer(analyzer=list), df["directors"], TFIDF_DIRECTOR_WEIGHT),
        # Words, not whole phrases: over half of the keyword phrases belong to a single
        # film, while their words ("toy", "superhero") are shared.
        (TfidfVectorizer(stop_words="english"), df["keywords"], TFIDF_KEYWORD_WEIGHT),
        (
            TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES),
            overview_and_tagline,
            TFIDF_TEXT_WEIGHT,
        ),
    )
    weighted_blocks = []
    for vectorizer, column, weight in blocks:
        # Scale each film's row of this block to length 1, then weight the block.
        block = normalize(vectorizer.fit_transform(column))
        weighted_blocks.append(block * weight)
    # Side by side: one row per film, every block's features as columns.
    matrix = sparse.hstack(weighted_blocks, format="csr")
    # Normalising the stacked rows makes plain dot products equal sklearn's cosine_similarity.
    return normalize(matrix)


def hybrid_score(
    similarity: np.ndarray,
    vote_average: np.ndarray,
    vote_count: np.ndarray,
    popularity: np.ndarray,
) -> np.ndarray:
    """Return the rank score of each similarity candidate, every signal scaled to 0-1 in the pool.

        SIMILARITY_WEIGHT * similarity / max(similarity)
        + VOTE_AVERAGE_WEIGHT * vote_average / RATING_MAX
        + VOTE_COUNT_WEIGHT * log1p(vote_count) / max(log1p(vote_count), 1)
        + POPULARITY_WEIGHT * log1p(popularity) / max(log1p(popularity), 1)

    The weights come from movierec.config; the arrays describe one candidate pool and the
    maxima are taken over it.
    """
    # Raw cosines in a pool span roughly 0.13-0.24 for a typical film; unscaled, the similarity
    # weight would lose to popularity and votes, and blockbusters would outrank close matches.
    largest_similarity = max(similarity.max(initial=0.0), 1e-9)  # 1e-9: a pool of zeros
    similarity_term = similarity / largest_similarity * SIMILARITY_WEIGHT
    rating_term = vote_average / RATING_MAX * VOTE_AVERAGE_WEIGHT
    # Vote counts and popularity are heavy-tailed; scaled linearly, one blockbuster in the
    # pool would push every other candidate's term to about 0.
    log_votes = np.log1p(vote_count)
    largest_log_votes = max(log_votes.max(initial=0.0), 1.0)
    votes_term = log_votes / largest_log_votes * VOTE_COUNT_WEIGHT
    log_popularity = np.log1p(popularity)
    largest_log_popularity = max(log_popularity.max(initial=0.0), 1.0)
    popularity_term = log_popularity / largest_log_popularity * POPULARITY_WEIGHT
    return similarity_term + rating_term + votes_term + popularity_term


def feature_score(df: pd.DataFrame) -> pd.Series:
    """Return the rank score of each filter result, evaluated left to right in this order:

        vote_average / RATING_MAX * FEATURE_RATING_WEIGHT
        + popularity / max(popularity, 1) * FEATURE_POPULARITY_WEIGHT
        + vote_count / max(vote_count, 1) * FEATURE_VOTE_COUNT_WEIGHT

    df: the filter results, rows of the movies table; maxima are taken over df.
    """
    max_votes = max(df["vote_count"].max(), 1.0)
    max_popularity = max(df["popularity"].max(), 1.0)
    rating_term = df["vote_average"] / RATING_MAX * FEATURE_RATING_WEIGHT
    popularity_term = df["popularity"] / max_popularity * FEATURE_POPULARITY_WEIGHT
    votes_term = df["vote_count"] / max_votes * FEATURE_VOTE_COUNT_WEIGHT
    return rating_term + popularity_term + votes_term
