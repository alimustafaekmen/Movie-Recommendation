"""Content-based movie recommendations over the TMDB 5000 dataset, computed offline."""

import logging
import time

import numpy as np
import pandas as pd
from thefuzz import fuzz, process, utils

from movierec.config import (
    DEFAULT_RECOMMENDATION_COUNT,
    FUZZY_SCORE_CUTOFF,
    MAX_RECOMMENDATION_COUNT,
    MIN_VOTES_THRESHOLD,
)
from movierec.dataset import load_movies
from movierec.genres import is_excluded
from movierec.search_filters import (
    apply_filters,
    genre_keys,
    is_plausible_name,
    parse_numeric_filters,
)
from movierec.similarity import (
    build_tfidf_matrix,
    feature_score,
    hybrid_score,
)

logger = logging.getLogger(__name__)

POPULAR_MOVIE_COUNT = 25
# Card grids hold 12 cards (home page before any search, similar movies on a detail page),
# which fills rows of 2, 3, 4 or 6.
POPULAR_CARD_COUNT = 12
DETAIL_SIMILAR_COUNT = 12
MAX_QUERY_LENGTH = 100
MIN_QUERY_LENGTH = 2
# Recommendations from a favourites list use this many of its films, newest first, and
# this many similar films of each.
MAX_FAVOURITES_USED = 20
FAVOURITE_NEIGHBOURS = MAX_RECOMMENDATION_COUNT
# Re-ranking looks a little past the longest list we serve. The pool does not depend on the
# requested count, so every list is a prefix of the MAX_RECOMMENDATION_COUNT list.
NEIGHBOUR_POOL_SIZE = 3 * MAX_RECOMMENDATION_COUNT + 1


class RecommendationEngine:
    """Loads the dataset once and answers similarity and filter queries."""

    def __init__(self, movies_path: str, credits_path: str):
        started = time.perf_counter()
        df = load_movies(movies_path, credits_path)
        if df.empty:
            raise ValueError(f"No movie in {movies_path} has credits in {credits_path}")
        self.movies_df = df
        self.tfidf_matrix = build_tfidf_matrix(df)
        self._title_keys = df["title"].str.casefold()
        self._ids = df["id"].to_numpy()
        self.row_of = {movie_id: row for row, movie_id in enumerate(self._ids.tolist())}
        self._vote_average = df["vote_average"].to_numpy(dtype=float)
        self._vote_count = df["vote_count"].to_numpy()
        self._popularity = df["popularity"].to_numpy(dtype=float)

        genres = df["genre_list"].explode().dropna().unique()
        self.all_genres = sorted(genre for genre in genres if not is_excluded(genre))
        self.all_movie_titles = sorted(df["title"].dropna().unique().tolist())
        self.all_director_names = sorted(df["directors"].explode().dropna().unique().tolist())
        well_voted = df[df["vote_count"] >= MIN_VOTES_THRESHOLD]
        by_votes = well_voted.sort_values("vote_count", ascending=False, kind="stable")
        self._most_voted_movies = by_votes.head(POPULAR_MOVIE_COUNT)
        self.popular_titles = self._most_voted_movies["title"].tolist()
        logger.info("Loaded %d movies in %.2fs", len(df), time.perf_counter() - started)

    def format_results(self, ranked: pd.DataFrame, num: int) -> list[dict]:
        """Turn ranked rows into the card dictionaries the templates render."""
        return [
            {
                "id": int(movie["id"]),
                # Filled in by MovieDetails.add_posters; the engine itself never calls TMDb.
                "poster_url": None,
                "year": int(movie["year"]),
                "vote_average": round(float(movie["vote_average"]), 1),
                "title": str(movie["title"]),
                "director": str(movie["director"]),
            }
            for movie in ranked.head(num).to_dict("records")
        ]

    def popular_cards(self, num: int = POPULAR_CARD_COUNT) -> list[dict]:
        """Cards for the most-voted movies; new dicts on every call, as add_posters edits them."""
        return self.format_results(self._most_voted_movies, num)

    def _find_best_match(self, query, choices):
        """Return the choice closest to ``query`` (thefuzz), or None below the score cutoff."""
        # A query of only punctuation or symbols scores 0 everywhere, and thefuzz would log
        # a warning for each one a visitor sends.
        if not choices or not utils.full_process(query or ""):
            return None
        # WRatio's partial scoring lets a short title win inside a longer word ("O" for "Bogs",
        # "Her" for "Godfather 2"), so a title under half the query's length only counts when
        # all its words are in the query ("Saw" for "Saw 2004").
        words = set(utils.full_process(query).split())
        choices = [
            choice
            for choice in choices
            if len(choice) * 2 >= len(query) or set(utils.full_process(choice).split()) <= words
        ]
        matches = process.extractBests(
            query, choices, scorer=fuzz.WRatio, score_cutoff=FUZZY_SCORE_CUTOFF, limit=None
        )
        if not matches:
            return None
        # WRatio often ties (every "... 2" title scores the same); the closest whole title wins.
        top = [choice for choice, score in matches if score == matches[0][1]]
        return max(top, key=lambda choice: fuzz.token_sort_ratio(query, choice))

    def find_movie_id(self, query: str) -> int | None:
        """Return the TMDb id of the movie a title query refers to, or None."""
        match = self._resolve_title(query)
        return None if match is None else int(self._ids[match[0]])

    def title_of(self, movie_id: int) -> str:
        """Return the dataset title of a movie id, e.g. one that find_movie_id returned."""
        return str(self.movies_df["title"].iat[self.row_of[movie_id]])

    def _resolve_title(self, query) -> tuple[int, bool] | None:
        """Return (row, is_exact) for a title query; exact case-insensitive matches win."""
        if not isinstance(query, str):
            return None
        query = query.strip()[:MAX_QUERY_LENGTH]
        if len(query) < MIN_QUERY_LENGTH:
            return None
        exact = (self._title_keys == query.casefold()).to_numpy()
        if exact.any():
            return self._most_voted(exact), True
        title = self._find_best_match(query, self.all_movie_titles)
        if title is None:
            return None
        return self._most_voted((self.movies_df["title"] == title).to_numpy()), False

    def _most_voted(self, mask: np.ndarray) -> int:
        # Remakes share a title; the better-known film is the likelier target.
        rows = np.flatnonzero(mask)
        return int(rows[np.argmax(self._vote_count[rows])])

    def recommend_by_similarity(
        self, movie_title_query: str, num: int | str = DEFAULT_RECOMMENDATION_COUNT
    ) -> tuple[list[dict], list[str]]:
        """Recommend movies similar to the given title; returns (results, warnings)."""
        if not isinstance(movie_title_query, str) or not movie_title_query.strip():
            return [], ["Lütfen bir film adı girin."]
        num = clamp_count(num)
        query = movie_title_query.strip()[:MAX_QUERY_LENGTH]
        match = self._resolve_title(query)
        if match is None:
            return [], [f"'{query}' ismine benzer bir film bulunamadı."]

        index, is_exact = match
        title = self.movies_df["title"].iat[index]
        warnings = []
        if not is_exact:
            warnings.append(
                f"'{query}' adında tam bir eşleşme bulunamadı. Bunun yerine en yakın isim olan "
                f"'{title}' için sonuçlar gösteriliyor."
            )
        ranked = self.rank_neighbours(index)
        if self._vote_count[ranked].max(initial=0) < MIN_VOTES_THRESHOLD:
            warnings.append(
                f"'{title}' için çok fazla oy almış popüler benzer film bulamadık. "
                "Daha az bilinen filmleri listeliyoruz."
            )
        return self.format_results(self.movies_df.iloc[ranked], num), warnings

    def similar_movie_ids(self, movie_id: int, num: int = DETAIL_SIMILAR_COUNT) -> list[int]:
        """Return the ids recommended next to a dataset movie, best first, without TMDb calls."""
        index = self.row_of.get(int(movie_id))
        if index is None:
            return []
        return self._ids[self.rank_neighbours(index)[: clamp_count(num)]].tolist()

    def recommend_from_favourites(
        self, favourite_ids: list[int], exclude_ids: list[int]
    ) -> list[int]:
        """Ids of films like the favourites, best first, leaving out favourites and exclude_ids.

        Each favourite's similar list gives the film at 0-based rank r a score of 1 / (r + 1);
        the scores add up over the favourites, and ties keep the order films first appear in.
        """
        used = [movie_id for movie_id in favourite_ids if movie_id in self.row_of]
        listed = {*favourite_ids, *exclude_ids}
        scores = {}
        for favourite in used[:MAX_FAVOURITES_USED]:
            similar = self.similar_movie_ids(favourite, FAVOURITE_NEIGHBOURS)
            for rank, movie_id in enumerate(similar):
                if movie_id not in listed:
                    scores[movie_id] = scores.get(movie_id, 0) + 1 / (rank + 1)
        # sorted() is stable with reverse=True too.
        return sorted(scores, key=scores.get, reverse=True)[:DETAIL_SIMILAR_COUNT]

    def rank_neighbours(self, index: int) -> np.ndarray:
        """Row positions of the movies most similar to row ``index``, best hybrid score first."""
        scores = (self.tfidf_matrix[index] @ self.tfidf_matrix.T).toarray().ravel()
        # A stable sort keeps tied scores in dataset order.
        order = np.argsort(-scores, kind="stable")
        pool = order[order != index][:NEIGHBOUR_POOL_SIZE]
        hybrid = hybrid_score(
            scores[pool], self._vote_average[pool], self._vote_count[pool], self._popularity[pool]
        )
        # Below MIN_VOTES_THRESHOLD votes TMDb ratings are noisy, so such films are only
        # shown when the pool has nothing better.
        well_voted = self._vote_count[pool] >= MIN_VOTES_THRESHOLD
        if well_voted.any():
            pool, hybrid = pool[well_voted], hybrid[well_voted]
        return pool[np.argsort(-hybrid, kind="stable")]

    def recommend_by_features(
        self,
        genres: list[str] | str | None = None,
        director_query: str | None = None,
        min_rating: str | float | None = None,
        min_year: str | int | None = None,
        max_year: str | int | None = None,
        num: int | str = DEFAULT_RECOMMENDATION_COUNT,
    ) -> tuple[list[dict], list[str]]:
        """Recommend movies matching genre, director, rating and year filters."""
        num = clamp_count(num)
        wanted_genres = genre_keys(genres)
        warnings = []

        director = ""
        if isinstance(director_query, str):
            # Cut like title queries, so the warnings never echo a pasted wall of text.
            director = director_query.strip()[:MAX_QUERY_LENGTH]
        director_match = None
        if director:
            director_match, director_warning = self._match_director(director)
            if director_match is None:
                return [], [director_warning]
            if director_warning:
                warnings.append(director_warning)

        numbers = parse_numeric_filters(min_rating, min_year, max_year)
        warnings.extend(numbers.warnings)
        if not (wanted_genres or director_match or numbers.count):
            return [], [
                "Lütfen arama yapmak için en az bir kriter (tür, yönetmen vb.) girin."
            ] + warnings

        matches = apply_filters(self.movies_df, wanted_genres, director_match, numbers)
        if matches.empty:
            return [], ["Seçtiğiniz kriterlere tam olarak uyan hiçbir film bulunamadı."] + warnings

        reliable = matches[matches["vote_count"] >= MIN_VOTES_THRESHOLD]
        if reliable.empty:
            reliable = matches
            warnings.append(
                "Yeterli oy sayısına sahip film bulunamadı; daha az bilinen filmler listeleniyor."
            )
        ranked = reliable.assign(feature_score=feature_score(reliable)).sort_values(
            "feature_score", ascending=False, kind="stable"
        )
        return self.format_results(ranked, num), warnings

    def _match_director(self, query: str) -> tuple[str | None, str | None]:
        """Return (director name or None, warning or None) for a free-text director query."""
        key = query.casefold()
        exact = next((name for name in self.all_director_names if name.casefold() == key), None)
        if exact:
            return exact, None
        match = self._find_best_match(query, self.all_director_names)
        if match and is_plausible_name(query, match):
            return match, f"Yönetmen '{query}' bulunamadı. '{match}' kullanılıyor."
        return None, f"'{query}' adlı yönetmen bulunamadı."


def clamp_count(num) -> int:
    """A requested result count (e.g. the ?n= text) clamped to [1, MAX]; junk gives the default."""
    try:
        num = int(num)
    except (TypeError, ValueError):
        return DEFAULT_RECOMMENDATION_COUNT
    return max(1, min(num, MAX_RECOMMENDATION_COUNT))
