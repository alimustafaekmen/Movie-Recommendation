"""Content-based movie recommendations from the dataset alone; the engine never calls TMDb."""

import logging
import re
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
from movierec.i18n import message
from movierec.search_filters import (
    apply_filters,
    genre_keys,
    is_plausible_name,
    parse_numeric_filters,
    title_key,
)
from movierec.similarity import (
    build_tfidf_matrix,
    feature_score,
    hybrid_score,
)

logger = logging.getLogger(__name__)

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
# Re-ranking reads about three times the longest list we serve. The pool does not depend on
# the requested count, so every list is a prefix of the MAX_RECOMMENDATION_COUNT list.
NEIGHBOUR_POOL_SIZE = 3 * MAX_RECOMMENDATION_COUNT + 1
# A run of digits or of letters is one word of a title key: "Alien³" (key "alien3") has
# "alien" and "3".
_WORD = re.compile(r"\d+|[^\W\d_]+")


class RecommendationEngine:
    """Loads the dataset once and answers similarity and filter queries.

    movies_df: the movies table from dataset.load_movies; row r of it is row r of tfidf_matrix.
    """

    def __init__(self, movies_path: str, credits_path: str):
        start_time = time.perf_counter()
        df = load_movies(movies_path, credits_path)
        if df.empty:
            raise ValueError(f"No movie in {movies_path} has credits in {credits_path}")
        self.movies_df = df
        self.tfidf_matrix = build_tfidf_matrix(df)
        self._title_keys = df["title"].map(title_key)
        # Turkish titles compare without case, accents or apostrophes, so "baslangic" finds
        # "Başlangıç". A film without one has "".
        self._turkish_keys = df["title_tr"].map(title_key)
        self._all_turkish_keys = sorted(set(self._turkish_keys) - {""})
        # Plain numpy copies of the columns each query reads. Position r in each one is row r of
        # movies_df and of tfidf_matrix; indexing an array is cheaper than a DataFrame column.
        self._ids = df["id"].to_numpy()
        self._vote_average = df["vote_average"].to_numpy(dtype=float)
        self._vote_count = df["vote_count"].to_numpy()
        self._popularity = df["popularity"].to_numpy(dtype=float)
        # TMDb id -> row position, to turn the ids in URLs and lists into rows.
        self.row_of = {movie_id: row for row, movie_id in enumerate(self._ids.tolist())}

        # genre_list and directors hold a list per film; explode() puts each item on its own row.
        genres = df["genre_list"].explode().dropna().unique()
        self.all_genres = sorted(genre for genre in genres if not is_excluded(genre))
        self.all_movie_titles = sorted(df["title"].dropna().unique().tolist())
        self.title_suggestions = self._title_suggestions(df)
        self.all_director_names = sorted(df["directors"].explode().dropna().unique().tolist())
        self._directors_by_key: dict[str, str] = {}
        for name in self.all_director_names:
            self._directors_by_key.setdefault(title_key(name), name)
        well_voted = df[df["vote_count"] >= MIN_VOTES_THRESHOLD]
        by_votes = well_voted.sort_values("vote_count", ascending=False, kind="stable")
        self._most_voted_movies = by_votes.head(POPULAR_CARD_COUNT)
        logger.info("Loaded %d movies in %.2fs", len(df), time.perf_counter() - start_time)

    def _title_suggestions(self, df: pd.DataFrame) -> list[str | list[str]]:
        """Return all_movie_titles, each as [title, Turkish title] when its film has one.

        df: the movies table. Remakes share a title; the most-voted one gives the Turkish title.
        """
        by_votes = df.sort_values("vote_count", ascending=False, kind="stable")
        most_voted = by_votes.drop_duplicates("title")
        turkish_of = dict(zip(most_voted["title"], most_voted["title_tr"], strict=True))
        suggestions = []
        for title in self.all_movie_titles:
            turkish = turkish_of[title]
            suggestions.append([title, turkish] if turkish else title)
        return suggestions

    def format_results(self, ranked: pd.DataFrame, num: int) -> list[dict]:
        """Return the card dictionaries for the first ``num`` ranked rows.

        ranked: rows of movies_df, best first.
        """
        cards = []
        for movie in ranked.head(num).to_dict("records"):
            card = {
                "id": int(movie["id"]),
                # Filled in by MovieDetails.add_posters; the engine itself never calls TMDb.
                "poster_url": None,
                "year": int(movie["year"]),
                "vote_average": round(float(movie["vote_average"]), 1),
                "title": str(movie["title"]),
                "director": str(movie["director"]),
            }
            cards.append(card)
        return cards

    def popular_cards(self, num: int = POPULAR_CARD_COUNT) -> list[dict]:
        """Return cards for the most-voted movies (at most POPULAR_CARD_COUNT).

        The dicts are new on every call, as add_posters edits them.
        """
        return self.format_results(self._most_voted_movies, num)

    def _find_best_match(self, query: str, choices: list[str]) -> str | None:
        """Return the choice closest to ``query`` (thefuzz WRatio), or None below the cutoff."""
        matches = self._fuzzy_matches(query, choices)
        if not matches:
            return None
        choice, _scores = self._closest(query, matches)
        return choice

    def _closest(self, query: str, matches: list[tuple[str, int]]) -> tuple[str, tuple[int, int]]:
        """Return (choice, (WRatio, token_sort_ratio)) for the best of ``matches``.

        matches: (choice, WRatio) pairs from _fuzzy_matches, best score first.
        """
        # WRatio often ties (every "... 2" title scores the same); the closest whole title wins.
        best_score = matches[0][1]
        tied = [choice for choice, score in matches if score == best_score]
        # On equal ratios max() returns the first one in tied.
        closest = max(tied, key=lambda choice: fuzz.token_sort_ratio(query, choice))
        return closest, (best_score, fuzz.token_sort_ratio(query, closest))

    def _closest_titles(
        self, query: str, titles: list[str]
    ) -> tuple[list[str], tuple[int, int, int]] | None:
        """Return (titles, rank) for the fuzzy matches that hold the most query words, or None.

        When they hold every query word, the query names part of each title ("john wick 2"
        is in John Wick: Chapter 2, not John Wick), so all of them are returned (only those
        that start with the query, if any) and the caller takes the most-voted film.
        Otherwise only the closest one is. rank is (words held, WRatio, token_sort_ratio), so
        the ranks of two lists compare as tuples.
        """
        matches = self._fuzzy_matches(query, titles)
        if not matches:
            return None
        query_words = _words(query)
        held = [len(query_words & _words(title)) for title, _score in matches]
        most_held = max(held)
        kept = [match for match, count in zip(matches, held, strict=True) if count == most_held]
        closest, scores = self._closest(query, kept)
        if most_held < len(query_words):
            return [closest], (most_held, *scores)
        # "the jungle" is The Jungle Book rather than Jumanji: Welcome to the Jungle.
        kept_titles = [title for title, _score in kept]
        query_key = title_key(query)
        starting = [title for title in kept_titles if title_key(title).startswith(query_key)]
        return starting or kept_titles, (most_held, *scores)

    def _fuzzy_matches(self, query: str, choices: list[str]) -> list[tuple[str, int]]:
        """Return (choice, WRatio) for the choices that reach the cutoff, best score first."""
        # full_process lowercases and drops punctuation. A query of only symbols becomes "",
        # scores 0 everywhere, and thefuzz would log a warning for each one a visitor sends.
        cleaned_query = utils.full_process(query)
        if not cleaned_query:
            return []
        query_words = set(cleaned_query.split())

        # WRatio's partial scoring lets a short title win inside a longer word ("O" for "Bogs",
        # "Her" for "Godfather 2"), so a title under half the query's length only counts when
        # all its words are in the query ("Saw" for "Saw 2004").
        candidates = []
        for choice in choices:
            if len(choice) * 2 >= len(query):
                candidates.append(choice)
                continue
            choice_words = set(utils.full_process(choice).split())
            if choice_words.issubset(query_words):
                candidates.append(choice)

        return process.extractBests(
            query, candidates, scorer=fuzz.WRatio, score_cutoff=FUZZY_SCORE_CUTOFF, limit=None
        )

    def find_movie_id(self, query: str) -> int | None:
        """Return the TMDb id of the movie a title query refers to, or None."""
        match = self._resolve_title(query)
        if match is None:
            return None
        row, _is_exact = match
        return int(self._ids[row])

    def title_of(self, movie_id: int) -> str:
        """Return the dataset title of a movie id, e.g. one that find_movie_id returned."""
        return str(self.movies_df["title"].iat[self.row_of[movie_id]])

    def _resolve_title(self, query) -> tuple[int, bool] | None:
        """Return (row, is_exact) for a title query; exact matches win.

        A title or Turkish title is an exact match when it equals the query without case,
        accents or apostrophes; of several, the most-voted film wins. A query inside an
        original title is that title being typed, so only the titles count for it. Otherwise
        the most-voted film whose Turkish title starts with the query wins, and then the
        closer fuzzy match of the titles and the Turkish titles (see _closest_titles). Without
        Turkish titles (the Kaggle files) only the titles count.
        """
        if not isinstance(query, str):
            return None
        query = query.strip()[:MAX_QUERY_LENGTH]
        if len(query) < MIN_QUERY_LENGTH:
            return None
        query_key = title_key(query)
        # One True/False per film: is its title or Turkish title the query? "Enigma" is a
        # film and the Turkish title of the far better-known The Imitation Game.
        is_same = ((self._title_keys == query_key) | (self._turkish_keys == query_key)).to_numpy()
        if query_key and is_same.any():
            return self._most_voted_row(is_same), True

        title_match = self._closest_titles(query, self.all_movie_titles)
        turkish_match = None
        # "terminator 2" is Terminator 2: Judgment Day, not The Terminator ("Terminatör").
        if (
            self._all_turkish_keys
            and query_key
            and not self._title_keys.str.contains(query_key, regex=False).any()
        ):
            starts_turkish = self._turkish_keys.str.startswith(query_key).to_numpy()
            if starts_turkish.any():
                return self._most_voted_row(starts_turkish), False
            turkish_match = self._closest_titles(query_key, self._all_turkish_keys)
        ranks = [match[1] for match in (title_match, turkish_match) if match]
        if not ranks:
            return None
        # The better-ranked match wins. On equal ranks the most-voted film of both does:
        # "enimga" is as close to the film Enigma as to the Turkish title of The Imitation Game.
        is_matched = np.zeros(len(self._ids), dtype=bool)
        if title_match and title_match[1] == max(ranks):
            is_matched |= self.movies_df["title"].isin(title_match[0]).to_numpy()
        if turkish_match and turkish_match[1] == max(ranks):
            is_matched |= self._turkish_keys.isin(turkish_match[0]).to_numpy()
        return self._most_voted_row(is_matched), False

    def _most_voted_row(self, is_candidate: np.ndarray) -> int:
        """Return the most-voted row among those where ``is_candidate`` is True.

        Remakes share a title; the better-known film is the likelier target. On equal votes
        the first row wins.
        """
        candidate_rows = np.flatnonzero(is_candidate)  # positions of the True values
        candidate_votes = self._vote_count[candidate_rows]
        best = np.argmax(candidate_votes)
        return int(candidate_rows[best])

    def recommend_by_similarity(
        self, movie_title_query: str, num: int | str = DEFAULT_RECOMMENDATION_COUNT
    ) -> tuple[list[dict], list[dict]]:
        """Recommend movies similar to the given title; returns (results, warnings).

        Each warning is a movierec.i18n.message(), which the page shows in its language.
        """
        if not isinstance(movie_title_query, str) or not movie_title_query.strip():
            return [], [message("warning.empty_query")]
        num = clamp_count(num)
        query = movie_title_query.strip()[:MAX_QUERY_LENGTH]
        match = self._resolve_title(query)
        if match is None:
            return [], [message("warning.movie_not_found", query=query)]

        row, is_exact = match
        title = self.movies_df["title"].iat[row]
        warnings = []
        if not is_exact:
            warnings.append(message("warning.not_exact", query=query, title=title))
        ranked_rows = self.rank_neighbours(row)
        # rank_neighbours keeps films under MIN_VOTES_THRESHOLD only if its pool has nothing better.
        is_well_voted = self._vote_count[ranked_rows] >= MIN_VOTES_THRESHOLD
        if not is_well_voted.any():
            warnings.append(message("warning.less_known_similar", title=title))
        neighbours = self.movies_df.iloc[ranked_rows]
        return self.format_results(neighbours, num), warnings

    def similar_movie_ids(self, movie_id: int, num: int = DETAIL_SIMILAR_COUNT) -> list[int]:
        """Return the ids recommended next to a dataset movie, best first, without TMDb calls."""
        row = self.row_of.get(int(movie_id))
        if row is None:
            return []
        ranked_rows = self.rank_neighbours(row)
        top_rows = ranked_rows[: clamp_count(num)]
        return self._ids[top_rows].tolist()

    def recommend_from_favourites(
        self, favourite_ids: list[int], exclude_ids: list[int]
    ) -> list[int]:
        """Return ids of films like the favourites, best first, without favourites and exclude_ids.

        Each favourite's similar list gives the film at 0-based rank r a score of 1 / (r + 1);
        the scores add up over the favourites, and ties keep the order films first appear in.
        At most DETAIL_SIMILAR_COUNT ids are returned.
        """
        # Favourites outside the dataset have no similar list; of the rest, the first
        # MAX_FAVOURITES_USED count.
        in_dataset = [movie_id for movie_id in favourite_ids if movie_id in self.row_of]
        used = in_dataset[:MAX_FAVOURITES_USED]
        already_listed = set(favourite_ids) | set(exclude_ids)

        scores = {}
        for favourite in used:
            similar_ids = self.similar_movie_ids(favourite, FAVOURITE_NEIGHBOURS)
            for rank, movie_id in enumerate(similar_ids):
                if movie_id in already_listed:
                    continue
                scores[movie_id] = scores.get(movie_id, 0) + 1 / (rank + 1)

        # Ids by total score, highest first. sorted() is stable with reverse=True too, so equal
        # totals keep the order the films were first scored in.
        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return ranked_ids[:DETAIL_SIMILAR_COUNT]

    def rank_neighbours(self, row: int) -> np.ndarray:
        """Return the row positions of films most similar to ``row``, best hybrid score first."""
        # Rows are L2-normalised, so this row times the transposed matrix is the film's cosine
        # similarity to every film: a 1 x N sparse matrix, flattened to a plain array.
        product = self.tfidf_matrix[row] @ self.tfidf_matrix.T
        similarity = product.toarray().ravel()
        # argsort sorts ascending, so sort the negated scores; "stable" keeps ties in dataset order.
        by_similarity = np.argsort(-similarity, kind="stable")
        # Leave out the film itself and keep the closest NEIGHBOUR_POOL_SIZE for re-ranking.
        others = by_similarity[by_similarity != row]
        pool = others[:NEIGHBOUR_POOL_SIZE]
        hybrid = hybrid_score(
            similarity[pool],
            self._vote_average[pool],
            self._vote_count[pool],
            self._popularity[pool],
        )
        # Below MIN_VOTES_THRESHOLD votes TMDb ratings are noisy, so such films are only
        # shown when the pool has nothing better.
        is_well_voted = self._vote_count[pool] >= MIN_VOTES_THRESHOLD
        if is_well_voted.any():
            pool = pool[is_well_voted]
            hybrid = hybrid[is_well_voted]
        best_first = np.argsort(-hybrid, kind="stable")
        return pool[best_first]

    def recommend_by_features(
        self,
        genres: list[str] | str | None = None,
        director_query: str | None = None,
        min_rating: str | float | None = None,
        min_year: str | int | None = None,
        max_year: str | int | None = None,
        num: int | str = DEFAULT_RECOMMENDATION_COUNT,
    ) -> tuple[list[dict], list[dict]]:
        """Recommend movies matching genre, director, rating and year filters.

        Returns (results, warnings) like recommend_by_similarity.
        """
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

        numeric_filters = parse_numeric_filters(min_rating, min_year, max_year)
        warnings.extend(numeric_filters.warnings)
        # director_match is None or a dataset name, never "".
        has_criteria = (
            bool(wanted_genres) or director_match is not None or numeric_filters.has_bounds
        )
        if not has_criteria:
            return [], [message("warning.no_criteria"), *warnings]

        matches = apply_filters(self.movies_df, wanted_genres, director_match, numeric_filters)
        if matches.empty:
            return [], [message("warning.no_match"), *warnings]

        candidates = matches[matches["vote_count"] >= MIN_VOTES_THRESHOLD]
        if candidates.empty:
            candidates = matches
            warnings.append(message("warning.less_known"))
        scores = feature_score(candidates)
        ranked = candidates.assign(feature_score=scores)
        ranked = ranked.sort_values("feature_score", ascending=False, kind="stable")
        return self.format_results(ranked, num), warnings

    def _match_director(self, query: str) -> tuple[str | None, dict | None]:
        """Return (director name or None, warning or None) for a free-text director query."""
        wanted = query.casefold()
        for name in self.all_director_names:
            if name.casefold() == wanted:
                return name, None
        # "Alfonso Cuaron" names Alfonso Cuarón exactly; accents and punctuation do not count.
        exact = self._directors_by_key.get(title_key(query))
        if exact is not None:
            return exact, None
        closest = self._find_best_match(query, self.all_director_names)
        if closest is not None and is_plausible_name(query, closest):
            return closest, message("warning.director_fuzzy", query=query, name=closest)
        return None, message("warning.director_not_found", query=query)


def _words(title: str) -> set[str]:
    """Return the _WORD words of title_key(title), with "E.T." as the word "et"."""
    return set(_WORD.findall(title_key(title.replace(".", ""))))


def clamp_count(num: int | str | None) -> int:
    """Return a requested result count (e.g. the ?n= text) limited to 1..MAX_RECOMMENDATION_COUNT.

    Values int() cannot convert give DEFAULT_RECOMMENDATION_COUNT.
    """
    try:
        count = int(num)
    except (TypeError, ValueError):
        return DEFAULT_RECOMMENDATION_COUNT
    if count < 1:
        return 1
    if count > MAX_RECOMMENDATION_COUNT:
        return MAX_RECOMMENDATION_COUNT
    return count
