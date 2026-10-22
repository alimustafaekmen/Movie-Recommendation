"""Parse and apply the feature-search filters: genres, director, minimum rating and years."""

import datetime
import math
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from thefuzz import fuzz

from movierec.similarity import RATING_MAX

MIN_YEAR = 1900
# A fuzzy director match must also look like the same person: the whole name is close, or
# every query word resembles a word of the name and one is near-identical. 'nolan' still
# works; 'John Smith' does not become 'Brian Trenchard-Smith'.
NAME_CLOSE_RATIO = 85
NAME_TOKEN_RATIO = 70

INVALID_NUMBER_WARNING = "Geçersiz puan veya yıl formatı girildi. Lütfen sadece sayı kullanın."
YEARS_SWAPPED_WARNING = "Yıl aralığı ters girilmişti, düzeltildi."

# Runs of anything but letters and digits; \W alone would keep "_".
_NON_ALNUM = re.compile(r"[\W_]+")


@dataclass(frozen=True)
class NumericFilters:
    """Validated rating and year bounds plus the warnings produced while parsing them."""

    min_rating: float | None = None
    min_year: int | None = None
    max_year: int | None = None
    warnings: tuple[str, ...] = ()

    @property
    def has_bounds(self) -> bool:
        """Return True when at least one of min_rating, min_year and max_year is set."""
        return self.min_rating is not None or self.min_year is not None or self.max_year is not None


def latest_year() -> int:
    """Return the latest year the year filters accept (announced films carry next year's date)."""
    return datetime.date.today().year + 1


def parse_number(
    value, cast: type[int] | type[float], low: float, high: float
) -> int | float | None:
    """Parse a form value, accepting a Turkish decimal comma ('7,5').

    Returns None for non-numeric, non-finite or out-of-range input; ``cast`` (int or
    float) is applied before the range check, so int truncates like JS Math.trunc.
    """
    # Turkish forms write a decimal comma: "7,5" means 7.5.
    text = str(value).strip().replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):  # "nan" and "inf" parse as floats
        return None
    number = cast(number)
    if low <= number <= high:
        return number
    return None


def _parse_field(
    value, cast: type[int] | type[float], low: float, high: float
) -> tuple[int | float | None, bool]:
    """Return (number or None, is_invalid) for one form field; an empty field is not invalid."""
    if value is None or not str(value).strip():
        return None, False
    number = parse_number(value, cast, low, high)
    return number, number is None


def parse_numeric_filters(
    min_rating: str | float | None = None,
    min_year: str | int | None = None,
    max_year: str | int | None = None,
) -> NumericFilters:
    """Validate each numeric field on its own; one bad field never discards the others."""
    newest_year = latest_year()
    rating, rating_is_invalid = _parse_field(min_rating, float, 0.0, RATING_MAX)
    first_year, first_is_invalid = _parse_field(min_year, int, MIN_YEAR, newest_year)
    last_year, last_is_invalid = _parse_field(max_year, int, MIN_YEAR, newest_year)
    warnings = []
    if rating_is_invalid or first_is_invalid or last_is_invalid:
        warnings.append(INVALID_NUMBER_WARNING)
    if first_year is not None and last_year is not None and first_year > last_year:
        first_year, last_year = last_year, first_year
        warnings.append(YEARS_SWAPPED_WARNING)
    return NumericFilters(rating, first_year, last_year, tuple(warnings))


def genre_keys(genres: list[str] | str | None) -> frozenset[str]:
    """Return the casefolded genre names from a list (or a single name) of form values."""
    if isinstance(genres, str):
        genres = [genres]
    if not genres:
        return frozenset()
    keys = set()
    for genre in genres:
        if not isinstance(genre, str):
            continue
        key = genre.strip().casefold()
        if key:
            keys.add(key)
    return frozenset(keys)


def name_key(text: str) -> str:
    """Return a person's name casefolded, without accents, with punctuation as spaces."""
    # NFKD splits an accented letter into the letter plus a combining accent ("é" -> "e" + "´");
    # dropping the combining characters leaves the plain letter.
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    # Each run of punctuation, symbols or underscores becomes one space:
    # "Trenchard-Smith" -> "trenchard smith".
    spaced = _NON_ALNUM.sub(" ", plain.casefold())
    return spaced.strip()


def is_plausible_name(query: str, name: str) -> bool:
    """Return True when a fuzzy director match plausibly names the person the user meant."""
    wanted = name_key(query)
    candidate = name_key(name)
    if fuzz.ratio(wanted, candidate) >= NAME_CLOSE_RATIO:
        return True
    name_words = candidate.split()
    # For each query word, the score of the name word it resembles most.
    best_scores = []
    for word in wanted.split():
        best_scores.append(max(fuzz.ratio(word, name_word) for name_word in name_words))
    if not best_scores:
        return False
    # Every query word must resemble a word of the name, and one must be near-identical.
    every_word_similar = all(score >= NAME_TOKEN_RATIO for score in best_scores)
    one_word_close = any(score >= NAME_CLOSE_RATIO for score in best_scores)
    return every_word_similar and one_word_close


def apply_filters(
    df: pd.DataFrame,
    wanted_genres: frozenset[str],
    director: str | None,
    numeric_filters: NumericFilters,
) -> pd.DataFrame:
    """Keep the films that match every filter.

    df: the movies table built by dataset.load_movies, one row per film.
    wanted_genres: casefolded names from genre_keys; a film needs all of them.
    director: an exact name from the directors column, or None.
    """
    if wanted_genres:
        # One True/False per film: does it have every wanted genre?
        has_every_genre = df["genre_list"].map(
            lambda names: wanted_genres.issubset(name.casefold() for name in names)
        )
        df = df[has_every_genre]
    if director:
        has_director = df["directors"].map(lambda names: director in names)
        # On an empty frame map() returns an object Series, which pandas would read as
        # column names; astype(bool) keeps it a True/False filter.
        df = df[has_director.astype(bool)]
    if numeric_filters.min_rating is not None:
        df = df[df["vote_average"] >= numeric_filters.min_rating]
    has_year_bound = numeric_filters.min_year is not None or numeric_filters.max_year is not None
    if has_year_bound:
        # Year 0 marks a missing release date, which never satisfies a year bound.
        df = df[df["year"] != 0]
    if numeric_filters.min_year is not None:
        df = df[df["year"] >= numeric_filters.min_year]
    if numeric_filters.max_year is not None:
        df = df[df["year"] <= numeric_filters.max_year]
    return df
