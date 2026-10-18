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

INVALID_NUMBER = "Geçersiz puan veya yıl formatı girildi. Lütfen sadece sayı kullanın."
YEARS_SWAPPED = "Yıl aralığı ters girilmişti, düzeltildi."

_NON_ALNUM = re.compile(r"[\W_]+")


@dataclass(frozen=True)
class NumericFilters:
    """Validated rating and year bounds plus the warnings produced while parsing them."""

    min_rating: float | None = None
    min_year: int | None = None
    max_year: int | None = None
    warnings: tuple[str, ...] = ()

    @property
    def count(self) -> int:
        return sum(value is not None for value in (self.min_rating, self.min_year, self.max_year))


def latest_year() -> int:
    """Latest year accepted in the year filters (announced films carry next year's date)."""
    return datetime.date.today().year + 1


def parse_number(value, cast, low, high):
    """Parse a form value, accepting a Turkish decimal comma ('7,5').

    Returns None for non-numeric, non-finite or out-of-range input; ``cast`` (int or
    float) is applied before the range check, so int truncates like JS Math.trunc.
    """
    try:
        number = float(str(value).strip().replace(",", "."))
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    number = cast(number)
    return number if low <= number <= high else None


def parse_numeric_filters(min_rating=None, min_year=None, max_year=None) -> NumericFilters:
    """Validate each numeric field on its own; one bad field never discards the others."""
    invalid = False

    def parse(value, cast, low, high):
        nonlocal invalid
        if value is None or not str(value).strip():
            return None
        number = parse_number(value, cast, low, high)
        invalid = invalid or number is None
        return number

    rating = parse(min_rating, float, 0.0, RATING_MAX)
    first = parse(min_year, int, MIN_YEAR, latest_year())
    last = parse(max_year, int, MIN_YEAR, latest_year())
    warnings = [INVALID_NUMBER] if invalid else []
    if first is not None and last is not None and first > last:
        first, last = last, first
        warnings.append(YEARS_SWAPPED)
    return NumericFilters(rating, first, last, tuple(warnings))


def genre_keys(genres) -> frozenset[str]:
    """Casefolded genre names from a list (or a single name) of form values."""
    if isinstance(genres, str):
        genres = [genres]
    return frozenset(
        genre.strip().casefold()
        for genre in genres or []
        if isinstance(genre, str) and genre.strip()
    )


def name_key(text: str) -> str:
    """Comparison key for people's names: casefolded, no accents, punctuation as spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _NON_ALNUM.sub(" ", plain.casefold()).strip()


def is_plausible_name(query: str, name: str) -> bool:
    """Return True when a fuzzy director match plausibly names the person the user meant."""
    wanted, candidate = name_key(query), name_key(name)
    if fuzz.ratio(wanted, candidate) >= NAME_CLOSE_RATIO:
        return True
    name_tokens = candidate.split()
    best = [max(fuzz.ratio(token, other) for other in name_tokens) for token in wanted.split()]
    return (
        bool(best)
        and all(score >= NAME_TOKEN_RATIO for score in best)
        and any(score >= NAME_CLOSE_RATIO for score in best)
    )


def apply_filters(
    movies: pd.DataFrame,
    genres: frozenset[str],
    director: str | None,
    numbers: NumericFilters,
) -> pd.DataFrame:
    """Keep movies with every wanted genre, the director, and rating/year in range."""
    df = movies
    if genres:
        df = df[df["genre_list"].map(lambda names: genres <= {name.casefold() for name in names})]
    if director:
        df = df[df["directors"].map(lambda names: director in names)]
    if numbers.min_rating is not None:
        df = df[df["vote_average"] >= numbers.min_rating]
    # Year 0 marks a missing release date, which never satisfies a year bound.
    if numbers.min_year is not None:
        df = df[(df["year"] >= numbers.min_year) & (df["year"] != 0)]
    if numbers.max_year is not None:
        df = df[(df["year"] <= numbers.max_year) & (df["year"] != 0)]
    return df
