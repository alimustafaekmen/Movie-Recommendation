"""Load the TMDB 5000 movies and credits CSVs into one clean DataFrame."""

import json
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

MOVIE_COLUMNS = [
    "id",
    "title",
    "overview",
    "genres",
    "keywords",
    "release_date",
    "vote_average",
    "vote_count",
    "popularity",
    "tagline",
    "runtime",
]
CREDIT_COLUMNS = ["movie_id", "cast", "crew"]
# The Turkish title, when it differs from the title. Only the catalogue that
# scripts/fetch_catalogue.py writes has it; without the column every film's is "".
OPTIONAL_MOVIE_COLUMNS = ["title_tr"]
# Stand-ins for empty cells: "[]" for the JSON list columns, "" for text.
TEXT_DEFAULTS = {
    "title_tr": "",
    "overview": "",
    "genres": "[]",
    "keywords": "[]",
    "cast": "[]",
    "crew": "[]",
    "tagline": "",
}
CAST_LIMIT = 15


def load_movies(movies_path: str, credits_path: str) -> pd.DataFrame:
    """Read both CSVs and return one row per movie with parsed, typed columns."""
    movies, credits = _read_csvs(movies_path, credits_path)
    return _preprocess(movies, credits)


def _read_csvs(movies_path: str, credits_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two CSVs, keeping only rows whose id is a number."""
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Movies CSV not found: {movies_path}")
    if not os.path.exists(credits_path):
        raise FileNotFoundError(f"Credits CSV not found: {credits_path}")

    header = pd.read_csv(movies_path, nrows=0).columns
    optional = [column for column in OPTIONAL_MOVIE_COLUMNS if column in header]
    movies = pd.read_csv(movies_path, usecols=MOVIE_COLUMNS + optional)
    for column in OPTIONAL_MOVIE_COLUMNS:
        if column not in movies:
            movies[column] = ""
    credits = pd.read_csv(credits_path, usecols=CREDIT_COLUMNS)
    movies = _keep_numeric_ids(movies, "id")
    credits = _keep_numeric_ids(credits, "movie_id")
    logger.info("Read %d movie rows and %d credit rows", len(movies), len(credits))
    return movies, credits


def _keep_numeric_ids(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return ``df`` without the rows whose ``column`` is not a number, with that column as int.

    df: one raw CSV as read by pandas.
    """
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])
    df[column] = df[column].astype(int)
    return df


def _preprocess(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    """Join credits onto movies and turn the raw columns into parsed, typed ones.

    movies, credits: the raw CSV frames from _read_csvs, before the merge.
    """
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
    # fillna with a dict gives each column its own default.
    df = df.drop(columns=["movie_id"]).fillna(TEXT_DEFAULTS)

    # The list columns keep co-directors and multi-word genres intact for TF-IDF, filtering
    # and display; TF-IDF reads keywords from the space-joined string, word by word.
    # apply() passes job= on to parse_name_list.
    df["directors"] = df["crew"].apply(parse_name_list, job="Director")
    # .str[0] takes the first item of each list (NaN when it is empty).
    df["director"] = df["directors"].str[0].fillna("")
    df["genre_list"] = df["genres"].apply(parse_name_list)
    df["genres"] = df["genre_list"].str.join(" ")
    df["keywords"] = df["keywords"].apply(parse_name_list).str.join(" ")
    cast_lists = df["cast"].apply(parse_name_list)
    # .str also works on lists: keep the first CAST_LIMIT names, then join them.
    df["cast"] = cast_lists.str[:CAST_LIMIT].str.join(", ")

    release_dates = pd.to_datetime(df["release_date"], errors="coerce")  # unreadable dates -> NaT
    # Year 0 marks a missing or unreadable release date.
    df["year"] = release_dates.dt.year.fillna(0).astype(int)
    df["release_date"] = df["release_date"].fillna("").astype(str)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0).astype(int)
    for column in ("title", "title_tr", "overview", "director", "tagline", "cast"):
        df[column] = df[column].astype(str)

    return df.drop(columns=["crew"])


def parse_name_list(json_str, job: str | None = None) -> list[str]:
    """Return every 'name' in a JSON list column, or only crew members with ``job``."""
    if not isinstance(json_str, str) or not json_str:
        return []
    try:
        data = json.loads(json_str)
    except ValueError:
        return []
    if not isinstance(data, list):
        return []
    names = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if job is not None and item.get("job") != job:
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return names
