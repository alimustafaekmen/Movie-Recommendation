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
TEXT_DEFAULTS = {
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
    for label, path in (("Movies", movies_path), ("Credits", credits_path)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} CSV not found: {path}")

    movies = pd.read_csv(movies_path, usecols=MOVIE_COLUMNS)
    credits = pd.read_csv(credits_path, usecols=CREDIT_COLUMNS)
    movies = _with_int_ids(movies, "id")
    credits = _with_int_ids(credits, "movie_id")
    logger.info("Read %d movie rows and %d credit rows", len(movies), len(credits))
    return movies, credits


def _with_int_ids(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=[column])
    df[column] = df[column].astype(int)
    return df


def _preprocess(movies: pd.DataFrame, credits: pd.DataFrame) -> pd.DataFrame:
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="inner")
    df = df.drop(columns=["movie_id"]).fillna(TEXT_DEFAULTS)

    # The list columns keep co-directors and multi-word genres intact for filtering and
    # display; the space-joined strings feed TF-IDF.
    df["directors"] = df["crew"].apply(parse_name_list, job="Director")
    df["director"] = df["directors"].str[0].fillna("")
    df["genre_list"] = df["genres"].apply(parse_name_list)
    df["genres"] = df["genre_list"].str.join(" ")
    df["keywords"] = df["keywords"].apply(parse_name_list).str.join(" ")
    df["cast"] = df["cast"].apply(parse_name_list).str[:CAST_LIMIT].str.join(", ")

    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    df["release_date"] = df["release_date"].fillna("").astype(str)
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0).astype(int)
    for column in ("title", "overview", "director", "tagline", "cast"):
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
        if not isinstance(item, dict) or (job is not None and item.get("job") != job):
            continue
        name = str(item.get("name") or "").strip()
        if name:
            names.append(name)
    return names
