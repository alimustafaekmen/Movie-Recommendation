"""Movie detail pages and list cards, enriched with live TMDb data."""

import logging
import math

import pandas as pd

from movierec import tmdb
from movierec.engine import DETAIL_SIMILAR_COUNT, RecommendationEngine
from movierec.genres import TMDB_GENRE_NAMES, genre_badge, is_excluded

log = logging.getLogger(__name__)

# Any visitor can make the app look up arbitrary ids, so the poster cache is emptied once it
# holds this many films; that is simpler than LRU, and the whole dataset still fits.
POSTER_CACHE_LIMIT = 5000


class MovieDetails:
    """Builds the detail dictionaries on top of an engine and remembers poster paths."""

    def __init__(self, engine: RecommendationEngine):
        self.engine = engine
        # TMDb poster path (None when the film has none) per movie id.
        self.poster_cache: dict[int, str | None] = {}

    def add_posters(self, cards: list[dict]) -> None:
        """Fill in the poster URL of result cards from the engine, which works offline."""
        for card in cards:
            card["poster_url"] = self.get_movie_poster_url(card["id"])

    def get_movie_poster_url(self, movie_id: int | str, size: str = "w342") -> str | None:
        """Return the TMDb poster URL for a movie; the poster path is cached per id."""
        if not tmdb.is_configured():
            return None
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            return None

        if movie_id not in self.poster_cache:
            self._fetch_movie(movie_id, language="tr-TR")
        return tmdb.image_url(self.poster_cache.get(movie_id), size)

    def _fetch_movie(self, movie_id: int, **params) -> dict | None:
        """GET /movie/{id} and remember its poster path; None if unavailable."""
        try:
            data = tmdb.get(f"/movie/{movie_id}", **params)
        except tmdb.TmdbUnavailable:
            return None
        # Only an actual answer is cached, so a network blip does not hide a poster
        # until the next restart.
        if movie_id not in self.poster_cache and len(self.poster_cache) >= POSTER_CACHE_LIMIT:
            self.poster_cache.clear()
        self.poster_cache[movie_id] = (data or {}).get("poster_path")
        return data

    def get_movie_details(self, movie_id: int | str) -> dict | None:
        """Return the detail-page dictionary for a movie, from the dataset or TMDb."""
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            log.error("Invalid movie id: %r", movie_id)
            return None

        index = self.engine.row_of.get(movie_id)
        in_dataset = index is not None
        movie = self.engine.movies_df.iloc[index].to_dict() if in_dataset else None
        # One request per page: credits are only needed for movies outside the dataset.
        online = self._fetch_movie(
            movie_id,
            language="tr-TR",
            append_to_response="videos" if in_dataset else "credits,videos",
            include_video_language=tmdb.VIDEO_LANGUAGES,
        )
        if in_dataset:
            if online:
                _apply_turkish_texts(movie, online)
            movie["genre_details_for_display"] = _genre_badges(movie["genre_list"])
        elif online:
            movie = self._movie_from_tmdb(online)
        if movie is None:
            return None

        movie["poster_url_large"] = self.get_movie_poster_url(movie_id, size="w500")
        movie["trailer_key"] = tmdb.pick_trailer(
            ((online or {}).get("videos") or {}).get("results", [])
        )
        _clean_detail_fields(movie)
        movie["runtime_formatted"] = format_runtime(movie.get("runtime"))
        if in_dataset:
            ranked = self.engine.rank_neighbours(index)
            movie["similar_movies"] = self.engine.format_results(
                self.engine.movies_df.iloc[ranked], DETAIL_SIMILAR_COUNT
            )
            self.add_posters(movie["similar_movies"])
        else:
            movie["similar_movies"] = self._get_similar_from_api(movie_id)
        return movie

    def _movie_from_tmdb(self, data: dict) -> dict:
        # tr-TR responses name genres in Turkish; the id gives the English value.
        genre_list = [
            TMDB_GENRE_NAMES.get(genre.get("id"), genre.get("name"))
            for genre in data.get("genres", [])
            if genre.get("id") and genre.get("name")
        ]
        credits = data.get("credits") or {}
        directors = [
            crew["name"]
            for crew in credits.get("crew", [])
            if crew.get("job") == "Director" and crew.get("name")
        ]
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "overview": data.get("overview"),
            "tagline": data.get("tagline"),
            "genres": " ".join(sorted({genre for genre in genre_list if not is_excluded(genre)})),
            "genre_list": genre_list,
            "genre_details_for_display": _genre_badges(genre_list),
            "year": _release_year(data),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "popularity": data.get("popularity"),
            "cast": ", ".join(actor["name"] for actor in credits.get("cast", [])[:15]),
            "director": directors[0] if directors else "",
            "directors": directors,
            # TMDb's details endpoint has no keywords; the field only exists for the template.
            "keywords": "",
            "runtime": data.get("runtime"),
        }

    def _get_similar_from_api(self, movie_id: int) -> list[dict]:
        try:
            data = tmdb.get(f"/movie/{movie_id}/similar", language="tr-TR", page=1) or {}
        except tmdb.TmdbUnavailable:
            return []
        return [
            {
                "id": item["id"],
                "title": item["title"],
                "poster_url": tmdb.image_url(item.get("poster_path"), "w342"),
                "year": _release_year(item),
            }
            for item in data.get("results", [])[:DETAIL_SIMILAR_COUNT]
            if item.get("id") and item.get("title")
        ]

    def get_movie_info_for_list(self, movie_id: int | str) -> dict | None:
        """Return the short card used on the favourites and watchlist page."""
        try:
            movie_id = int(movie_id)
        except (TypeError, ValueError):
            return None
        index = self.engine.row_of.get(movie_id)
        if index is not None:
            row = self.engine.movies_df.iloc[index]
            return {
                "id": movie_id,
                "title": row["title"],
                "year": int(row["year"]),
                "vote_average": round(float(row["vote_average"]), 1),
                "poster_url": self.get_movie_poster_url(movie_id, size="w342"),
            }
        data = self._fetch_movie(movie_id, language="tr-TR")
        if not data or not data.get("title"):
            return None
        return {
            "id": int(data.get("id", movie_id)),
            "title": data["title"],
            "year": _release_year(data),
            "vote_average": round(float(data.get("vote_average") or 0.0), 1),
            "poster_url": tmdb.image_url(data.get("poster_path"), "w342"),
        }


def _genre_badges(genre_list) -> list[dict]:
    return [genre_badge(genre) for genre in genre_list if not is_excluded(genre)]


def _apply_turkish_texts(movie, online):
    # The dataset is English-only and the UI is Turkish, so TMDb's tr-TR texts win.
    for field in ("overview", "tagline", "runtime"):
        if online.get(field):
            movie[field] = online[field]


def _release_year(data) -> int:
    release_date = data.get("release_date") or ""
    return int(release_date[:4]) if release_date else 0


def _clean_detail_fields(movie):
    """Coerce detail fields to the plain types the template expects."""
    year = movie.get("year", 0)
    movie["year"] = int(year) if pd.notna(year) else 0
    vote_average = movie.get("vote_average", 0.0)
    movie["vote_average"] = round(float(vote_average), 1) if pd.notna(vote_average) else 0.0
    vote_count = movie.get("vote_count", 0)
    movie["vote_count"] = int(vote_count) if pd.notna(vote_count) else 0
    popularity = movie.get("popularity", 0.0)
    movie["popularity"] = float(popularity) if pd.notna(popularity) else 0.0
    for field in ("title", "overview", "genres", "keywords", "cast", "director", "tagline"):
        movie[field] = str(movie.get(field, ""))
    for field in ("genre_list", "directors"):
        movie[field] = list(movie.get(field) or [])


def format_runtime(minutes) -> str | None:
    """Format a runtime in minutes as Turkish hours/minutes, e.g. 152 -> '2 sa 32 dk'."""
    try:
        value = float(minutes)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or int(value) <= 0:
        return None

    hours, mins = divmod(int(value), 60)
    if hours and mins:
        return f"{hours} sa {mins} dk"
    if hours:
        return f"{hours} sa"
    return f"{mins} dk"
