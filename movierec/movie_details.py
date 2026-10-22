"""Movie detail pages and list cards, enriched with live TMDb data."""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, wait

from movierec import tmdb
from movierec.dataset import CAST_LIMIT
from movierec.engine import DETAIL_SIMILAR_COUNT, RecommendationEngine
from movierec.genres import TMDB_GENRE_NAMES, genre_badge, is_excluded

logger = logging.getLogger(__name__)

# Any visitor can make the app look up arbitrary ids, so the poster cache is emptied once it
# holds this many films; that is simpler than LRU, and the whole dataset still fits.
POSTER_CACHE_LIMIT = 5000
# Uncached posters are fetched this many at a time. The pool is shared by all requests, so
# this caps the lookups in flight however many pages load at once: at 16 in parallel, TMDb
# answered several times slower.
POSTER_WORKERS = 8
# A page waits this long for its posters. A slow TMDb leaves the rest as placeholders; the
# lookups under way still fill the cache for the next visit.
POSTER_WAIT_SECONDS = 5.0
CAST_LIST_LIMIT = 12
# The dataset joins cast names with ", ", which some names also contain ("Sammy Davis, Jr.").
_NAME_SUFFIXES = {"Jr.", "Jr", "Sr.", "Sr"}
# Marks a poster cache miss; None is a cached "this film has no poster".
_UNCACHED = object()


class MovieDetails:
    """Builds the detail dictionaries on top of an engine and remembers poster paths."""

    def __init__(self, engine: RecommendationEngine):
        self.engine = engine
        # TMDb poster path (None when the film has none) per movie id.
        self.poster_cache: dict[int, str | None] = {}
        # Threads start on first use, so an app without a TMDb key never creates any, and none
        # exist before gunicorn forks its worker. At exit Python joins them; each lookup is
        # bounded by tmdb.TIMEOUT and stops at once during the cooldown.
        self._poster_pool = ThreadPoolExecutor(POSTER_WORKERS, thread_name_prefix="poster")

    def add_posters(self, cards: list[dict]) -> None:
        """Fill in the poster URL of result cards from the engine, which works offline."""
        paths = self.prefetch_posters([card["id"] for card in cards])
        for card in cards:
            card["poster_url"] = tmdb.image_url(paths.get(card["id"]), "w342")

    def prefetch_posters(self, movie_ids: list[int]) -> dict[int, str | None]:
        """Return the poster path per movie id; uncached ones are fetched in parallel."""
        if not tmdb.is_configured():
            return {}
        # The fetched paths are returned directly instead of being read back from the cache,
        # which a concurrent lookup may empty at any time.
        paths = {movie_id: self.poster_cache.get(movie_id, _UNCACHED) for movie_id in movie_ids}
        missing = [movie_id for movie_id, path in paths.items() if path is _UNCACHED]
        futures = [
            self._poster_pool.submit(self._fetch_poster_path, movie_id) for movie_id in missing
        ]
        done, late = wait(futures, timeout=POSTER_WAIT_SECONDS)
        for future in late:
            # Only a lookup that has not started is cancelled: no more work for a slow TMDb.
            future.cancel()
        paths.update(
            (movie_id, future.result() if future in done else None)
            for movie_id, future in zip(missing, futures, strict=True)
        )
        return paths

    def _fetch_poster_path(self, movie_id: int) -> str | None:
        # A poster is decoration: no error of its lookup may fail the page that shows it.
        try:
            data = self._fetch_movie(movie_id, language="tr-TR") or {}
        except Exception:
            logger.exception("Poster lookup for movie %s failed", movie_id)
            return None
        return tmdb.text(data.get("poster_path")) or None

    def get_movie_poster_url(self, movie_id: int | str, size: str = "w342") -> str | None:
        """Return the TMDb poster URL for a movie; the poster path is cached per id."""
        if not tmdb.is_configured():
            return None
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            return None
        # Like prefetch_posters, a fetched path is used directly: a concurrent request may
        # empty the cache right after it is written.
        path = self.poster_cache.get(movie_id, _UNCACHED)
        if path is _UNCACHED:
            path = self._fetch_poster_path(movie_id)
        return tmdb.image_url(path, size)

    def _fetch_movie(self, movie_id: int, *, strict: bool = False, **params) -> dict | None:
        """GET /movie/{id} and remember its poster path; None if unavailable.

        strict raises TmdbUnavailableError instead when a configured TMDb fails.
        """
        try:
            data = tmdb.get(f"/movie/{movie_id}", **params)
        except tmdb.TmdbUnavailableError:
            if strict and tmdb.is_configured():
                raise
            return None
        # Only an actual answer is cached, so a network blip does not hide a poster
        # until the next restart.
        if movie_id not in self.poster_cache and len(self.poster_cache) >= POSTER_CACHE_LIMIT:
            self.poster_cache.clear()
        self.poster_cache[movie_id] = tmdb.text((data or {}).get("poster_path")) or None
        return data

    def get_movie_details(self, movie_id: int | str) -> dict | None:
        """Return the detail-page dictionary for a movie, from the dataset or TMDb.

        A film outside the dataset has no page without TMDb, so while a configured TMDb fails
        it raises TmdbUnavailableError instead of returning None like an unknown id.
        """
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            logger.error("Invalid movie id: %r", movie_id)
            return None

        index = self.engine.row_of.get(movie_id)
        in_dataset = index is not None
        movie = self.engine.movies_df.iloc[index].to_dict() if in_dataset else None
        # One request per page: credits are only needed for movies outside the dataset.
        tmdb_movie = self._fetch_movie(
            movie_id,
            strict=not in_dataset,
            language="tr-TR",
            append_to_response="videos" if in_dataset else "credits,videos",
            include_video_language=tmdb.VIDEO_LANGUAGES,
        )
        if in_dataset:
            if tmdb_movie:
                _apply_turkish_texts(movie, tmdb_movie)
            movie["genre_badges"] = _genre_badges(movie["genre_list"])
            movie["cast_list"] = cast_names(movie["cast"])
        elif tmdb_movie:
            movie = self._movie_from_tmdb(movie_id, tmdb_movie)
        if movie is None:
            return None

        if tmdb_movie:
            movie["poster_url_large"] = tmdb.image_url(tmdb_movie.get("poster_path"), "w500")
        else:
            # TMDb failed or does not know the film; a poster cached earlier may still be there.
            movie["poster_url_large"] = self.get_movie_poster_url(movie_id, size="w500")
        videos = tmdb.record((tmdb_movie or {}).get("videos"))
        movie["trailer_key"] = tmdb.pick_trailer(videos.get("results"))
        _clean_detail_fields(movie)
        movie["runtime_formatted"] = format_runtime(movie.get("runtime"))
        if in_dataset:
            ranked = self.engine.rank_neighbours(index)
            movie["similar_movies"] = self.engine.format_results(
                self.engine.movies_df.iloc[ranked], DETAIL_SIMILAR_COUNT
            )
            self.add_posters(movie["similar_movies"])
        else:
            movie["similar_movies"] = self._similar_from_tmdb(movie_id)
        return movie

    def _movie_from_tmdb(self, movie_id: int, data: dict) -> dict | None:
        """The detail dictionary of a film outside the dataset; None without a title."""
        title = tmdb.text(data.get("title"))
        if not title:
            return None
        # tr-TR responses name genres in Turkish; the id gives the English value.
        genre_list = [
            TMDB_GENRE_NAMES.get(genre["id"], genre["name"])
            for genre in tmdb.records(data.get("genres"))
            if tmdb.valid_id(genre.get("id")) and tmdb.text(genre.get("name"))
        ]
        credits = tmdb.record(data.get("credits"))
        directors = [
            crew["name"]
            for crew in tmdb.records(credits.get("crew"))
            if crew.get("job") == "Director" and tmdb.text(crew.get("name"))
        ]
        actors = [
            actor["name"]
            for actor in tmdb.records(credits.get("cast"))
            if tmdb.text(actor.get("name"))
        ]
        return {
            "id": movie_id,
            "title": title,
            "overview": data.get("overview"),
            "tagline": data.get("tagline"),
            "genres": " ".join(sorted({genre for genre in genre_list if not is_excluded(genre)})),
            "genre_list": genre_list,
            "genre_badges": _genre_badges(genre_list),
            "year": tmdb.release_year(data),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "popularity": data.get("popularity"),
            "cast": ", ".join(actors[:CAST_LIMIT]),
            "cast_list": actors[:CAST_LIST_LIMIT],
            "director": directors[0] if directors else "",
            "directors": directors,
            # TMDb's details endpoint has no keywords; the field only exists for the template.
            "keywords": "",
            "runtime": data.get("runtime"),
        }

    def _similar_from_tmdb(self, movie_id: int) -> list[dict]:
        try:
            data = tmdb.get(f"/movie/{movie_id}/similar", language="tr-TR", page=1) or {}
        except tmdb.TmdbUnavailableError:
            return []
        similar = [
            {
                "id": item["id"],
                "title": item["title"],
                "poster_url": tmdb.image_url(item.get("poster_path"), "w342"),
                "year": tmdb.release_year(item),
            }
            for item in tmdb.records(data.get("results"))
            if tmdb.valid_id(item.get("id")) and tmdb.text(item.get("title"))
        ]
        return similar[:DETAIL_SIMILAR_COUNT]

    def get_list_card(
        self, movie_id: int | str, poster_paths: dict[int, str | None] | None = None
    ) -> dict | None:
        """Return the short card used on the favourites and watchlist page.

        poster_paths are prefetch_posters results; a dataset film missing there is looked up.
        """
        try:
            movie_id = int(movie_id)
        except (TypeError, ValueError):
            return None
        index = self.engine.row_of.get(movie_id)
        if index is not None:
            row = self.engine.movies_df.iloc[index]
            if movie_id in (poster_paths or {}):
                poster_url = tmdb.image_url(poster_paths[movie_id], "w342")
            else:
                poster_url = self.get_movie_poster_url(movie_id, size="w342")
            return {
                "id": movie_id,
                "title": row["title"],
                "year": int(row["year"]),
                "vote_average": round(float(row["vote_average"]), 1),
                "poster_url": poster_url,
            }
        data = self._fetch_movie(movie_id, language="tr-TR") or {}
        title = tmdb.text(data.get("title"))
        if not title:
            return None
        # The requested id, which the browser's list holds, even if TMDb answered with another.
        return {
            "id": movie_id,
            "title": title,
            "year": tmdb.release_year(data),
            "vote_average": round(tmdb.number(data.get("vote_average")), 1),
            "poster_url": tmdb.image_url(data.get("poster_path"), "w342"),
        }


def _genre_badges(genre_list) -> list[dict]:
    return [genre_badge(genre) for genre in genre_list if not is_excluded(genre)]


def _apply_turkish_texts(movie, tmdb_movie):
    # The dataset is English-only and the UI is Turkish, so TMDb's tr-TR texts win.
    for field in ("overview", "tagline"):
        if tmdb.text(tmdb_movie.get(field)):
            movie[field] = tmdb_movie[field]
    runtime = int(tmdb.number(tmdb_movie.get("runtime")))
    if runtime > 0:
        movie["runtime"] = runtime


def _clean_detail_fields(movie):
    """Coerce detail fields to the plain types the template expects; odd values count as unknown."""
    movie["year"] = int(tmdb.number(movie.get("year")))
    movie["vote_average"] = round(tmdb.number(movie.get("vote_average")), 1)
    movie["vote_count"] = int(tmdb.number(movie.get("vote_count")))
    movie["popularity"] = tmdb.number(movie.get("popularity"))
    for field in ("title", "overview", "genres", "keywords", "cast", "director", "tagline"):
        movie[field] = tmdb.text(movie.get(field))
    for field in ("genre_list", "directors", "cast_list"):
        movie[field] = list(movie.get(field) or [])


def cast_names(cast: str, limit: int = CAST_LIST_LIMIT) -> list[str]:
    """Return the first ``limit`` names of the dataset's comma-separated cast."""
    names = []
    for part in str(cast).split(","):
        part = part.strip()
        if part in _NAME_SUFFIXES and names:
            names[-1] += ", " + part
        elif part:
            names.append(part)
    return names[:limit]


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
