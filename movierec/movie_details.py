"""Movie detail pages and list cards, enriched with live TMDb data."""

import logging
from concurrent.futures import ThreadPoolExecutor, wait

from movierec import tmdb
from movierec.dataset import CAST_LIMIT
from movierec.engine import DETAIL_SIMILAR_COUNT, RecommendationEngine
from movierec.genres import TMDB_GENRE_NAMES, genre_badge, is_excluded
from movierec.i18n import TMDB_LANGUAGES, format_runtime

logger = logging.getLogger(__name__)

# Both caches are emptied once full. Only dataset films have their poster cached, so the
# whole catalogue fits in both languages.
POSTER_CACHE_LIMIT = 30000
# List cards of films outside the dataset, whose ids visitors choose.
LIST_CARD_CACHE_LIMIT = 2000
# Shared by all requests, so it caps TMDb lookups in flight; at 16, TMDb answered slower.
POSTER_WORKERS = 8
# Posters still missing after this show as placeholders; their lookups still fill the cache.
POSTER_WAIT_SECONDS = 5.0
CAST_LIST_LIMIT = 12
# Cast names are joined with ", ", which some names contain ("Sammy Davis, Jr.").
_NAME_SUFFIXES = {"Jr.", "Jr", "Sr.", "Sr"}
# Sentinel: None = no poster, _UNCACHED = not fetched yet.
_UNCACHED = object()


class MovieDetails:
    """Builds the detail dictionaries on top of an engine and remembers poster paths.

    Every method that asks TMDb takes the page language, "tr" or "en": TMDb's texts, posters
    and trailers differ per language.
    """

    def __init__(self, engine: RecommendationEngine):
        self.engine = engine
        # TMDb poster path (None when the film has none) per (movie id, language).
        self.poster_cache: dict[tuple[int, str], str | None] = {}
        # List card per (movie id, language) of films outside the dataset; None if TMDb has none.
        self.list_card_cache: dict[tuple[int, str], dict | None] = {}
        # Threads start on first use, so none exist before gunicorn forks its worker.
        self._poster_pool = ThreadPoolExecutor(
            max_workers=POSTER_WORKERS, thread_name_prefix="poster"
        )

    def add_posters(self, cards: list[dict], lang: str) -> None:
        """Set poster_url on the engine's result cards; None without a poster or TMDb."""
        poster_paths = self.prefetch_posters([card["id"] for card in cards], lang)
        for card in cards:
            card["poster_url"] = tmdb.image_url(poster_paths.get(card["id"]), "w342")

    def prefetch_posters(self, movie_ids: list[int], lang: str) -> dict[int, str | None]:
        """Return the poster path per movie id; uncached ones are fetched in parallel."""
        if not tmdb.is_configured():
            return {}
        # Fetched paths are returned directly: a concurrent lookup may empty the cache.
        poster_paths = {
            movie_id: self.poster_cache.get((movie_id, lang), _UNCACHED) for movie_id in movie_ids
        }
        futures = {}  # movie id -> its lookup in the poster pool
        for movie_id, path in poster_paths.items():
            if path is _UNCACHED:
                futures[movie_id] = self._poster_pool.submit(
                    self._fetch_poster_path, movie_id, lang
                )
        done, late = wait(futures.values(), timeout=POSTER_WAIT_SECONDS)
        for future in late:
            # Cancels only lookups not yet started.
            future.cancel()
        for movie_id, future in futures.items():
            poster_paths[movie_id] = future.result() if future in done else None
        return poster_paths

    def _fetch_poster_path(self, movie_id: int, lang: str) -> str | None:
        """Return the film's TMDb poster path, or None; never raises (a poster is decoration)."""
        try:
            data = self._fetch_movie(movie_id, lang) or {}
        except Exception:
            logger.exception("Poster lookup for movie %s failed", movie_id)
            return None
        return tmdb.text(data.get("poster_path")) or None

    def get_movie_poster_url(
        self, movie_id: int | str, lang: str, size: str = "w342"
    ) -> str | None:
        """Return the TMDb poster URL for a movie; a dataset film's path is cached per language."""
        if not tmdb.is_configured():
            return None
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            return None
        # A fetched path is used directly: a concurrent request may empty the cache.
        path = self.poster_cache.get((movie_id, lang), _UNCACHED)
        if path is _UNCACHED:
            path = self._fetch_poster_path(movie_id, lang)
        return tmdb.image_url(path, size)

    def _fetch_movie(
        self, movie_id: int, lang: str, *, strict: bool = False, **params
    ) -> dict | None:
        """GET /movie/{id} in lang and remember a dataset film's poster path; None if unavailable.

        strict raises TmdbUnavailableError instead when a configured TMDb fails.
        """
        try:
            data = tmdb.get(f"/movie/{movie_id}", language=TMDB_LANGUAGES[lang], **params)
        except tmdb.TmdbUnavailableError:
            if strict and tmdb.is_configured():
                raise
            return None
        # Cache only real answers, and only for dataset films, so random ids cannot evict them.
        if movie_id not in self.engine.row_of:
            return data
        cache_key = (movie_id, lang)
        is_new_entry = cache_key not in self.poster_cache
        if is_new_entry and len(self.poster_cache) >= POSTER_CACHE_LIMIT:
            self.poster_cache.clear()
        # data is None when TMDb does not know the film (404): cached as "no poster".
        poster_path = None
        if data is not None:
            poster_path = tmdb.text(data.get("poster_path")) or None
        self.poster_cache[cache_key] = poster_path
        return data

    def get_movie_details(self, movie_id: int | str, lang: str) -> dict | None:
        """Return the detail-page dictionary for a movie, from the dataset or TMDb.

        A film outside the dataset has no page without TMDb, so while a configured TMDb fails
        it raises TmdbUnavailableError instead of returning None like an unknown id.
        """
        try:
            movie_id = int(movie_id)
        except (ValueError, TypeError):
            logger.error("Invalid movie id: %r", movie_id)
            return None

        row = self.engine.row_of.get(movie_id)
        in_dataset = row is not None
        # One request per page: credits are only needed for movies outside the dataset.
        tmdb_movie = self._fetch_movie(
            movie_id,
            lang,
            strict=not in_dataset,
            append_to_response="videos" if in_dataset else "credits,videos",
            include_video_language=tmdb.VIDEO_LANGUAGES[lang],
        )
        if in_dataset:
            movie = self.engine.movies_df.iloc[row].to_dict()
            if tmdb_movie:
                _apply_tmdb_texts(movie, tmdb_movie)
            movie["genre_badges"] = _genre_badges(movie["genre_list"], lang)
            movie["cast_list"] = cast_names(movie["cast"])
        elif tmdb_movie:
            movie = self._movie_from_tmdb(movie_id, tmdb_movie, lang)
        else:
            # Outside the dataset and unknown to TMDb, or TMDb is not configured.
            return None
        if movie is None:  # TMDb's answer had no title
            return None

        if tmdb_movie:
            movie["poster_url_large"] = tmdb.image_url(tmdb_movie.get("poster_path"), "w500")
            videos = tmdb.record(tmdb_movie.get("videos"))
            movie["trailer_key"] = tmdb.pick_trailer(videos.get("results"), lang)
        else:
            # TMDb failed or does not know the film; a poster cached earlier may still be there.
            movie["poster_url_large"] = self.get_movie_poster_url(movie_id, lang, size="w500")
            movie["trailer_key"] = None
        _clean_detail_fields(movie)
        movie["runtime_formatted"] = format_runtime(movie.get("runtime"), lang)
        if in_dataset:
            ranked_rows = self.engine.rank_neighbours(row)
            neighbours = self.engine.movies_df.iloc[ranked_rows]
            movie["similar_movies"] = self.engine.format_results(neighbours, DETAIL_SIMILAR_COUNT)
            self.add_posters(movie["similar_movies"], lang)
        else:
            movie["similar_movies"] = self._fetch_similar_from_tmdb(movie_id, lang)
        return movie

    def _movie_from_tmdb(self, movie_id: int, data: dict, lang: str) -> dict | None:
        """Return the detail dictionary of a film outside the dataset; None without a title.

        data: TMDb's /movie/{id} answer in lang, with credits and videos.
        """
        title = tmdb.title(data)
        if not title:
            return None
        # tr-TR responses name genres in Turkish; the id gives the English value.
        genre_list = []
        for genre in tmdb.records(data.get("genres")):
            if not tmdb.valid_id(genre.get("id")) or not tmdb.text(genre.get("name")):
                continue
            genre_list.append(TMDB_GENRE_NAMES.get(genre["id"], genre["name"]))
        # The genres the page shows, each once, sorted and space-separated.
        shown_genres = {genre for genre in genre_list if not is_excluded(genre)}
        genres_text = " ".join(sorted(shown_genres))
        credits = tmdb.record(data.get("credits"))
        directors = [
            member["name"]
            for member in tmdb.records(credits.get("crew"))
            if member.get("job") == "Director" and tmdb.text(member.get("name"))
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
            "genres": genres_text,
            "genre_list": genre_list,
            "genre_badges": _genre_badges(genre_list, lang),
            "year": tmdb.release_year(data),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "popularity": data.get("popularity"),
            "cast": ", ".join(actors[:CAST_LIMIT]),
            "cast_list": actors[:CAST_LIST_LIMIT],
            "director": directors[0] if directors else "",
            "directors": directors,
            # TMDb's details endpoint has no keywords; "" keeps the keys of a dataset film.
            "keywords": "",
            "runtime": data.get("runtime"),
        }

    def _fetch_similar_from_tmdb(self, movie_id: int, lang: str) -> list[dict]:
        """Return up to DETAIL_SIMILAR_COUNT cards from TMDb's similar films; [] if it fails."""
        language = TMDB_LANGUAGES[lang]
        try:
            data = tmdb.get(f"/movie/{movie_id}/similar", language=language, page=1) or {}
        except tmdb.TmdbUnavailableError:
            return []
        similar = []
        for item in tmdb.records(data.get("results")):
            title = tmdb.title(item)
            if not tmdb.valid_id(item.get("id")) or not title:
                continue
            # A dataset film reads the same as on its own page and cards.
            if item["id"] in self.engine.row_of:
                title = self.engine.title_of(item["id"])
            similar.append(
                {
                    "id": item["id"],
                    "title": title,
                    "poster_url": tmdb.image_url(item.get("poster_path"), "w342"),
                    "year": tmdb.release_year(item),
                }
            )
            if len(similar) == DETAIL_SIMILAR_COUNT:
                break
        return similar

    def get_list_card(
        self, movie_id: int | str, lang: str, poster_paths: dict[int, str | None] | None = None
    ) -> dict | None:
        """Return the short card used on the favourites and watchlist page.

        poster_paths are prefetch_posters results; a dataset film missing there is looked up.
        """
        try:
            movie_id = int(movie_id)
        except (TypeError, ValueError):
            return None
        row = self.engine.row_of.get(movie_id)
        if row is not None:
            movie = self.engine.movies_df.iloc[row]
            if poster_paths is not None and movie_id in poster_paths:
                poster_url = tmdb.image_url(poster_paths[movie_id], "w342")
            else:
                poster_url = self.get_movie_poster_url(movie_id, lang, size="w342")
            return {
                "id": movie_id,
                "title": movie["title"],
                "year": int(movie["year"]),
                "vote_average": round(float(movie["vote_average"]), 1),
                "poster_url": poster_url,
            }
        if not tmdb.is_configured():
            return None
        cache_key = (movie_id, lang)
        card = self.list_card_cache.get(cache_key, _UNCACHED)
        if card is not _UNCACHED:
            return card
        try:
            data = self._fetch_movie(movie_id, lang, strict=True)
        except tmdb.TmdbUnavailableError:
            # Not remembered, so the film shows up once TMDb answers again.
            return None
        card = _list_card(movie_id, data) if data is not None else None
        if len(self.list_card_cache) >= LIST_CARD_CACHE_LIMIT:
            self.list_card_cache.clear()
        self.list_card_cache[cache_key] = card
        return card


def _list_card(movie_id: int, data: dict) -> dict | None:
    """Return the list card of a film outside the dataset; None without a title.

    data: TMDb's /movie/{id} answer.
    """
    title = tmdb.title(data)
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


def _genre_badges(genre_list: list[str], lang: str) -> list[dict]:
    """Return the badge of each genre the page shows, in the film's genre order."""
    return [genre_badge(genre, lang) for genre in genre_list if not is_excluded(genre)]


def _apply_tmdb_texts(movie: dict, tmdb_movie: dict) -> None:
    """Replace the overview, tagline and runtime with TMDb's values where it has them.

    TMDb answers in the page language; where it has no text in that language, the dataset's
    English text stays.
    """
    for field in ("overview", "tagline"):
        if tmdb.text(tmdb_movie.get(field)):
            movie[field] = tmdb_movie[field]
    runtime = int(tmdb.number(tmdb_movie.get("runtime")))
    if runtime > 0:
        movie["runtime"] = runtime


def _clean_detail_fields(movie: dict) -> None:
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
