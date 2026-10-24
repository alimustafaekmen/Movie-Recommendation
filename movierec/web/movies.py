"""Recommendation pages, movie and person details, search and the JSON suggestion endpoints."""

import datetime
import logging
from urllib.parse import urlencode

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, url_for

from movierec.config import DEFAULT_RECOMMENDATION_COUNT
from movierec.engine import RecommendationEngine, clamp_count
from movierec.genres import genre_label
from movierec.i18n import decimal_text, message
from movierec.people import find_person
from movierec.search_filters import YEARS_SWAPPED_WARNING
from movierec.tmdb import TmdbUnavailableError
from movierec.web import MOVIE_ID_CONVERTER, get_details, get_engine
from movierec.web.language import get_lang, t, t_count, t_message
from movierec.web.redirects import redirect_back

logger = logging.getLogger(__name__)

bp = Blueprint("movies", __name__)

# Text the page shows back: the search fields, whose inputs have maxlength="100" (the engine
# reads no further), and flash messages, which live in the session cookie that browsers drop
# beyond about 4 KB.
_MAX_ECHOED_LENGTH = 100
# The suggestion lists only change when the dataset does, i.e. on a redeploy.
_SUGGESTIONS_MAX_AGE_SECONDS = 24 * 60 * 60
# The filter search: ?tur= may repeat, the other fields are single values. ?film= is the
# similarity search.
_SINGLE_VALUE_FIELDS = ("yonetmen", "min_yil", "max_yil", "min_puan")
_FILTER_FIELDS = ("tur", *_SINGLE_VALUE_FIELDS)
# The "Show more" link adds one more grid of results; a longer ?n= still works by URL.
_MORE_RESULTS_LIMIT = 2 * DEFAULT_RECOMMENDATION_COUNT
# Shown instead of results when the engine fails; the error itself only goes to the log.
_SEARCH_FAILED_WARNING = message("warning.search_failed")


def _filter_summary(genres: list[str], filters: dict[str, str]) -> str:
    """Return the description of the active filters shown under the results heading."""
    lang = get_lang()
    parts = []
    if genres:
        genre_names = ", ".join(genre_label(genre, lang) for genre in genres)
        parts.append(t_count("summary.genres", len(genres), genres=genre_names))
    if filters["yonetmen"]:
        parts.append(t("summary.director", name=filters["yonetmen"]))
    min_year, max_year = filters["min_yil"], filters["max_yil"]
    if min_year and max_year:
        parts.append(t("summary.years", years=f"{min_year} - {max_year}"))
    elif min_year:
        parts.append(t("summary.years_from", year=min_year))
    elif max_year:
        parts.append(t("summary.years_to", year=max_year))
    if filters["min_puan"]:
        parts.append(t("summary.min_rating", rating=decimal_text(filters["min_puan"], lang)))
    return " · ".join(parts)


def _read_field(args, name: str) -> str:
    """Return a text field of the search, stripped and cut like its input."""
    return args.get(name, "").strip()[:_MAX_ECHOED_LENGTH]


def _similar_search(engine: RecommendationEngine, args, num: int) -> dict:
    """Return the search context of a ?film= similarity search.

    args: request.args; num: how many results to return.
    """
    film = _read_field(args, "film")
    try:
        suggestions, warnings = engine.recommend_by_similarity(film, num=num)
    except Exception:
        logger.exception("Recommendation request failed")
        suggestions = []
        warnings = [_SEARCH_FAILED_WARNING]
    get_details().add_posters(suggestions, get_lang())
    # Name the film the results are for: a misspelt or lower-case query shows its match.
    if suggestions:
        movie_id = engine.find_movie_id(film)
        if movie_id is not None:
            film = engine.title_of(movie_id)
    return {"suggestions": suggestions, "warnings": warnings, "form": {"film": film}}


def _filter_search(engine: RecommendationEngine, args, num: int) -> dict:
    """Return the search context of a filter search (genres, director, years, rating).

    args: request.args; num: how many results to return.
    """
    # ?tur= may repeat; keep each genre once, in the order given.
    genres = []
    for genre in args.getlist("tur"):
        if genre and genre not in genres:
            genres.append(genre)
    filters = {field: _read_field(args, field) for field in _SINGLE_VALUE_FIELDS}
    try:
        suggestions, warnings = engine.recommend_by_features(
            genres=genres,
            director_query=filters["yonetmen"],
            min_rating=filters["min_puan"],
            min_year=filters["min_yil"],
            max_year=filters["max_yil"],
            num=num,
        )
    except Exception:
        logger.exception("Recommendation request failed")
        suggestions = []
        warnings = [_SEARCH_FAILED_WARNING]
    get_details().add_posters(suggestions, get_lang())
    # The engine searched a reversed year range in order; the page shows it that way too.
    if YEARS_SWAPPED_WARNING in warnings:
        filters["min_yil"], filters["max_yil"] = filters["max_yil"], filters["min_yil"]
    # tur first: the "more" link keeps /?tur=…&n=24
    form = {"tur": genres}
    form.update(filters)
    return {
        "suggestions": suggestions,
        "warnings": warnings,
        "summary": _filter_summary(genres, filters),
        "form": form,
    }


def _search(engine: RecommendationEngine, args, num: int) -> dict:
    """Run the search named by the query string; returns the page's search context.

    args: request.args. form holds the search as the page shows it, which also makes the
    "more" link.
    """
    search = {"mode": None, "suggestions": [], "warnings": [], "summary": "", "form": {}}
    if "film" in args:
        search.update(_similar_search(engine, args, num))
        search["mode"] = "similar"
    elif any(field in args for field in _FILTER_FIELDS):
        search.update(_filter_search(engine, args, num))
        search["mode"] = "filter"
    return search


def _more_url(form: dict, num: int, has_more: bool) -> str | None:
    """Return the link to the same search with more results, or None.

    form: the search as the page shows it; num: the results shown now (?n=); has_more:
    whether the engine found a result past num.
    """
    # Every result is already on the page.
    if not has_more:
        return None
    # The link adds at most one grid past the first; a longer ?n= still works by URL.
    if num >= _MORE_RESULTS_LIMIT:
        return None
    # Rankings are prefix-consistent, so the old results stay put and the link can jump
    # straight to the first new card.
    query = {key: value for key, value in form.items() if value}
    next_num = min(num + DEFAULT_RECOMMENDATION_COUNT, _MORE_RESULTS_LIMIT)
    first_new_card = f"sonuc-{num + 1}"
    return url_for("movies.home", **query, n=next_num, _anchor=first_new_card)


def _legacy_genre_redirect(args):
    """Older pages linked genres as /?genre=<name>; the same search is now ?tur=<name>."""
    pairs = []
    for key, value in args.items(multi=True):
        if key == "genre":
            # An empty ?genre= named no genre; leave it out.
            if not value:
                continue
            key = "tur"
        pairs.append((key, value))
    home_url = url_for("movies.home")
    return redirect(f"{home_url}?{urlencode(pairs)}", 301)


@bp.route("/")
def home():
    """Home page: popular movies, or the results of a ?film= or filter search."""
    engine = get_engine()
    if engine is None:
        text = t("error.engine_unavailable")
        return render_template("error.html", code=503, message=text), 503

    args = request.args
    if any(args.getlist("genre")):
        return _legacy_genre_redirect(args)

    num = clamp_count(args.get("n"))
    # One result past num tells whether a "more" link would show anything new.
    search = _search(engine, args, num + 1)
    has_more = len(search["suggestions"]) > num
    search["suggestions"] = search["suggestions"][:num]
    popular = []
    if search["mode"] is None:
        popular = engine.popular_cards()
        # Posters come from TMDb; during its cooldown the grid shows placeholders instead.
        get_details().add_posters(popular, get_lang())
    # The engine returns catalogue keys; the page shows them in its language.
    search["warnings"] = [t_message(warning) for warning in search["warnings"]]
    return render_template(
        "home.html",
        **search,
        more_url=_more_url(search["form"], num, has_more),
        popular=popular,
        all_genres=engine.all_genres,
    )


@bp.route(f"/movie/<{MOVIE_ID_CONVERTER}:movie_id>")
def movie_detail(movie_id: int):
    """Movie detail page; movie_detail.js sets the like/watchlist buttons from the browser."""
    if get_engine() is None:
        abort(503)

    try:
        movie_data = get_details().get_movie_details(movie_id, get_lang())
    except TmdbUnavailableError:
        text = t("error.movie_unavailable")
        return render_template("error.html", code=503, message=text), 503
    except Exception:
        logger.exception("Movie details failed for id %s", movie_id)
        text = t("error.movie_unavailable")
        return render_template("error.html", code=500, message=text), 500
    if movie_data is None:
        abort(404)
    return render_template("movie_detail.html", movie=movie_data)


@bp.route("/genre/<path:genre_name>")
def genre(genre_name: str):
    """Pretty URL for a one-genre filter search."""
    # Older links spelled spaces as "+" (/genre/Science+Fiction); no genre name has a "+".
    return redirect(url_for("movies.home", tur=genre_name.replace("+", " ")), 301)


def _render_person_page(name: str, person_type: str, not_found_key: str):
    """Render the person page, or go home with a message when TMDb fails or knows nobody.

    not_found_key: the catalogue key of the message for a name TMDb does not know.
    """
    # Flashes are written in the request's language; the page that shows them is the next one.
    try:
        person = find_person(name, person_type, get_lang())
    except TmdbUnavailableError:
        flash(t("flash.person_unavailable"), "warning")
        return redirect(url_for("movies.home"))
    if person is None:
        flash(t(not_found_key, name=name[:_MAX_ECHOED_LENGTH]), "warning")
        return redirect(url_for("movies.home"))
    # A copy: find_person remembers the person it returns.
    person = {**person, "movies": _with_dataset_titles(person["movies"])}
    return render_template("person_detail.html", person=person, person_type=person_type)


def _with_dataset_titles(movies: list[dict]) -> list[dict]:
    """Return filmography cards with the dataset title for dataset films.

    A film then reads the same on a person page as on its own page and cards. Without the
    engine the cards keep TMDb's titles.
    """
    engine = get_engine()
    if engine is None:
        return movies
    cards = []
    for movie in movies:
        if movie["id"] in engine.row_of:
            movie = {**movie, "title": engine.title_of(movie["id"])}
        cards.append(movie)
    return cards


@bp.route("/director/<path:director_name>")
def director(director_name: str):
    """Director page with biography and filmography from TMDb."""
    return _render_person_page(director_name, "director", "person.not_found_director")


@bp.route("/actor/<path:actor_name>")
def actor(actor_name: str):
    """Actor page with biography and filmography from TMDb."""
    return _render_person_page(actor_name, "actor", "person.not_found_actor")


@bp.route("/search_movie_redirect", methods=["POST"])
def header_search():
    """Header search: open the closest-matching movie, or go back with a message."""
    engine = get_engine()
    if engine is None:
        flash(t("flash.engine_unavailable"), "danger")
        return redirect(url_for("movies.home"))

    query = request.form.get("search_query", "").strip()
    if not query:
        flash(t("warning.empty_query"), "warning")
        return redirect_back()

    movie_id = engine.find_movie_id(query)
    if movie_id is None:
        flash(t("flash.movie_not_found", query=query[:_MAX_ECHOED_LENGTH]), "warning")
        return redirect_back()
    return redirect(url_for("movies.movie_detail", movie_id=movie_id))


def _cacheable_json_response(items: list):
    """Return items as JSON browsers may keep for a day; an unchanged list answers 304."""
    response = jsonify(items)
    response.cache_control.public = True
    response.cache_control.max_age = _SUGGESTIONS_MAX_AGE_SECONDS
    response.add_etag()
    return response.make_conditional(request)


@bp.route("/api/titles")
def api_titles():
    """Every movie title, for the film and header search suggestions.

    A title whose film has a Turkish title comes as [title, Turkish title], so base.js can
    show why typing the Turkish title suggests it.
    """
    engine = get_engine()
    if engine is None:
        abort(503)
    return _cacheable_json_response(engine.title_suggestions)


@bp.route("/api/directors")
def api_directors():
    """Every director name, for the director filter's suggestions."""
    engine = get_engine()
    if engine is None:
        abort(503)
    return _cacheable_json_response(engine.all_director_names)


@bp.route("/health")
def health():
    """Liveness probe; 503 while the recommendation engine is unavailable."""
    engine = get_engine()
    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    if engine is None:
        return jsonify(status="degraded", engine_loaded=False, movies=0, timestamp=timestamp), 503
    return jsonify(
        status="ok", engine_loaded=True, movies=len(engine.movies_df), timestamp=timestamp
    ), 200
