"""Recommendation pages, movie and person details, search and the JSON suggestion endpoints."""

import datetime
import logging
from urllib.parse import urlencode

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, url_for

from movierec.config import DEFAULT_RECOMMENDATION_COUNT
from movierec.engine import clamp_count
from movierec.genres import genre_label
from movierec.people import find_person
from movierec.search_filters import YEARS_SWAPPED
from movierec.tmdb import TmdbUnavailableError
from movierec.web import MOVIE_ID_CONVERTER, get_details, get_engine
from movierec.web.redirects import redirect_back

logger = logging.getLogger(__name__)

bp = Blueprint("movies", __name__)

# Text the page shows back: the search fields, whose inputs have maxlength="100" (the engine
# reads no further), and flash messages, which live in the session cookie that browsers drop
# beyond about 4 KB.
_MAX_ECHOED_INPUT = 100
# The suggestion lists only change when the dataset does, i.e. on a redeploy.
_SUGGESTIONS_MAX_AGE = 86400
# The filter search: ?tur= may repeat, the other fields are single values. ?film= is the
# similarity search.
_FILTER_SCALARS = ("yonetmen", "min_yil", "max_yil", "min_puan")
_FILTER_FIELDS = ("tur", *_FILTER_SCALARS)
# "Daha fazla göster" adds one more grid of results; a longer ?n= still works by URL.
_MORE_RESULTS_LIMIT = 2 * DEFAULT_RECOMMENDATION_COUNT


def _recommend(method, *args, **kwargs):
    """Call an engine recommend_* method and add posters; on failure return a generic warning."""
    try:
        suggestions, warnings = method(*args, **kwargs)
    except Exception:
        logger.exception("Recommendation request failed")
        return [], ["Öneriler sayfaya yüklenirken bir hata oluştu."]
    get_details().add_posters(suggestions)
    return suggestions, warnings


def _filter_summary(genres, filters):
    """Describe the active filters under the results heading."""
    parts = []
    if genres:
        label = "Tür" if len(genres) == 1 else "Türler"
        parts.append(f"{label}: " + ", ".join(genre_label(genre) for genre in genres))
    if filters["yonetmen"]:
        parts.append(f"Yönetmen: {filters['yonetmen']}")
    years = [year for year in (filters["min_yil"], filters["max_yil"]) if year]
    if years:
        parts.append("Yıl: " + " - ".join(years))
    if filters["min_puan"]:
        parts.append("Min. puan: " + filters["min_puan"].replace(".", ","))
    return " · ".join(parts)


def _field(args, name):
    """A text field of the search, stripped and cut like its input."""
    return args.get(name, "").strip()[:_MAX_ECHOED_INPUT]


def _similar_search(engine, args, num):
    film = _field(args, "film")
    suggestions, warnings = _recommend(engine.recommend_by_similarity, film, num=num)
    # Name the film the results are for: a misspelt or lower-case query shows its match.
    movie_id = engine.find_movie_id(film) if suggestions else None
    if movie_id is not None:
        film = engine.title_of(movie_id)
    return {"suggestions": suggestions, "warnings": warnings, "form": {"film": film}}


def _filter_search(engine, args, num):
    genres = list(dict.fromkeys(genre for genre in args.getlist("tur") if genre))
    filters = {field: _field(args, field) for field in _FILTER_SCALARS}
    suggestions, warnings = _recommend(
        engine.recommend_by_features,
        genres=genres,
        director_query=filters["yonetmen"],
        min_rating=filters["min_puan"],
        min_year=filters["min_yil"],
        max_year=filters["max_yil"],
        num=num,
    )
    # The engine searched a reversed year range in order; the page shows it that way too.
    if YEARS_SWAPPED in warnings:
        filters["min_yil"], filters["max_yil"] = filters["max_yil"], filters["min_yil"]
    return {
        "suggestions": suggestions,
        "warnings": warnings,
        "summary": _filter_summary(genres, filters),
        "form": {"tur": genres, **filters},
    }


def _search(engine, args, num):
    """Run the search named by the query string; returns the page's search context.

    form holds the search as the page shows it, which also makes the "more" link.
    """
    search = {"mode": None, "suggestions": [], "warnings": [], "summary": "", "form": {}}
    if "film" in args:
        search.update(_similar_search(engine, args, num), mode="similar")
    elif any(field in args for field in _FILTER_FIELDS):
        search.update(_filter_search(engine, args, num), mode="filter")
    return search


def _more_url(form, num, shown):
    """Link to the same search with more results, or None when every result is shown."""
    if shown < num or num >= _MORE_RESULTS_LIMIT:
        return None
    # Rankings are prefix-consistent, so the old results stay put and the link can jump
    # straight to the first new card.
    query = {key: value for key, value in form.items() if value}
    more = min(num + DEFAULT_RECOMMENDATION_COUNT, _MORE_RESULTS_LIMIT)
    return url_for("movies.home", **query, n=more, _anchor=f"sonuc-{num + 1}")


def _legacy_genre_redirect(args):
    """Older pages linked genres as /?genre=<name>; the same search is now ?tur=<name>."""
    pairs = [
        ("tur" if key == "genre" else key, value)
        for key, value in args.items(multi=True)
        if value or key != "genre"
    ]
    return redirect(f"{url_for('movies.home')}?{urlencode(pairs)}", 301)


@bp.route("/")
def home():
    """Home page: popular movies, or the results of a ?film= or filter search."""
    engine = get_engine()
    if engine is None:
        message = "Öneri motoru şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."
        return render_template("error.html", code=503, message=message), 503

    args = request.args
    if any(args.getlist("genre")):
        return _legacy_genre_redirect(args)

    num = clamp_count(args.get("n"))
    search = _search(engine, args, num)
    popular = []
    if search["mode"] is None:
        popular = engine.popular_cards()
        # Posters come from TMDb; during its cooldown the grid shows placeholders instead.
        get_details().add_posters(popular)
    return render_template(
        "home.html",
        **search,
        more_url=_more_url(search["form"], num, len(search["suggestions"])),
        popular=popular,
        all_genres=engine.all_genres,
    )


@bp.route(f"/movie/<{MOVIE_ID_CONVERTER}:movie_id>")
def movie_detail(movie_id):
    """Movie detail page; movie_detail.js sets the like/watchlist buttons from the browser."""
    if get_engine() is None:
        abort(503)

    try:
        movie_data = get_details().get_movie_details(movie_id)
    except Exception:
        logger.exception("Movie details failed for id %s", movie_id)
        message = "Film detayı şu anda alınamıyor."
        return render_template("error.html", code=500, message=message), 500
    if movie_data is None:
        abort(404)
    return render_template("movie_detail.html", movie=movie_data)


@bp.route("/genre/<path:genre_name>")
def genre(genre_name):
    """Pretty URL for a one-genre filter search."""
    # Older links spelled spaces as "+" (/genre/Science+Fiction); no genre name has a "+".
    return redirect(url_for("movies.home", tur=genre_name.replace("+", " ")), 301)


def _person_page(name, person_type, label):
    try:
        person = find_person(name, person_type)
    except TmdbUnavailableError:
        flash(
            "Kişi bilgileri şu anda TMDb'den alınamıyor. Lütfen daha sonra tekrar deneyin.",
            "warning",
        )
        return redirect(url_for("movies.home"))
    if person is None:
        flash(f"'{name[:_MAX_ECHOED_INPUT]}' adlı {label} bulunamadı.", "warning")
        return redirect(url_for("movies.home"))
    return render_template("person_detail.html", person=person)


@bp.route("/director/<path:director_name>")
def director(director_name):
    """Director page with biography and filmography from TMDb."""
    return _person_page(director_name, "director", "yönetmen")


@bp.route("/actor/<path:actor_name>")
def actor(actor_name):
    """Actor page with biography and filmography from TMDb."""
    return _person_page(actor_name, "actor", "oyuncu")


@bp.route("/search_movie_redirect", methods=["POST"])
def header_search():
    """Header search: open the closest-matching movie, or go back with a message."""
    engine = get_engine()
    if engine is None:
        flash("Öneri motoru kullanılamıyor.", "danger")
        return redirect(url_for("movies.home"))

    query = request.form.get("search_query", "").strip()
    if not query:
        flash("Lütfen bir film adı girin.", "warning")
        return redirect_back()

    movie_id = engine.find_movie_id(query)
    if movie_id is None:
        flash(f"'{query[:_MAX_ECHOED_INPUT]}' ismine benzer hiçbir film bulamadık.", "warning")
        return redirect_back()
    return redirect(url_for("movies.movie_detail", movie_id=movie_id))


def _suggestion_list(items):
    response = jsonify(items)
    response.cache_control.public = True
    response.cache_control.max_age = _SUGGESTIONS_MAX_AGE
    response.add_etag()
    return response.make_conditional(request)


@bp.route("/api/titles")
def api_titles():
    """Every movie title, for the film and header search suggestions."""
    engine = get_engine()
    if engine is None:
        abort(503)
    return _suggestion_list(engine.all_movie_titles)


@bp.route("/api/directors")
def api_directors():
    """Every director name, for the director filter's suggestions."""
    engine = get_engine()
    if engine is None:
        abort(503)
    return _suggestion_list(engine.all_director_names)


@bp.route("/health")
def health():
    """Liveness probe; 503 while the recommendation engine is unavailable."""
    engine = get_engine()
    ready = engine is not None
    payload = jsonify(
        status="ok" if ready else "degraded",
        engine_loaded=ready,
        movies=len(engine.movies_df) if ready else 0,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat(),
    )
    return payload, 200 if ready else 503
