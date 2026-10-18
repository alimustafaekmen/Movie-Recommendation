"""Recommendation pages, movie and person details, search and the JSON suggestion endpoints."""

import datetime
import logging

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, url_for

from movierec.config import DEFAULT_RECOMMENDATION_COUNT, MAX_RECOMMENDATION_COUNT
from movierec.genres import genre_label
from movierec.people import find_person
from movierec.tmdb import TmdbUnavailable
from movierec.web import MOVIE_ID, get_details, get_engine
from movierec.web.redirects import redirect_back

logger = logging.getLogger(__name__)

bp = Blueprint("movies", __name__)

# Flash messages live in the session cookie, which browsers drop beyond about 4 KB.
_MAX_ECHOED_INPUT = 100
# The suggestion lists only change when the dataset does, i.e. on a redeploy.
_SUGGESTIONS_MAX_AGE = 86400


def _parse_num_recs(form):
    """Return the requested result count clamped to [1, MAX]; raise ValueError if not an int."""
    num = int(form.get("num_recommendations", DEFAULT_RECOMMENDATION_COUNT))
    return max(1, min(num, MAX_RECOMMENDATION_COUNT))


def _recommend(method, *args, **kwargs):
    """Call an engine recommend_* method and add posters; on failure return a generic warning."""
    try:
        suggestions, warnings = method(*args, **kwargs)
    except Exception:
        logger.exception("Recommendation request failed")
        return None, ["Öneriler sayfaya yüklenirken bir hata oluştu."]
    get_details().add_posters(suggestions)
    return suggestions, warnings


def _similarity_request(form, num):
    """Handle the 'similar movies' form; return (suggestions, warnings, search_term)."""
    field = "film_ismi_custom" if form.get("toggle_film") == "on" else "film_ismi"
    query = form.get(field)
    search_term = query.strip() if query else None
    if not search_term:
        return None, ["Lütfen bir film ismi girin veya listeden seçin."], search_term
    suggestions, warnings = _recommend(get_engine().recommend_by_similarity, search_term, num=num)
    return suggestions, warnings, search_term


def _features_request(form, num, genres):
    """Handle the filter form; return (suggestions, warnings, search_term)."""
    field = "yonetmen_custom" if form.get("toggle_yonetmen") == "on" else "yonetmen"
    director = form.get(field)
    director = director.strip() if director else None
    min_rating = form.get("min_puan", "")
    min_year = form.get("min_yil", "")
    max_year = form.get("max_yil", "")
    search_term = _features_title(genres, director, min_year, max_year, min_rating)
    suggestions, warnings = _recommend(
        get_engine().recommend_by_features,
        genres=genres,
        director_query=director,
        min_rating=min_rating,
        min_year=min_year,
        max_year=max_year,
        num=num,
    )
    return suggestions, warnings, search_term


def _features_title(genres, director, min_year, max_year, min_rating):
    """Describe the active filters for the results heading."""
    parts = []
    if genres:
        parts.append("Türler: " + ", ".join(genre_label(genre) for genre in genres))
    if director:
        parts.append(f"Yönetmen: {director}")
    years = [year for year in (min_year, max_year) if year]
    if years:
        parts.append("Yıl: " + " - ".join(years))
    if min_rating:
        parts.append(f"Min. Puan: {min_rating}")
    return " | ".join(parts) or "Seçilen Filtreler"


@bp.route("/", methods=["GET", "POST"])
def home():
    """Home page: popular movies, ?genre= listing and both recommendation forms."""
    engine = get_engine()
    if engine is None:
        message = "Film öneri motoru kullanılamıyor. Lütfen daha sonra tekrar deneyin."
        return render_template("error.html", code=503, message=message), 503

    suggestions = None
    search_term = None
    warnings = []
    recommendation_type = None
    form_data = {}
    selected_genres = []

    genre = request.args.get("genre")
    if request.method == "GET" and genre:
        selected_genres = [genre]
        form_data["tur"] = selected_genres
        recommendation_type = "features"
        suggestions, warnings = engine.recommend_by_features(
            genres=selected_genres, num=DEFAULT_RECOMMENDATION_COUNT
        )
        get_details().add_posters(suggestions)
        search_term = f"Tür: {genre_label(selected_genres[0])}"

    if request.method == "POST":
        form_data = request.form.to_dict()
        selected_genres = request.form.getlist("tur")
        form_data["tur"] = selected_genres
        try:
            num = _parse_num_recs(request.form)
        except ValueError:
            warnings.append("Lütfen sayısal alanlara sadece geçerli sayılar girin.")
        else:
            if "submit_similarity" in request.form:
                recommendation_type = "similarity"
                suggestions, warnings, search_term = _similarity_request(request.form, num)
            elif "submit_features" in request.form:
                recommendation_type = "features"
                suggestions, warnings, search_term = _features_request(
                    request.form, num, selected_genres
                )

    return render_template(
        "index.html",
        suggestions=suggestions,
        search_term=search_term,
        warnings=warnings,
        recommendation_type=recommendation_type,
        popular_movies=engine.popular_movies_list,
        popular_directors=engine.popular_directors_list,
        all_genres=engine.all_genres,
        form_data=form_data,
        selected_genres=selected_genres,
        max_recommendations=MAX_RECOMMENDATION_COUNT,
        default_recommendations=DEFAULT_RECOMMENDATION_COUNT,
    )


@bp.route(f"/movie/<{MOVIE_ID}:movie_id>")
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
    """Pretty URL for the home page's ?genre= listing."""
    # Older links spelled spaces as "+" (/genre/Science+Fiction); no genre name has a "+".
    return redirect(url_for("movies.home", genre=genre_name.replace("+", " ")))


def _person_page(name, person_type, label):
    try:
        person = find_person(name, person_type)
    except TmdbUnavailable:
        flash(
            "Kişi bilgileri şu anda TMDb'den alınamıyor. Lütfen daha sonra tekrar deneyin.",
            "warning",
        )
        return redirect(url_for("movies.home"))
    if person is None:
        flash(f"'{name[:_MAX_ECHOED_INPUT]}' adlı {label} bulunamadı.", "warning")
        return redirect(url_for("movies.home"))
    return render_template("person_detail.html", person=person)


@bp.route("/director/<path:director_name_url>")
def director(director_name_url):
    """Director page with biography and filmography from TMDb."""
    return _person_page(director_name_url, "director", "yönetmen")


@bp.route("/actor/<path:actor_name_url>")
def actor(actor_name_url):
    """Actor page with biography and filmography from TMDb."""
    return _person_page(actor_name_url, "actor", "oyuncu")


@bp.route("/search_movie_redirect", methods=["POST"])
def search():
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
    """Every movie title, for the home page's free-text film suggestions."""
    engine = get_engine()
    if engine is None:
        abort(503)
    return _suggestion_list(engine.all_movie_titles)


@bp.route("/api/directors")
def api_directors():
    """Every director name, for the home page's free-text director suggestions."""
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
