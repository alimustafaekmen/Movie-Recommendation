"""Flask app factory: configuration, security, error pages and template helpers."""

import datetime
import hashlib
import logging
from pathlib import Path
from xml.etree import ElementTree

from flask import Flask, current_app, jsonify, render_template, request, url_for

from movierec.config import (
    BASE_DIR,
    CREDITS_CSV,
    DATASET_DIR,
    FLASK_SECRET_KEY,
    LOG_LEVEL,
    MOVIES_CSV,
    SECRET_KEY_IS_EPHEMERAL,
    SESSION_COOKIE_SECURE,
    TMDB_API_KEY,
)
from movierec.engine import RecommendationEngine
from movierec.genres import genre_label
from movierec.i18n import format_date, format_number, format_rating
from movierec.movie_details import MovieDetails
from movierec.web.language import get_lang, lang_url, remember_language, t
from movierec.web.redirects import API_PREFIXES

logger = logging.getLogger(__name__)

# HTTP status -> (error name for JSON answers, Turkish message for the error page).
_ERRORS = {
    400: ("bad_request", "İstek işlenemedi. Lütfen tekrar deneyin."),
    404: ("not_found", "Aradığınız sayfa bulunamadı."),
    405: ("method_not_allowed", "Bu adres bu şekilde kullanılamaz."),
    413: ("payload_too_large", "Gönderilen istek çok büyük."),
    500: ("server_error", "Beklenmedik bir hata oluştu. Lütfen daha sonra tekrar deneyin."),
    503: ("unavailable", "Servis şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."),
}
# The only request body is the header search form. Werkzeug reads a urlencoded body whole,
# so without a limit one request could fill the memory.
MAX_REQUEST_BODY_BYTES = 16 * 1024
# How long browsers may cache versioned static files and remember HSTS.
ONE_YEAR_SECONDS = 365 * 24 * 60 * 60
# Pages load scripts and styles only from this site (no inline code), the fonts from Google,
# images from this site, TMDb and YouTube, and the trailer player from youtube-nocookie.com.
CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' https://fonts.googleapis.com",
        "font-src https://fonts.gstatic.com",
        "img-src 'self' https://image.tmdb.org https://i.ytimg.com https://www.themoviedb.org data:",
        "frame-src https://www.youtube-nocookie.com",
        "connect-src 'self'",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    )
)
# Features the site never uses, so neither an embed nor injected code can ask for them.
PERMISSIONS_POLICY = "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
# TMDb ids are positive 32-bit integers; anything else is a 404 before it reaches the engine.
MAX_MOVIE_ID = 2147483647
MOVIE_ID_CONVERTER = f"int(min=1, max={MAX_MOVIE_ID})"


class _App(Flask):
    """Flask whose static files are cached for long only under a versioned URL."""

    def get_send_file_max_age(self, filename: str | None) -> int | None:
        # Only asset_url() links carry ?v=. Module imports such as "./list_store.js" cannot, so
        # those are revalidated on every use; a long cache would mix old and new files.
        if "v" not in request.args:
            return 0
        return super().get_send_file_max_age(filename)


def get_engine() -> RecommendationEngine | None:
    """Return the recommendation engine, or None when it failed to start."""
    return current_app.extensions["engine"]


def get_details() -> MovieDetails | None:
    """Return the MovieDetails helper for the engine, or None when the engine failed to start."""
    return current_app.extensions["details"]


def build_asset_versions(folder: str | Path) -> dict[str, str]:
    """Map each file under folder to a short hash of its content.

    Keys are POSIX paths relative to folder, e.g. "js/base.js".
    """
    root = Path(folder)
    versions = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(root).as_posix()
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        versions[relative_path] = content_hash[:12]
    return versions


def create_app() -> Flask:
    """Build the Flask app: the engine is loaded here, once per process.

    A broken setup (no dataset, no TMDb key, a bad LOG_LEVEL) is logged in one line each;
    the app still starts and answers 503 where it needs what is missing.
    """
    # logging.basicConfig raises on an unknown level name, which would stop every worker.
    is_known_level = LOG_LEVEL in logging.getLevelNamesMapping()
    level = LOG_LEVEL if is_known_level else "INFO"
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    if not is_known_level:
        logger.warning("LOG_LEVEL %r is not a logging level; using INFO", LOG_LEVEL)
    # templates/ and static/ stay at the repo root, outside the package.
    app = _App(
        __name__,
        template_folder=str(BASE_DIR / "templates"),
        static_folder=str(BASE_DIR / "static"),
    )
    # Signs the session cookie, which only carries flash messages.
    app.secret_key = FLASK_SECRET_KEY
    app.config.update(
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=SESSION_COOKIE_SECURE,
        MAX_CONTENT_LENGTH=MAX_REQUEST_BODY_BYTES,
    )
    # Templates link static files through asset_url(), whose ?v= changes with the file
    # content, so browsers may keep them for a year. Not in debug mode, so edits show on reload.
    if not app.debug:
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = ONE_YEAR_SECONDS
    if SECRET_KEY_IS_EPHEMERAL:
        logger.warning(
            "FLASK_SECRET_KEY is not set; using a random key, sessions will not survive a restart"
        )
    if not TMDB_API_KEY:
        logger.warning(
            "TMDB_API_KEY is not set; pages show no posters, trailers or Turkish texts, "
            "and person pages are unavailable"
        )

    app.after_request(_set_security_headers)
    app.after_request(remember_language)
    for code in _ERRORS:
        app.register_error_handler(code, _handle_http_error)
    _register_template_helpers(app)

    try:
        engine = RecommendationEngine(MOVIES_CSV, CREDITS_CSV)
    except (OSError, ValueError) as exc:
        # A missing, unreadable or malformed CSV is a setup problem, not a bug: no traceback.
        logger.error(
            "Recommendation engine failed to start: %s. Put the TMDB 5000 CSVs in %s "
            "or point DATASET_DIR at them.",
            exc,
            DATASET_DIR,
        )
        engine = None
    except Exception:
        logger.exception("Recommendation engine failed to start")
        engine = None
    app.extensions["engine"] = engine
    app.extensions["details"] = MovieDetails(engine) if engine is not None else None

    # Imported here because the blueprints import the accessors defined above.
    from movierec.web import lists, movies

    app.register_blueprint(movies.bp)
    app.register_blueprint(lists.bp)
    return app


def _set_security_headers(response):
    """Add the security headers to every response that does not set them itself."""
    response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    response.headers.setdefault("Permissions-Policy", PERMISSIONS_POLICY)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    # For browsers without CSP frame-ancestors.
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # No Cross-Origin-Embedder-Policy: TMDb posters and the YouTube trailer do not opt in.
    response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
    # SESSION_COOKIE_SECURE=1 says the site is served over HTTPS. No includeSubDomains:
    # other sites on the same domain may still need plain HTTP.
    if current_app.config["SESSION_COOKIE_SECURE"]:
        response.headers.setdefault("Strict-Transport-Security", f"max-age={ONE_YEAR_SECONDS}")
    return response


def _handle_http_error(error):
    """Answer an HTTP error with JSON under /api/ and with the Turkish error page elsewhere."""
    code = error.code
    name, message = _ERRORS[code]
    if request.path.startswith(API_PREFIXES):
        return jsonify(status="error", error=name), code
    return render_template("error.html", code=code, message=message), code


def _register_template_helpers(app: Flask) -> None:
    """Give templates asset_url(), icon_view_box(), current_year, lang, t() and the formats."""
    # Hashed once at startup: static files only change with a deploy, which restarts the app.
    asset_versions = build_asset_versions(app.static_folder)
    # Read once for the same reason. templates/_icon.html puts the symbol's viewBox on each
    # icon, so the icon keeps its Font Awesome width at a height of 1em.
    sprite = ElementTree.parse(Path(app.static_folder) / "icons.svg").getroot()
    # ElementTree spells a namespaced tag as {namespace}name: every SVG <symbol>.
    icon_view_boxes = {
        symbol.get("id"): symbol.get("viewBox")
        for symbol in sprite.iter("{http://www.w3.org/2000/svg}symbol")
    }

    @app.template_global()
    def asset_url(filename):
        """URL of a static file with a content version, so a changed file gets a new URL."""
        return url_for("static", filename=filename, v=asset_versions.get(filename, "0"))

    @app.template_global()
    def icon_view_box(name):
        """viewBox of an icon in static/icons.svg; an unknown name raises KeyError."""
        return icon_view_boxes[name]

    @app.context_processor
    def inject_year_and_language():
        return {"current_year": datetime.date.today().year, "lang": get_lang()}

    # Globals, not context values: a macro imported with {% from %}, such as the movie card,
    # does not see the context. The labels and formats follow the page language.
    app.add_template_global(t)
    app.add_template_global(lang_url)
    app.add_template_global(lambda name: genre_label(name, get_lang()), "genre_label")
    app.add_template_filter(lambda value: format_date(value, get_lang()), "format_date")
    app.add_template_filter(lambda value: format_number(value, get_lang()), "format_number")
    app.add_template_filter(lambda value: format_rating(value, get_lang()), "format_rating")
