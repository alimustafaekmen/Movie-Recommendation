"""Flask app factory: configuration, security, error pages and template helpers."""

import datetime
import hashlib
import logging
from pathlib import Path

from flask import Flask, current_app, jsonify, render_template, request, url_for

from movierec.config import (
    BASE_DIR,
    CREDITS_CSV,
    FLASK_SECRET_KEY,
    LOG_LEVEL,
    MOVIES_CSV,
    SECRET_KEY_IS_EPHEMERAL,
    SESSION_COOKIE_SECURE,
)
from movierec.details import MovieDetails
from movierec.engine import RecommendationEngine
from movierec.genres import genre_label
from movierec.web.redirects import API_PREFIXES

logger = logging.getLogger(__name__)

_ERRORS = {
    400: ("bad_request", "İstek işlenemedi. Lütfen tekrar deneyin."),
    404: ("not_found", "Aradığınız sayfa bulunamadı."),
    405: ("method_not_allowed", "Bu adres bu şekilde kullanılamaz."),
    500: ("server_error", "Beklenmedik bir hata oluştu. Lütfen daha sonra tekrar deneyin."),
    503: ("unavailable", "Servis şu anda kullanılamıyor. Lütfen daha sonra tekrar deneyin."),
}
# TMDb ids are positive 32-bit integers; anything else is a 404 before it reaches the engine.
MAX_MOVIE_ID = 2147483647
MOVIE_ID = f"int(min=1, max={MAX_MOVIE_ID})"

_TURKISH_MONTHS = (
    "Ocak",
    "Şubat",
    "Mart",
    "Nisan",
    "Mayıs",
    "Haziran",
    "Temmuz",
    "Ağustos",
    "Eylül",
    "Ekim",
    "Kasım",
    "Aralık",
)


class _App(Flask):
    """Flask whose static files are cached for long only under a versioned URL."""

    def get_send_file_max_age(self, filename):
        # Only asset_url() links carry ?v=. Module imports such as "./lists.js" cannot, so
        # those are revalidated on every use; a long cache would mix old and new files.
        if "v" not in request.args:
            return 0
        return super().get_send_file_max_age(filename)


def get_engine():
    """The recommendation engine, or None when it failed to start."""
    return current_app.extensions["engine"]


def get_details():
    """The MovieDetails helper for the engine, or None when the engine failed to start."""
    return current_app.extensions["details"]


def build_asset_versions(folder):
    """Map each file under folder (as a POSIX relative path) to a short hash of its content."""
    root = Path(folder)
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def format_date_filter(value):
    """Format an ISO date such as '2023-10-26' as '26 Ekim 2023'."""
    if not value or not str(value).strip():
        return "Bilinmiyor"
    try:
        day = datetime.date.fromisoformat(str(value))
    except ValueError:
        return value
    return f"{day.day:02d} {_TURKISH_MONTHS[day.month - 1]} {day.year}"


def create_app():
    """Build the Flask app: the engine is loaded here, once per process."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
    )
    # Templates link static files through asset_url(), whose ?v= changes with the file
    # content, so browsers may keep them for a year. Not in debug mode, so edits show on reload.
    if not app.debug:
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 31536000
    app.jinja_env.globals["genre_label"] = genre_label
    if SECRET_KEY_IS_EPHEMERAL:
        logger.warning(
            "FLASK_SECRET_KEY is not set; using a random key, sessions will not survive a restart"
        )

    _register_security_headers(app)
    _register_error_handlers(app)
    _register_template_helpers(app)

    try:
        engine = RecommendationEngine(MOVIES_CSV, CREDITS_CSV)
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


def _register_security_headers(app):
    @app.after_request
    def set_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        return response


def _register_error_handlers(app):
    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(405)
    @app.errorhandler(500)
    @app.errorhandler(503)
    def handle_http_error(error):
        code = error.code
        name, message = _ERRORS[code]
        if request.path.startswith(API_PREFIXES):
            return jsonify(status="error", error=name), code
        return render_template("error.html", code=code, message=message), code


def _register_template_helpers(app):
    # Hashed once at startup: static files only change with a deploy, which restarts the app.
    asset_versions = build_asset_versions(app.static_folder)

    @app.template_global()
    def asset_url(filename):
        """URL of a static file with a content version, so a changed file gets a new URL."""
        return url_for("static", filename=filename, v=asset_versions.get(filename, "0"))

    @app.context_processor
    def inject_current_year():
        return {"current_year": datetime.date.today().year}

    app.add_template_filter(format_date_filter, "format_date")
