"""Entry point: `app:app` for gunicorn, `python app.py` for the dev server."""

import logging
import os
import re
import sys
from time import perf_counter

import flask.cli
from flask import g, request

from movierec.web import create_app

logger = logging.getLogger(__name__)

app = create_app()


def _port(raw: str) -> int:
    """The PORT setting; an invalid one stops the dev server with a clear message."""
    if not re.fullmatch(r"[0-9]{1,5}", raw) or not 1 <= int(raw) <= 65535:
        sys.exit(f"PORT must be a number from 1 to 65535, not {raw!r}")
    return int(raw)


def _start_timer():
    g.started = perf_counter()


def _log_request(response):
    """Log one short line per page request; static files and health checks stay quiet."""
    if not (request.path.startswith("/static/") or request.path == "/health"):
        ms = (perf_counter() - g.started) * 1000
        logger.info(
            "%s %s %s (%.0f ms)",
            request.method,
            request.full_path.rstrip("?"),
            response.status_code,
            ms,
        )
    return response


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = _port(os.getenv("PORT", "").strip() or "5001")
    debug = os.getenv("FLASK_DEBUG") == "1"
    # Werkzeug logs every request (static files included) with its own timestamp and colour
    # codes, and Flask prints a banner; the two hooks above log page requests instead.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    flask.cli.show_server_banner = lambda *args: None
    app.before_request(_start_timer)
    app.after_request(_log_request)
    shown = "localhost" if host == "127.0.0.1" else host
    logger.info("MovieRec is running at http://%s:%s (press Ctrl+C to stop)", shown, port)
    # The engine is built at import time; the reloader would build it twice.
    app.run(host=host, port=port, debug=debug, use_reloader=False)
