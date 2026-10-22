"""Entry point: `app:app` for gunicorn, `python app.py` for the dev server."""

import logging
import os
import re
import sys

from movierec.web import create_app

logger = logging.getLogger(__name__)

app = create_app()


def _port(raw: str) -> int:
    """The PORT setting; an invalid one stops the dev server with a clear message."""
    if not re.fullmatch(r"[0-9]{1,5}", raw) or not 1 <= int(raw) <= 65535:
        sys.exit(f"PORT must be a number from 1 to 65535, not {raw!r}")
    return int(raw)


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = _port(os.getenv("PORT", "").strip() or "5001")
    debug = os.getenv("FLASK_DEBUG") == "1"
    logger.info("Serving on http://%s:%s", host, port)
    # The engine is built at import time; the reloader would build it twice.
    app.run(host=host, port=port, debug=debug, use_reloader=False)
