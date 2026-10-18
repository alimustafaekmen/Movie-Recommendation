"""Entry point: `app:app` for gunicorn, `python app.py` for the dev server."""

import logging
import os

from movierec.web import create_app

logger = logging.getLogger(__name__)

app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5001"))
    debug = os.getenv("FLASK_DEBUG") == "1"
    logger.info("Serving on http://%s:%s", host, port)
    # The engine is built at import time; the reloader would build it twice.
    app.run(host=host, port=port, debug=debug, use_reloader=False)
