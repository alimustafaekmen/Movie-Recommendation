"""The favourites and watchlist page and the card endpoint it uses.

The lists themselves live in the visitor's browser (static/lists.js); the server keeps no state.
"""

import re

from flask import Blueprint, abort, jsonify, render_template, request

from movierec.web import MAX_MOVIE_ID, get_details, get_engine

bp = Blueprint("lists", __name__)

# Keeps the query string short; my_lists.js sends longer lists in chunks of this size.
MAX_IDS_PER_REQUEST = 100
# Each id outside the dataset costs a TMDb request, so one call looks up at most this many
# and leaves out the rest. Dataset ids do not count.
MAX_TMDB_LOOKUPS = 20
# At most ten digits, so int() never sees a huge number.
_ID_TOKEN = re.compile(r"[0-9]{1,10}")


def _requested_ids(raw):
    """Return the distinct valid movie ids of a comma-separated list, in request order."""
    ids = {}
    for token in raw.split(","):
        token = token.strip()
        if _ID_TOKEN.fullmatch(token) and 1 <= int(token) <= MAX_MOVIE_ID:
            ids[int(token)] = None
    return list(ids)


@bp.route("/my_lists")
def my_lists():
    """The favourites and watchlist page; my_lists.js fills it from the browser's storage."""
    return render_template("personal_page.html")


@bp.route("/api/movies")
def api_movies():
    """Short cards for ?ids=155,19995 in request order; ids that resolve to nothing are left out."""
    engine = get_engine()
    if engine is None:
        abort(503)
    ids = _requested_ids(request.args.get("ids", ""))
    if len(ids) > MAX_IDS_PER_REQUEST:
        abort(400)
    outside = [movie_id for movie_id in ids if movie_id not in engine.row_of]
    skipped = set(outside[MAX_TMDB_LOOKUPS:])
    kept = [movie_id for movie_id in ids if movie_id not in skipped]
    cards = map(get_details().get_movie_info_for_list, kept)
    return jsonify([card for card in cards if card is not None])
