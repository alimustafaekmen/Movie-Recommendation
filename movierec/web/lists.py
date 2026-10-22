"""The favourites and watchlist page and the card endpoints it uses.

The lists themselves live in the visitor's browser (static/js/list_store.js); the server keeps
no state.
"""

import re
from time import monotonic

from flask import Blueprint, abort, jsonify, render_template, request

from movierec.web import MAX_MOVIE_ID, get_details, get_engine

bp = Blueprint("lists", __name__)

# Keeps the query string short.
MAX_IDS_PER_REQUEST = 100
# Each id outside the dataset costs a TMDb request, so one call looks up at most this many
# and leaves out the rest. Dataset ids do not count. my_lists.js asks for cards in chunks of
# this size, so it loses none.
MAX_TMDB_LOOKUPS = 20
# ...and starts none after this many seconds, so a slow TMDb cannot hold the request for
# minutes. Ids left out stay in the browser's list and are asked for again on the next load.
TMDB_LOOKUP_SECONDS = 8.0
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
    return render_template("my_lists.html")


@bp.route("/api/movies")
def api_movies():
    """Short cards for ?ids=155,19995 in request order; ids that resolve to nothing are left out."""
    engine = get_engine()
    if engine is None:
        abort(503)
    deadline = monotonic() + TMDB_LOOKUP_SECONDS
    ids = _requested_ids(request.args.get("ids", ""))
    if len(ids) > MAX_IDS_PER_REQUEST:
        abort(400)
    return jsonify(_list_cards(engine, ids, deadline))


@bp.route("/api/recommendations")
def api_recommendations():
    """Cards of films like ?favorites= that are on neither list; each list counts 100 ids."""
    engine = get_engine()
    if engine is None:
        abort(503)
    deadline = monotonic() + TMDB_LOOKUP_SECONDS
    favourites, watchlist = (
        _requested_ids(request.args.get(name, ""))[:MAX_IDS_PER_REQUEST]
        for name in ("favorites", "watchlist")
    )
    ids = engine.recommend_from_favourites(favourites, watchlist)
    return jsonify(_list_cards(engine, ids, deadline))


def _list_cards(engine, ids, deadline):
    """List cards for ids in order, with the TMDb lookups of ids outside the dataset limited."""
    outside = [movie_id for movie_id in ids if movie_id not in engine.row_of]
    skipped = set(outside[MAX_TMDB_LOOKUPS:])
    kept = [movie_id for movie_id in ids if movie_id not in skipped]
    details = get_details()
    # Dataset cards only need a poster, so those are fetched in parallel up front and handed
    # to the cards. Ids outside the dataset are full lookups of ids the visitor chose, and
    # stay one at a time.
    paths = details.prefetch_posters([movie_id for movie_id in kept if movie_id in engine.row_of])
    cards = []
    for movie_id in kept:
        if movie_id not in engine.row_of and monotonic() > deadline:
            continue
        card = details.get_list_card(movie_id, paths)
        if card is not None:
            cards.append(card)
    return cards
