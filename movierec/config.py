"""Settings for the web app and the recommendation engine, read from the environment and .env."""

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Values shipped in .env.example and older versions of this file; never treat them as real keys.
_PLACEHOLDER_SECRET_KEYS = {"", "your_secret_key_here", "dev_secret_key_!@#$"}
_PLACEHOLDER_TMDB_KEYS = {"", "your_api_key", "your_tmdb_api_key"}


def _resolve_secret_key(raw):
    """Return (key, is_ephemeral); placeholders are replaced by a random per-process key."""
    key = (raw or "").strip()
    if key in _PLACEHOLDER_SECRET_KEYS:
        return secrets.token_hex(32), True
    return key, False


def _resolve_tmdb_key(raw):
    """Return the TMDb API key, or None when it is unset or still a placeholder."""
    key = (raw or "").strip()
    return None if key in _PLACEHOLDER_TMDB_KEYS else key


# TMDb
TMDB_API_KEY = _resolve_tmdb_key(os.getenv("TMDB_API_KEY"))
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/"
TMDB_API_BASE_URL = "https://api.themoviedb.org/3"

# Files. DATASET_DIR holds the TMDB 5000 CSVs; the app writes no files.
DATASET_DIR = os.getenv("DATASET_DIR") or str(BASE_DIR / "dataset")
MOVIES_CSV = os.path.join(DATASET_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(DATASET_DIR, "tmdb_5000_credits.csv")

# Recommendations
# 12 fills grid rows of 2, 3, 4 or 6 cards; "Daha fazla göster" adds 12 more.
DEFAULT_RECOMMENDATION_COUNT = 12
MAX_RECOMMENDATION_COUNT = 30
MIN_VOTES_THRESHOLD = 100

# Hybrid score weights; they sum to 1.
SIMILARITY_WEIGHT = 0.50
VOTE_AVERAGE_WEIGHT = 0.20
VOTE_COUNT_WEIGHT = 0.10
POPULARITY_WEIGHT = 0.20

# Content-similarity weights of the TF-IDF blocks; each block is L2-normalised first, so
# a block's share of the cosine grows with the square of its weight.
TFIDF_GENRE_WEIGHT = 0.3
TFIDF_DIRECTOR_WEIGHT = 0.2
TFIDF_KEYWORD_WEIGHT = 0.4
TFIDF_TEXT_WEIGHT = 0.6

# Feature-search score weights; they sum to 1.
FEATURE_RATING_WEIGHT = 0.40
FEATURE_POPULARITY_WEIGHT = 0.45
FEATURE_VOTE_COUNT_WEIGHT = 0.15

# thefuzz score (0-100) a title must reach to count as a match.
FUZZY_SCORE_CUTOFF = 75

# Flask
FLASK_SECRET_KEY, SECRET_KEY_IS_EPHEMERAL = _resolve_secret_key(os.getenv("FLASK_SECRET_KEY"))
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE") == "1"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
