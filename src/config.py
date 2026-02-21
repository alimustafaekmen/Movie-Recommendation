# =============================================================================
# Yapılandırma ve Sabitler / Configuration and Constants
# =============================================================================

import os
from dotenv import load_dotenv  # .env dosyasını yüklemek için / To load .env file

# ROOT_DIR is the parent directory of this file
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# --- API Ayarları / API Settings ---

# TMDb API anahtarı / TMDb API key
TMDB_API_KEY = os.getenv('TMDB_API_KEY', None)

# TMDb poster URL / TMDb poster base URL
TMDB_POSTER_BASE_URL = 'https://image.tmdb.org/t/p/'

# TMDb API URL / TMDb API base URL
TMDB_API_BASE_URL = 'https://api.themoviedb.org/3'

# API zaman aşımı (saniye) / API timeout (seconds)
API_TIMEOUT = 10


# --- Dosya Yolları / File Paths ---

# Proje kök dizini / Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

# Veri seti klasörü / Dataset folder
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')

# Film verisi CSV / Movies CSV path
MOVIES_CSV = os.path.join(DATASET_FOLDER, 'tmdb_5000_movies.csv')

# Ekip verisi CSV / Credits CSV path
CREDITS_CSV = os.path.join(DATASET_FOLDER, 'tmdb_5000_credits.csv')


# --- Öneri Sistemi Ayarları / Recommendation System Settings ---

# Varsayılan öneri sayısı / Default number of recommendations
DEFAULT_RECOMMENDATION_COUNT = 15

# Maksimum öneri sayısı / Maximum number of recommendations
MAX_RECOMMENDATION_COUNT = 30

# Minimum oy eşiği / Minimum vote threshold
MIN_VOTES_THRESHOLD = 100

# Hibrit skor ağırlıkları / Hybrid score weights
SIMILARITY_WEIGHT = 0.50    # İçerik benzerliği / Content similarity
VOTE_AVERAGE_WEIGHT = 0.20  # Oy ortalaması / Vote average
VOTE_COUNT_WEIGHT = 0.10    # Oy sayısı / Vote count
POPULARITY_WEIGHT = 0.20    # Popülerlik / Popularity

# Bulanık eşleşme eşik değeri / Fuzzy matching threshold (0-100)
FUZZY_SCORE_CUTOFF = 75

# Hariç tutulan türler / Excluded genres (too generic)
EXCLUDED_GENRES_FOR_POPULATE = {'tv', 'foreign', 'movie'}

# Flask gizli anahtar / Flask secret key
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'dev_secret_key_!@#$')
