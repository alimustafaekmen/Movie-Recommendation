"""Genre names: English values used in data and URLs, Turkish labels shown in the UI."""

# 'Foreign' and 'TV Movie' describe where a film comes from rather than what it is,
# and only a handful of dataset films carry them, so they are not offered as filters.
EXCLUDED_GENRES = frozenset({"foreign", "tv movie"})

GENRE_LABELS_TR = {
    "Action": "Aksiyon",
    "Adventure": "Macera",
    "Animation": "Animasyon",
    "Comedy": "Komedi",
    "Crime": "Suç",
    "Documentary": "Belgesel",
    "Drama": "Dram",
    "Family": "Aile",
    "Fantasy": "Fantastik",
    "Foreign": "Yabancı",
    "History": "Tarih",
    "Horror": "Korku",
    "Music": "Müzik",
    "Mystery": "Gizem",
    "Romance": "Romantik",
    "Science Fiction": "Bilim Kurgu",
    "TV Movie": "TV Filmi",
    "Thriller": "Gerilim",
    "War": "Savaş",
    "Western": "Vahşi Batı",
}

# TMDb's movie genre ids; tr-TR responses name genres in Turkish, so ids are the stable key.
TMDB_GENRE_NAMES = {
    28: "Action",
    12: "Adventure",
    16: "Animation",
    35: "Comedy",
    80: "Crime",
    99: "Documentary",
    18: "Drama",
    10751: "Family",
    14: "Fantasy",
    36: "History",
    27: "Horror",
    10402: "Music",
    9648: "Mystery",
    10749: "Romance",
    878: "Science Fiction",
    10770: "TV Movie",
    53: "Thriller",
    10752: "War",
    37: "Western",
}
TMDB_GENRE_IDS = {name: genre_id for genre_id, name in TMDB_GENRE_NAMES.items()}


def genre_label(name: str) -> str:
    """Return the Turkish label for an English genre name, or the name itself."""
    return GENRE_LABELS_TR.get(name, name)


def is_excluded(name: str) -> bool:
    """Return True for genres that are not offered as filters or badges."""
    return name.casefold() in EXCLUDED_GENRES


def genre_badge(name: str) -> dict:
    """Return the detail-page badge for an English genre name."""
    return {"id": TMDB_GENRE_IDS.get(name), "display_name": genre_label(name), "link_name": name}
