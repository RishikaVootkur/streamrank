"""Raw MovieLens file names and CSV headers, shared by the real and synthetic data paths."""

from typing import Final

RATINGS_FILE: Final = "ratings.csv"
MOVIES_FILE: Final = "movies.csv"
TAGS_FILE: Final = "tags.csv"
LINKS_FILE: Final = "links.csv"

RAW_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    RATINGS_FILE: ("userId", "movieId", "rating", "timestamp"),
    MOVIES_FILE: ("movieId", "title", "genres"),
    TAGS_FILE: ("userId", "movieId", "tag", "timestamp"),
    LINKS_FILE: ("movieId", "imdbId", "tmdbId"),
}

GENRES: Final[tuple[str, ...]] = (
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)
NO_GENRES: Final = "(no genres listed)"
