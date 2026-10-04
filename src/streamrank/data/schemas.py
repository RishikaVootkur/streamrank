"""Validation schemas for the processed tables. Ingest fails fast if data breaks them."""

from typing import Final

import pandera.polars as pa
import polars as pl

HALF_STARS: Final = [x / 2 for x in range(1, 11)]
POSITIVE_THRESHOLD: Final = 4.0


class Ratings(pa.DataFrameModel):
    """One explicit rating per user-movie pair."""

    user_id: int = pa.Field(ge=1)
    item_id: int = pa.Field(ge=1)
    rating: float = pa.Field(isin=HALF_STARS)
    ts: int = pa.Field(ge=0)

    class Config:
        strict = True

    @pa.dataframe_check
    @classmethod
    def unique_user_item(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """Each user rates a movie at most once."""
        return data.lazyframe.select(~pl.struct("user_id", "item_id").is_duplicated())


class Movies(pa.DataFrameModel):
    """Movie metadata with parsed release year and genre list."""

    item_id: int = pa.Field(ge=1, unique=True)
    title: str
    year: int = pa.Field(nullable=True, ge=1870, le=2030)
    genres: list[str]

    class Config:
        strict = True


class Tags(pa.DataFrameModel):
    """Free-text tags applied by users, normalized to lower case."""

    user_id: int = pa.Field(ge=1)
    item_id: int = pa.Field(ge=1)
    tag: str = pa.Field(str_length={"min_value": 1})
    ts: int = pa.Field(ge=0)

    class Config:
        strict = True


class Links(pa.DataFrameModel):
    """External identifiers per movie."""

    item_id: int = pa.Field(ge=1, unique=True)
    imdb_id: int = pa.Field(nullable=True)
    tmdb_id: int = pa.Field(nullable=True)

    class Config:
        strict = True
