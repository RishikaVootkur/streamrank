"""Feast definitions generated from the shared feature specs.

Snapshot tables come from `make features` (Spark). Each snapshot holds aggregates over
days before its timestamp, so the "latest row at or before t" lookup Feast performs is
point-in-time correct. TTL is unlimited: a snapshot stays valid until the next one.
"""

import os
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Array, Float64, Int64, PrimitiveFeastType, String

from streamrank.features.definitions import (
    ITEM_FEATURES,
    USER_FEATURES,
    Aggregate,
    FeatureSet,
    value_type,
)

FEATURES_DIR = Path(
    os.environ.get(
        "STREAMRANK_FEATURES_DIR", Path(__file__).resolve().parent.parent / "data" / "features"
    )
)
NO_TTL = timedelta(0)

user = Entity(
    name="user", join_keys=["user_id"], value_type=ValueType.INT64, description="MovieLens user"
)
item = Entity(
    name="item", join_keys=["item_id"], value_type=ValueType.INT64, description="MovieLens movie"
)
ENTITIES = {"user": user, "item": item}


def _dtype(spec: Aggregate) -> PrimitiveFeastType:
    return Int64 if value_type(spec) == "int64" else Float64


def _view(fs: FeatureSet) -> FeatureView:
    source = FileSource(
        name=f"{fs.entity}_snapshots",
        path=str(FEATURES_DIR / f"{fs.entity}_features.parquet"),
        timestamp_field="event_timestamp",
    )
    schema = [Field(name=a.name, dtype=_dtype(a)) for a in fs.aggregates]
    schema += [Field(name=r.name, dtype=Float64) for r in fs.ratios]
    return FeatureView(
        name=f"{fs.entity}_stats",
        entities=[ENTITIES[fs.entity]],
        ttl=NO_TTL,
        schema=schema,
        source=source,
        online=True,
    )


user_stats = _view(USER_FEATURES)
item_stats = _view(ITEM_FEATURES)

item_content = FeatureView(
    name="item_content",
    entities=[item],
    ttl=NO_TTL,
    schema=[Field(name="item_year", dtype=Int64), Field(name="item_genres", dtype=Array(String))],
    source=FileSource(
        name="item_content_source",
        path=str(FEATURES_DIR / "item_content.parquet"),
        timestamp_field="event_timestamp",
    ),
    online=True,
)
