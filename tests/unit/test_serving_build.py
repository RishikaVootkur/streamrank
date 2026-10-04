import numpy as np
import polars as pl

from streamrank.data.columns import GENRES
from streamrank.serving.build import item_arrays, profile_arrays


def test_item_and_profile_arrays() -> None:
    movies = pl.DataFrame(
        {
            "item_id": [1, 2, 3],
            "title": ["A (1990)", "B", "C (2010)"],
            "year": [1990, None, 2010],
            "genres": [["Drama"], ["Comedy", "Unknown"], []],
        }
    )
    items = item_arrays(np.array([3, 1, 2, 99]), movies)
    assert items["has_movie"].tolist() == [True, True, True, False]
    assert items["item_ids"].tolist() == [3, 1, 2, 99]
    assert items["titles"].tolist() == ["C (2010)", "A (1990)", "B", ""]
    assert np.isnan(items["year"][2]) and items["year"][1] == 1990
    assert items["genres"][1, GENRES.index("Drama")] == 1.0
    assert items["genres"][2].sum() == 1  # unknown genre ignored
    history = pl.DataFrame(
        {"user_id": [5, 5, 6], "item_id": [1, 3, 2], "rating": [5.0, 1.0, 4.0], "ts": [1, 2, 3]}
    )
    prof = profile_arrays(history, movies)
    assert prof["user_ids"].tolist() == [5, 6]
    assert prof["genres"].shape == (2, len(GENRES))
    assert prof["genres"][0, GENRES.index("Drama")] == 1.0
    assert prof["mean_year"][0] == 2000.0


def test_manifest_lists_only_files_the_builder_writes() -> None:
    import inspect  # noqa: PLC0415

    from streamrank.serving import build  # noqa: PLC0415

    source = inspect.getsource(build.run)
    listed = source.split("outputs = [", 1)[1].split("]", 1)[0]
    for name in ("ranker.onnx", "item_vectors.npy", "loadtest_users.json", "items.json"):
        assert f'"{name}"' in listed
    assert '"ranker.txt"' not in listed
