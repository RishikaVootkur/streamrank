import numpy as np
import polars as pl
import pytest

from streamrank.data.split import SplitConfig, temporal_split
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.eval.dataset import EvalSetup, build_targets, build_train
from streamrank.eval.evaluate import fit_and_evaluate
from streamrank.models.baselines import ALS, EASE, ItemKNN, Popularity, RecentPopularity


@pytest.fixture(scope="module")
def setup() -> EvalSetup:
    raw = generate(
        SyntheticConfig(
            n_users=1500, n_items=300, affinity_strength=4.0, popularity_exponent=0.5, seed=5
        )
    ).ratings
    ratings = raw.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    split = temporal_split(ratings, SplitConfig(val_fraction=0.1, test_fraction=0.1))
    train = build_train(split.train, split.t1)
    return EvalSetup(train=train, targets=build_targets(train, split.val), partition="val")


def _tiny() -> EvalSetup:
    history = pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 3],
            "item_id": [10, 11, 10, 11, 10, 11, 12],
            "rating": [5.0, 5.0, 5.0, 4.0, 5.0, 5.0, 2.0],
            "ts": [1, 2, 3, 4, 50, 60, 95],
        }
    )
    train = build_train(history, cutoff_ts=100)
    return EvalSetup(
        train=train,
        targets=build_targets(train, history.head(0).with_columns(label=pl.lit(1, pl.Int8))),
        partition="val",
    )


def test_popularity_counts_signal() -> None:
    train = _tiny().train
    model = Popularity()
    model.fit(train)
    assert model.score(np.array([0, 1])).tolist() == [[3, 3, 1], [3, 3, 1]]
    model = Popularity(signal="positive")
    model.fit(train)
    assert model.score(np.array([0]))[0].tolist() == [3, 3, 0]


def test_recent_popularity_prefers_recent_items() -> None:
    model = RecentPopularity(window_days=0)
    model.fit(_tiny().train)
    scores = model.score(np.array([0]))[0]
    # Nothing falls in a zero-day window, so ties break by all-time popularity.
    assert scores[0] == scores[1] > scores[2]
    model = RecentPopularity(window_days=1)
    model.fit(_tiny().train)
    scores = model.score(np.array([0]))[0]
    assert scores.argmax() in (0, 1)
    assert scores.min() > 0


def test_item_knn_scores_co_watched_items() -> None:
    model = ItemKNN(neighbors=5, shrink=0.0)
    model.fit(_tiny().train)
    # Item 12 was only watched with 10 and 11, so a user who watched 10 and 11 scores it.
    assert model.score(np.array([0]))[0, 2] > 0


def test_ease_zero_diagonal_and_outside_items() -> None:
    model = EASE(l2=1.0, max_items=2)
    model.fit(_tiny().train)
    scores = model.score(np.array([0]))
    assert scores.shape == (1, 3)
    assert np.isfinite(scores).all()
    # Item 12 is outside the two most popular items and scores lowest.
    assert scores[0, 2] < scores[0, :2].min()


def test_scorers_need_fit() -> None:
    for model in (ItemKNN(), EASE()):
        with pytest.raises(RuntimeError, match="fit"):
            model.score(np.array([0]))
    with pytest.raises(RuntimeError, match="fit"):
        ALS().score(np.array([0]))


@pytest.mark.parametrize(
    "model",
    [
        Popularity(),
        RecentPopularity(window_days=365),
        ItemKNN(neighbors=50),
        EASE(l2=50.0),
        ALS(factors=16, iterations=5),
    ],
    ids=lambda m: m.name,
)
def test_models_produce_valid_lists(setup: EvalSetup, model: object) -> None:
    result = fit_and_evaluate(model, setup, n_resamples=50)  # type: ignore[arg-type]
    assert result.n_users == setup.targets.user_rows.size
    assert 0.0 <= result.metrics["recall@200"].mean <= 1.0


def test_personalized_models_beat_popularity(setup: EvalSetup) -> None:
    pop = fit_and_evaluate(Popularity(), setup, n_resamples=50).metrics["recall@50"].mean
    for model in (ItemKNN(neighbors=100), EASE(l2=50.0)):
        recall = fit_and_evaluate(model, setup, n_resamples=50).metrics["recall@50"].mean
        assert recall > pop, model.name


def _negatives_only_user() -> EvalSetup:
    # User 4 rated only item 13 (low), so its positive-signal row is empty.
    history = pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 4],
            "item_id": [10, 11, 11, 12, 11, 13],
            "rating": [5.0, 5.0, 5.0, 4.0, 5.0, 2.0],
            "ts": [1, 2, 3, 4, 5, 6],
        }
    )
    train = build_train(history, cutoff_ts=100)
    return EvalSetup(
        train=train,
        targets=build_targets(train, history.head(0).with_columns(label=pl.lit(1, pl.Int8))),
        partition="val",
    )


@pytest.mark.parametrize(
    "model",
    [
        ItemKNN(signal="positive"),
        EASE(l2=1.0, signal="positive"),
        ALS(factors=4, iterations=2, signal="positive"),
    ],
    ids=lambda m: m.name,
)
def test_empty_rows_fall_back_to_popularity_order(model: object) -> None:
    setup = _negatives_only_user()
    model.fit(setup.train)  # type: ignore[attr-defined]
    scores = model.score(np.array([3]))[0]  # type: ignore[attr-defined]
    # Item 11 is the most watched; ties among zero scores break by popularity.
    assert int(np.argmax(scores)) == 1
