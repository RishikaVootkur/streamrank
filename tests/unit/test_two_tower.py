import numpy as np
import pytest
import torch

from streamrank.models.sequences import ItemContent
from streamrank.models.two_tower import (
    TensorBatch,
    TwoTower,
    TwoTowerConfig,
    attention_mask,
    cosine_lr,
    count_parameters,
    pick_device,
)

N_ITEMS = 30
L = 6


def _content() -> ItemContent:
    rng = np.random.default_rng(0)
    genres = (rng.random((N_ITEMS + 1, 4)) < 0.3).astype(np.float32)
    genres[0] = 0
    year = rng.integers(1, 5, N_ITEMS + 1)
    year[0] = 0
    tags = rng.integers(0, 6, (N_ITEMS + 1, 3))
    tags[0] = 0
    return ItemContent(genres=genres, year=year, tags=tags, n_tags=5)


def _model(**overrides: object) -> TwoTower:
    torch.manual_seed(0)
    params: dict[str, object] = {
        "dim": 16,
        "max_len": L,
        "n_layers": 2,
        "n_heads": 2,
        "dropout": 0.0,
        "n_random_negatives": 8,
    }
    cfg = TwoTowerConfig(**(params | overrides))  # type: ignore[arg-type]
    counts = torch.arange(1, N_ITEMS + 1, dtype=torch.float32)
    return TwoTower(N_ITEMS, _content(), counts, cfg).eval()


def _inputs(tokens: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.tensor(tokens)
    return t, (t % 2 == 0) & (t > 0), torch.where(t > 0, 3, 0)


def test_attention_mask_is_causal_and_hides_padding() -> None:
    valid = torch.tensor([[False, True, True]])
    m = attention_mask(valid)[0, 0]
    assert m[2, 1] == 0 and m[2, 2] == 0
    assert m[1, 2] < -1e8  # no peeking ahead
    assert m[2, 0] < -1e8  # padded key hidden
    assert m[0, 0] == 0  # padded query attends to itself (no NaN rows)


def test_outputs_are_unit_vectors() -> None:
    model = _model()
    tok, pos, gap = _inputs([[0, 0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10]])
    users = model.user_tower(tok, pos, gap)
    assert users.shape == (2, L, 16)
    assert torch.allclose(users[:, -1].norm(dim=-1), torch.ones(2), atol=1e-5)
    items = model.all_items(batch_size=7)
    assert items.shape == (N_ITEMS, 16)
    assert torch.allclose(items.norm(dim=-1), torch.ones(N_ITEMS), atol=1e-5)


@pytest.mark.parametrize("use_sequence", [True, False])
def test_user_tower_is_causal(use_sequence: bool) -> None:
    model = _model(use_sequence=use_sequence)
    a = _inputs([[1, 2, 3, 4, 5, 6]])
    b = _inputs([[1, 2, 3, 9, 9, 9]])
    out_a, out_b = model.user_tower(*a), model.user_tower(*b)
    assert torch.allclose(out_a[:, :3], out_b[:, :3], atol=1e-5)
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-3)


def test_left_padding_does_not_change_the_query_vector() -> None:
    model = _model()
    short = model.last_position(*_inputs([[0, 0, 0, 4, 5, 6]]))
    # Whatever sits in padded slots must not leak into the query vector.
    tok, pos, gap = _inputs([[0, 0, 0, 4, 5, 6]])
    gap_noise = gap.clone()
    gap_noise[0, :3] = 7
    pos_noise = pos.clone()
    pos_noise[0, :3] = True
    assert torch.allclose(model.last_position(tok, pos_noise, gap_noise), short, atol=1e-6)


def _batch() -> TensorBatch:
    tok, pos, gap = _inputs([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
    targets = torch.tensor([[0, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    return TensorBatch(tok, pos, gap, targets, targets > 0)


def test_loss_is_finite_and_trains() -> None:
    model = _model().train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(0)
    first = None
    for _ in range(60):
        loss = model.loss(_batch(), generator=g)
        assert torch.isfinite(loss)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first


def test_logq_changes_the_loss() -> None:
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    with_q = _model(use_logq=True).loss(_batch(), generator=g1)
    without = _model(use_logq=False).loss(_batch(), generator=g2)
    assert not torch.isclose(with_q, without)


def test_random_hits_on_the_positive_are_masked() -> None:
    model = _model(n_random_negatives=500)  # many draws: the positives are surely drawn
    batch = _batch()
    loss = model.loss(batch, generator=torch.Generator().manual_seed(2))
    assert torch.isfinite(loss)


def test_loss_position_cap() -> None:
    model = _model(max_loss_positions=3)
    assert torch.isfinite(model.loss(_batch(), generator=torch.Generator().manual_seed(3)))


def test_content_ablation_drops_content_parameters_from_use() -> None:
    model = _model(use_content=False)
    tok = torch.tensor([1, 2])
    before = model.item_tower(tok)
    with torch.no_grad():
        model.item_tower.genre_proj.weight.add_(1.0)
    assert torch.allclose(model.item_tower(tok), before)
    assert count_parameters(model) > 0


def test_schedule_and_device() -> None:
    assert cosine_lr(0, 100, 10) == pytest.approx(0.1)
    assert cosine_lr(9, 100, 10) == pytest.approx(1.0)
    assert cosine_lr(100, 100, 10) == pytest.approx(0.1)
    assert pick_device().type in ("mps", "cpu")
