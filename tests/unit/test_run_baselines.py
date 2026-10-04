import json
from pathlib import Path

import pytest

from streamrank.common.provenance import verify_outputs
from streamrank.data.split import SplitConfig, temporal_split, write_split
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.eval.run_baselines import candidates, main


@pytest.fixture(scope="module")
def split_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("split")
    raw = generate(SyntheticConfig(n_users=600, n_items=200, seed=9)).ratings
    ratings = raw.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    cfg = SplitConfig(val_fraction=0.1, test_fraction=0.1)
    write_split(temporal_split(ratings, cfg), out, cfg, data_hash="synthetic")
    return out


def test_quick_grid_has_one_setting_per_family() -> None:
    cands = candidates(seed=1, quick=True)
    assert [c.name for c in cands] == ["popularity", "recent_popularity", "item_knn", "ease", "als"]
    assert all(len(list(c.configs())) == 1 for c in cands)
    assert sum(len(list(c.configs())) for c in candidates(seed=1)) == 38


def test_run_writes_results_manifest_and_doc(split_dir: Path, tmp_path: Path) -> None:
    out, doc = tmp_path / "out", tmp_path / "baselines.md"
    main(
        [
            "--tune-dir", str(split_dir),
            "--eval-dir", str(split_dir),
            "--out-dir", str(out),
            "--n-resamples", "20",
            "--quick",
            "--doc", str(doc),
        ]
    )  # fmt: skip
    manifest = verify_outputs(out)
    assert manifest["stage"] == "baselines"
    assert manifest["data_hash"] == "synthetic"
    rows = json.loads((out / "results.json").read_text())
    assert [r["model"] for r in rows] == [c.name for c in candidates(1, quick=True)]
    tuning = json.loads((out / "tuning.json").read_text())
    assert tuning["als"]["best_params"]["factors"] == 64
    text = doc.read_text()
    assert "| item_knn |" in text
    assert "Best baseline by recall@100" in text
