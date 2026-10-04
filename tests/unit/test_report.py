import json
from pathlib import Path

import pytest

from streamrank.report import README_BEGIN, README_END, main, render, splice


def cell(mean: float, half: float = 0.01) -> dict[str, float]:
    return {"mean": mean, "low": mean - half, "high": mean + half}


def write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def make_artifacts(root: Path) -> Path:
    art = root / "artifacts"
    row = {
        "model": "ease",
        "n_users": 100,
        "recall@10": cell(0.1),
        "recall@100": cell(0.3),
        "ndcg@10": cell(0.12),
        "mrr": cell(0.2),
        "coverage@10": 0.05,
    }
    write(
        art / "final" / "retrieval" / "summary.json",
        {
            "result": row | {"model": "two_tower", "recall@100": cell(0.28)},
            "ease": row,
            "recall@100_minus_ease": cell(-0.02, 0.005),
        },
    )
    metric = {
        "two_stage": cell(0.15),
        "ease": cell(0.12),
        "retrieval": cell(0.11),
        "difference": cell(0.03, 0.02),
        "lift_over_retrieval": cell(0.04, 0.01),
    }
    write(art / "final" / "two_stage_test.json", {"ndcg@10": metric, "recall@10": metric})
    write(art / "final" / "baselines" / "results.json", [row])
    write(
        art / "retrieval" / "s10_abl_no_logq" / "summary.json",
        {"heldout": {"recall@100": cell(0.18)}},
    )
    return art


def test_render_reports_intervals_and_verdicts(tmp_path: Path) -> None:
    block, doc = render(make_artifacts(tmp_path))
    assert "| EASE (best baseline) | 0.3000 [0.2900, 0.3100] |" in block
    assert "Recall@100: -0.0200 [-0.0250, -0.0150] (worse)" in block
    assert "NDCG@10: +0.0400 [+0.0300, +0.0500] (better)" in block
    assert "NDCG@10: +0.0300 [+0.0100, +0.0500] (better)" in block
    assert "Baselines on test" in doc
    assert "| No log-Q correction | 0.1800 [0.1700, 0.1900] |" in doc
    assert "Serving latency" not in doc  # optional sections without files are skipped
    assert "No log-Q correction" in block


def test_splice_replaces_only_the_marked_block() -> None:
    readme = f"intro\n{README_BEGIN}\nold\n{README_END}\noutro\n"
    assert splice(readme, "new\n") == f"intro\n{README_BEGIN}\nnew\n{README_END}\noutro\n"
    with pytest.raises(ValueError, match="markers"):
        splice("no markers", "new\n")


def test_main_writes_doc_and_readme(tmp_path: Path) -> None:
    art = make_artifacts(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n{README_BEGIN}\n{README_END}\n")
    doc = tmp_path / "results.md"
    main(["--artifacts-dir", str(art), "--doc", str(doc), "--readme", str(readme)])
    assert doc.read_text().startswith("# Results")
    assert "Two-tower + LambdaMART ranker" in readme.read_text()
