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
            "partition": "test",
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
    write(
        art / "final" / "two_stage_test.json",
        {"partition": "test", "users": 100, "ndcg@10": metric, "recall@10": metric},
    )
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


def test_render_rejects_validation_files(tmp_path: Path) -> None:
    art = make_artifacts(tmp_path)
    path = art / "final" / "two_stage_test.json"
    path.write_text(path.read_text().replace('"test"', '"val"'))
    with pytest.raises(ValueError, match="test partition"):
        render(art)


def test_main_writes_nothing_without_markers(tmp_path: Path) -> None:
    art = make_artifacts(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("# x\n")
    doc = tmp_path / "results.md"
    with pytest.raises(ValueError, match="markers"):
        main(["--artifacts-dir", str(art), "--doc", str(doc), "--readme", str(readme)])
    assert not doc.exists()


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


def k6_summary(p50: float, p99: float, dropped: int = 0) -> dict[str, object]:
    durations = {"p(50)": p50, "p(95)": p50 * 2, "p(99)": p99}
    return {
        "metrics": {
            "http_req_duration{phase:load}": {"values": durations},
            "dropped_iterations": {"values": {"count": dropped}},
        }
    }


def test_serving_section_labels_runs_and_skips_missing(tmp_path: Path) -> None:
    from streamrank.report import serving_section  # noqa: PLC0415

    art = tmp_path / "artifacts"
    assert serving_section(art) == []
    write(art / "serving" / "load_summary_100.json", k6_summary(5.0, 7.0))
    write(art / "serving" / "final_load_150.json", k6_summary(5.5, 110.0))
    write(art / "k8s" / "load_summary_kind_150.json", k6_summary(6.5, 1900.0, dropped=243))
    rows = [line for line in serving_section(art) if line.startswith(("| Compose", "| kind"))]
    assert rows == [
        "| Compose, 100 req/s, quiet host (M7) | 5.0 | 10.0 | 7.0 | 0 |",
        "| Compose, 150 req/s, busy host (final) | 5.5 | 11.0 | 110.0 | 0 |",
        "| kind, 150 req/s, HPA scale-out from 1 pod (new connection per request) | 6.5 | 13.0 "
        "| 1900.0 | 243 |",
    ]
