"""Collect saved results into docs/results.md and the README results table (`make report`).

Reads only result files that earlier stages wrote (nothing is recomputed), so the report
always matches the artifacts. The final-run files are required; every other section is
skipped with a note when its file is missing.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

from streamrank.common.config import get_settings

README_BEGIN = "<!-- results:begin -->"
README_END = "<!-- results:end -->"
ABLATIONS = {
    "s10_recent_0.5_temp_0.1_dropout_0.3": "Full model",
    "s10_abl_no_sequence": "No sequence encoder (mean of history embeddings)",
    "s10_abl_no_content": "No content features (item ID only)",
    "s10_abl_no_logq": "No log-Q correction",
    "s10_abl_uniform_windows": "Uniform training windows (no recent-window sampling)",
}
BASELINE_NAMES = {
    "popularity": "Popularity",
    "recent_popularity": "Recent popularity",
    "item_knn": "Item KNN",
    "ease": "EASE",
    "als": "ALS",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def ci(cell: dict[str, float], digits: int = 4) -> str:
    """'mean [low, high]' for an interval cell."""
    return f"{cell['mean']:.{digits}f} [{cell['low']:.{digits}f}, {cell['high']:.{digits}f}]"


def signed(cell: dict[str, float], digits: int = 4) -> str:
    """A difference interval with explicit signs."""
    return f"{cell['mean']:+.{digits}f} [{cell['low']:+.{digits}f}, {cell['high']:+.{digits}f}]"


def verdict(cell: dict[str, float]) -> str:
    if cell["low"] > 0:
        return "better"
    if cell["high"] < 0:
        return "worse"
    return "tie"


def headline(retrieval: dict[str, Any], two_stage: dict[str, Any]) -> list[str]:
    """The README table: final run on the test period."""
    tt, ease = retrieval["result"], retrieval["ease"]
    nd, rc = two_stage["ndcg@10"], two_stage["recall@10"]
    diff = retrieval["recall@100_minus_ease"]
    return [
        f"Final run on the test period: {tt['n_users']:,} warm test users, full-catalog "
        "ranking, 95% bootstrap intervals over users. Retrieval models were refitted on "
        "train + validation history; nothing was tuned or early-stopped on test labels.",
        "",
        "| System | Recall@100 | NDCG@10 | Recall@10 |",
        "| --- | ---: | ---: | ---: |",
        f"| EASE (best baseline) | {ci(ease['recall@100'])} | {ci(nd['ease'])} | "
        f"{ci(rc['ease'])} |",
        f"| Two-tower retrieval | {ci(tt['recall@100'])} | {ci(nd['retrieval'])} | "
        f"{ci(rc['retrieval'])} |",
        f"| Two-tower + LambdaMART ranker | {ci(tt['recall@100'])} | {ci(nd['two_stage'])} | "
        f"{ci(rc['two_stage'])} |",
        "",
        f"- Two-tower against EASE, Recall@100: {signed(diff)} ({verdict(diff)}).",
        f"- Ranker against retrieval order, NDCG@10: {signed(nd['lift_over_retrieval'])} "
        f"({verdict(nd['lift_over_retrieval'])}).",
        f"- Two-stage against EASE, NDCG@10: {signed(nd['difference'])} "
        f"({verdict(nd['difference'])}).",
        "- The ranker reorders the same 200 candidates, so Recall@100 is shared by both "
        "two-tower rows. It was trained on validation labels and applied unchanged, with "
        "features as of the test cutoff and retrieval scores from the refitted model.",
    ]


def baselines_table(rows: list[dict[str, Any]], title: str) -> list[str]:
    out = [
        f"### {title}",
        "",
        "| Model | Recall@10 | Recall@100 | NDCG@10 | MRR | Coverage@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        name = BASELINE_NAMES.get(r["model"], r["model"])
        out.append(
            f"| {name} | {ci(r['recall@10'])} | {ci(r['recall@100'])} | {ci(r['ndcg@10'])} | "
            f"{ci(r['mrr'])} | {r['coverage@10']:.4f} |"
        )
    return [*out, ""]


def ablation_table(retrieval_dir: Path) -> list[str]:
    rows = []
    for run, label in ABLATIONS.items():
        path = retrieval_dir / run / "summary.json"
        if path.exists():
            rows.append(f"| {label} | {ci(_load(path)['heldout']['recall@100'])} |")
    if not rows:
        return []
    return [
        "#### Two-tower ablations (validation, 10% user sample)",
        "",
        "Recall@100 on the sample's users that early stopping did not use.",
        "",
        "| Variant | Recall@100 |",
        "| --- | ---: |",
        *rows,
        "",
    ]


def optional_sections(art: Path) -> list[str]:
    out: list[str] = []
    ranker = art / "ranker" / "summary.json"
    if ranker.exists():
        r = _load(ranker)
        top = sorted(r["shap_importance"].items(), key=lambda kv: -kv[1])[:5]
        out += [
            "### Ranker (validation)",
            "",
            f"NDCG@10 on {r['n_users']['eval']:,} held-out validation users: retrieval order "
            f"{ci(r['ndcg@10_retrieval'])}, ranker {ci(r['ndcg@10_ranker'])}, lift "
            f"{signed(r['ndcg@10_lift'])}. Top features by mean |SHAP|: "
            + ", ".join(f"{name} ({value:.3f})" for name, value in top)
            + ".",
            "",
        ]
    index = art / "index" / "items.json"
    if index.exists():
        b = _load(index)["benchmark"]
        out += [
            "### Retrieval index",
            "",
            f"{b['label']}: recall@200 against exact search {b['recall@200_vs_exact']:.4f}, "
            f"single-query p50 {b['p50_ms']:.2f} ms, p99 {b['p99_ms']:.2f} ms.",
            "",
        ]
    sims = sorted((art / "simulation").glob("interleaving_*.json"))
    if sims:
        out += [
            "### Interleaving replay (validation events)",
            "",
            "| A | B | Batch features | Preference for B | Winner |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for path in sims:
            s = _load(path)
            winner = {"A": s["pair"]["A"], "B": s["pair"]["B"]}.get(s["winner"], s["winner"])
            out.append(
                f"| {s['pair']['A']} | {s['pair']['B']} | {s['features']} | "
                f"{signed(s['delta'], 3)} | {winner} |"
            )
        out.append("")
    parity = art / "streaming" / "parity.json"
    if parity.exists():
        p = _load(parity)
        out += [
            "### Streaming parity",
            "",
            f"{p['events']:,} replayed events for {p['users']:,} users: {p['mismatches']} "
            f"mismatches between online and batch session features; event-to-Redis latency "
            f"p50 {p['latency']['p50_ms']:.0f} ms.",
            "",
        ]
    drift = art / "drift" / "summary.json"
    if drift.exists():
        d = _load(drift)
        flagged = [c for c, info in d["columns"].items() if info["drift"]]
        out += [
            "### Data drift",
            "",
            f"{d['drifted_columns']} of {len(d['columns'])} columns drifted between "
            f"{d['reference'][0]} to {d['reference'][1]} and {d['current'][0]} to "
            f"{d['current'][1]}: {', '.join(flagged) or 'none'}.",
            "",
        ]
    return out


def load_line(path: Path, label: str) -> str | None:
    if not path.exists():
        return None
    m = _load(path)["metrics"]
    dur = m["http_req_duration{phase:load}"]["values"]
    dropped = m.get("dropped_iterations", {}).get("values", {}).get("count", 0)
    return f"| {label} | {dur['p(50)']:.1f} | {dur['p(95)']:.1f} | {dur['p(99)']:.1f} | {dropped} |"


def serving_section(art: Path) -> list[str]:
    serving, k8s = art / "serving", art / "k8s"
    candidates = [
        *(
            (serving / f"load_summary_{rate}.json", f"Compose, {rate} req/s, quiet host (M7)")
            for rate in (100, 200, 250, 300)
        ),
        *(
            (serving / f"final_load_{rate}.json", f"Compose, {rate} req/s, busy host (final)")
            for rate in (100, 150, 200)
        ),
        (
            k8s / "load_summary_kind_150.json",
            "kind, 150 req/s, HPA scale-out from 1 pod (new connection per request)",
        ),
        (k8s / "load_summary_kind_steady_reuse.json", "kind, 150 req/s, 4 pods"),
    ]
    kept = [line for path, label in candidates if (line := load_line(path, label))]
    if not kept:
        return []
    return [
        "### Serving latency (k6)",
        "",
        "One API worker with 2 CPUs in Compose; 1 CPU per pod on kind. The quiet-host runs "
        "were measured with nothing else running; the final runs had other work on the "
        "laptop (load average about 5), which costs the single Python worker most of its "
        "headroom.",
        "",
        "| Run | p50 ms | p95 ms | p99 ms | Dropped |",
        "| --- | ---: | ---: | ---: | ---: |",
        *kept,
        "",
    ]


def render(art: Path) -> tuple[str, str]:
    """(README block, docs/results.md)."""
    final = art / "final"
    retrieval = _load(final / "retrieval" / "summary.json")
    two_stage = _load(final / "two_stage_test.json")
    if retrieval.get("partition") != "test" or two_stage.get("partition") != "test":
        raise ValueError("the final-run files must come from the test partition")
    if retrieval["result"]["n_users"] != two_stage["users"]:
        raise ValueError("retrieval and two-stage results cover different users")
    head = headline(retrieval, two_stage)
    ablations = ablation_table(art / "retrieval")
    doc = [
        "# Results",
        "",
        "Written by `make report` from saved artifacts.",
        "",
        "## Final run",
        "",
    ]
    doc += [*head, ""]
    base_test = final / "baselines" / "results.json"
    if base_test.exists():
        doc += baselines_table(_load(base_test), "Baselines on test (tuned on validation)")
    doc += ["## Development results", ""]
    base_val = art / "baselines" / "results.json"
    if base_val.exists():
        doc += baselines_table(_load(base_val), "Baselines on validation")
    doc += ablations
    doc += optional_sections(art)
    doc += serving_section(art)
    block = [*head, "", *ablations] if ablations else head
    return "\n".join(block).rstrip() + "\n", "\n".join(doc).rstrip() + "\n"


def splice(readme: str, block: str) -> str:
    """Replace the text between the README result markers with `block`."""
    pattern = re.compile(re.escape(README_BEGIN) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(readme):
        raise ValueError(f"README has no {README_BEGIN} ... {README_END} markers")
    return pattern.sub(lambda _: f"{README_BEGIN}\n{block}{README_END}", readme)


def main(argv: list[str] | None = None) -> None:
    """Write docs/results.md and refresh the README results table."""
    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--artifacts-dir", type=Path, default=s.artifacts_dir)
    p.add_argument("--doc", type=Path, default=Path("docs/results.md"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    args = p.parse_args(argv)
    block, doc = render(args.artifacts_dir)
    readme = splice(args.readme.read_text(), block)  # fails before anything is written
    args.doc.write_text(doc)
    args.readme.write_text(readme)
    print(f"wrote {args.doc} and the results table in {args.readme}")


if __name__ == "__main__":
    main()
