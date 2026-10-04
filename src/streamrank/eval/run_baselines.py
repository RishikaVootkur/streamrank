"""Tune every baseline on the sample split, then evaluate the best settings on the full split.

Writes `tuning.json`, `results.json`, and a manifest to the output directory, and a
Markdown results table that the roadmap and README quote.
"""

import argparse
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import begin_stage, verify_outputs, write_manifest
from streamrank.common.seeds import set_global_seed
from streamrank.eval.dataset import EvalSetup, load_setup
from streamrank.eval.evaluate import EvalResult, Model, fit_and_evaluate
from streamrank.models.baselines import ALS, EASE, ItemKNN, Popularity, RecentPopularity

log = logging.getLogger(__name__)

SELECTION_METRIC = "recall@100"
TABLE_METRICS = ("recall@10", "recall@50", "recall@100", "recall@200", "ndcg@10", "mrr")
SIGNALS = ("all", "positive")


@dataclass(frozen=True)
class Candidate:
    """One model family with its hyperparameter grid."""

    name: str
    build: Callable[..., Model]
    grid: dict[str, tuple[Any, ...]]

    def configs(self) -> Iterator[dict[str, Any]]:
        keys = list(self.grid)
        for values in product(*(self.grid[k] for k in keys)):
            yield dict(zip(keys, values, strict=True))


def candidates(seed: int, quick: bool = False) -> list[Candidate]:
    """The baseline families and their search grids (one setting each when `quick`)."""
    full: list[Candidate] = [
        Candidate("popularity", Popularity, {"signal": SIGNALS}),
        Candidate(
            "recent_popularity",
            RecentPopularity,
            {"window_days": (30, 90, 365), "signal": SIGNALS},
        ),
        Candidate(
            "item_knn",
            ItemKNN,
            {"neighbors": (50, 100, 400), "shrink": (10.0, 100.0), "signal": SIGNALS},
        ),
        Candidate("ease", EASE, {"l2": (500.0, 2000.0, 8000.0), "signal": SIGNALS}),
        Candidate(
            "als",
            lambda **kw: ALS(seed=seed, **kw),
            {
                "factors": (64, 128),
                "regularization": (0.1, 1.0),
                "alpha": (10.0, 40.0),
                "signal": SIGNALS,
            },
        ),
    ]
    if quick:
        return [Candidate(c.name, c.build, {k: v[:1] for k, v in c.grid.items()}) for c in full]
    return full


def tune(
    cands: list[Candidate], setup: EvalSetup, n_resamples: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Evaluate every grid point and keep the best per family by the selection metric."""
    if setup.partition != "val":
        raise ValueError("tuning must use the validation partition")
    out: dict[str, dict[str, Any]] = {}
    for cand in cands:
        trials: list[dict[str, Any]] = []
        for params in cand.configs():
            result = fit_and_evaluate(
                cand.build(**params), setup, params, n_resamples=n_resamples, seed=seed
            )
            trials.append(result.row())
            log.info(
                "trial",
                extra={
                    "model": cand.name,
                    "params": params,
                    SELECTION_METRIC: str(result.metrics[SELECTION_METRIC]),
                },
            )
        best = max(trials, key=lambda r: r[SELECTION_METRIC]["mean"])
        out[cand.name] = {"best_params": best["params"], "trials": trials}
    return out


def evaluate_best(
    cands: list[Candidate],
    tuning: dict[str, dict[str, Any]],
    setup: EvalSetup,
    n_resamples: int,
    seed: int,
) -> list[EvalResult]:
    """Fit each family with its tuned settings and evaluate it."""
    results = []
    for cand in cands:
        params = tuning[cand.name]["best_params"]
        result = fit_and_evaluate(
            cand.build(**params), setup, params, n_resamples=n_resamples, seed=seed
        )
        log.info(
            "final", extra={"model": cand.name, **{k: str(v) for k, v in result.metrics.items()}}
        )
        results.append(result)
    return results


def _fmt(cell: dict[str, float]) -> str:
    return f"{cell['mean']:.4f} [{cell['low']:.4f}, {cell['high']:.4f}]"


def render_markdown(rows: list[dict[str, Any]], context: dict[str, Any]) -> str:
    """Render the results table with 95% bootstrap intervals."""
    header = ["Model", *TABLE_METRICS, "coverage@10", "ARP@10", "fit s"]
    lines = [
        "# Baseline results",
        "",
        f"Evaluation: {context['eval_split']} split, `{context['partition']}` partition, "
        f"{context['n_users']:,} users with training history and at least one positive "
        f"({context['n_cold_users']:,} more users with positives have no training history; "
        "they are served by the popularity fallback and are not in this table). "
        f"Ranked against all {context['n_items']:,} catalog items, excluding each user's "
        "training items. Cells show the mean with a 95% bootstrap interval over users "
        f"({context['n_resamples']} resamples).",
        "",
        f"Hyperparameters were tuned on the {context['tune_split']} split by "
        f"{SELECTION_METRIC}. ARP@10 is the mean training popularity share of recommended "
        "items (lower means less popularity bias).",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |",
    ]
    for r in rows:
        cells = [r["model"], *(_fmt(r[m]) for m in TABLE_METRICS)]
        cells += [
            f"{r['coverage@10']:.3f}",
            f"{r['arp@10']['mean']:.5f}",
            f"{r['fit_seconds']:.0f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Tuned settings:", ""]
    lines += [f"- {r['model']}: `{json.dumps(r['params'], sort_keys=True)}`" for r in rows]
    best = max(rows, key=lambda r: r[SELECTION_METRIC]["mean"])
    lines += ["", f"Best baseline by {SELECTION_METRIC}: **{best['model']}**.", ""]
    return "\n".join(lines)


def check_compatible(tune_dir: Path, eval_dir: Path) -> dict[str, Any]:
    """Verify both splits and confirm they come from the same data and cut points.

    Returns a description of the inputs (paths and file hashes) for the manifest.
    """
    tune_m, eval_m = verify_outputs(tune_dir), verify_outputs(eval_dir)
    for key in ("data_hash", "t1", "t2"):
        if tune_m.get(key) != eval_m.get(key):
            raise ValueError(f"tuning and evaluation splits differ in {key}")
    return {
        "data_hash": eval_m["data_hash"],
        "tune_split": {"path": str(tune_dir), "outputs": tune_m["outputs"]},
        "eval_split": {"path": str(eval_dir), "outputs": eval_m["outputs"]},
    }


def run(
    tune_dir: Path,
    eval_dir: Path,
    out_dir: Path,
    *,
    partition: str = "val",
    n_resamples: int = 1000,
    seed: int = 42,
    quick: bool = False,
    doc: Path | None = None,
) -> list[dict[str, Any]]:
    """Tune on `tune_dir`, evaluate on `eval_dir`, and write results to `out_dir`.

    Tuning always uses the validation partition, so test data never influences the chosen
    settings; `partition` only picks where the tuned models are evaluated.
    """
    set_global_seed(seed)
    inputs = check_compatible(tune_dir, eval_dir)
    begin_stage(out_dir)
    cands = candidates(seed, quick)
    t0 = time.perf_counter()
    tuning = tune(cands, load_setup(tune_dir, "val"), n_resamples, seed)
    (out_dir / "tuning.json").write_text(json.dumps(tuning, indent=2) + "\n")

    setup = load_setup(eval_dir, partition)
    results = evaluate_best(cands, tuning, setup, n_resamples, seed)
    rows = [r.row() for r in results]
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    context = {
        "tune_split": tune_dir.name,
        "eval_split": eval_dir.name,
        "partition": partition,
        "n_users": int(setup.targets.user_rows.size),
        "n_cold_users": setup.targets.n_cold_users,
        "n_items": setup.train.n_items,
        "n_resamples": n_resamples,
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
    }
    write_manifest(
        out_dir,
        stage="baselines",
        config={
            "partition": partition,
            "n_resamples": n_resamples,
            "quick": quick,
            "selection_metric": SELECTION_METRIC,
            "grids": {c.name: {k: list(v) for k, v in c.grid.items()} for c in cands},
        },
        data_hash=str(inputs["data_hash"]),
        seed=seed,
        outputs=[out_dir / "tuning.json", out_dir / "results.json"],
        extra={"context": context},
    )
    if doc is not None:
        doc.write_text(render_markdown(rows, context))
    return rows


def main(argv: list[str] | None = None) -> None:
    """Tune and evaluate the baseline recommenders."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=main.__doc__)
    split = settings.data_dir / "split"
    parser.add_argument("--tune-dir", type=Path, default=split / "sample10")
    parser.add_argument("--eval-dir", type=Path, default=split / "full")
    parser.add_argument("--out-dir", type=Path, default=settings.artifacts_dir / "baselines")
    parser.add_argument("--partition", default="val", choices=["val", "test"])
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="one setting per model")
    parser.add_argument("--doc", type=Path, default=None, help="write the Markdown table here")
    args = parser.parse_args(argv)
    configure_logging(settings.log_level)
    run(
        args.tune_dir,
        args.eval_dir,
        args.out_dir,
        partition=args.partition,
        n_resamples=args.n_resamples,
        seed=settings.seed,
        quick=args.quick,
        doc=args.doc,
    )


if __name__ == "__main__":
    main()
