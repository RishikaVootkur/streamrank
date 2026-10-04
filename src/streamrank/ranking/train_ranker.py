"""Train a LightGBM LambdaMART ranker on retrieval candidates and measure its lift.

Users are split three ways so no evaluation number is selected on its own data:
- early-stop users of the retrieval model (optional, e.g. the 10% sample) are dropped,
- the rest are split by a seeded draw into ranker-train and ranker-eval halves,
- within ranker-train, 20% of users are an inner validation set for early stopping and
  the Optuna search.

The headline metric is NDCG@10 on ranker-eval users for the ranker's order versus the
retrieval order of the same candidates, with a paired bootstrap interval.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import numpy.typing as npt
import optuna
import polars as pl

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import begin_stage, verify_outputs, write_manifest
from streamrank.eval import metrics
from streamrank.ranking.features import FEATURES

log = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class RankerConfig:
    """Search space bounds and fixed training settings."""

    n_trials: int = 20
    timeout_minutes: float = 15.0
    max_rounds: int = 2000
    early_stopping_rounds: int = 50
    eval_share: float = 0.5
    inner_val_share: float = 0.2
    seed: int = 42


def split_users(
    users: npt.NDArray[np.int64], excluded: npt.NDArray[np.int64], share: float, seed: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Drop `excluded` users, then split the rest into (first, second) with `share` second."""
    pool = np.setdiff1d(np.unique(users), excluded)
    rng = np.random.default_rng(seed)
    second = np.sort(rng.choice(pool, size=round(pool.size * share), replace=False))
    return np.setdiff1d(pool, second), second


def _matrix(df: pl.DataFrame) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int8], list[int]]:
    """Feature matrix, labels, and group sizes (df must be sorted by user)."""
    groups = df.group_by("user_id", maintain_order=True).len()["len"].to_list()
    x = df.select(FEATURES).to_numpy().astype(np.float32)
    return x, df["label"].to_numpy().astype(np.int8), groups


def ndcg_by_user(df: pl.DataFrame, score_col: str, k: int = 10) -> FloatArray:
    """Per-user NDCG@k of candidates ordered by `score_col`, relative to all relevant items."""
    ranked = df.sort(
        ["user_id", score_col, "retrieval_rank"],
        descending=[False, True, False],
        maintain_order=True,
    )
    per_user = ranked.group_by("user_id", maintain_order=True).agg(
        labels=pl.col("label").head(k), n_rel=pl.col("n_relevant").first()
    )
    hits = np.zeros((per_user.height, k), dtype=bool)
    for i, labels in enumerate(per_user["labels"].to_list()):
        hits[i, : len(labels)] = np.asarray(labels, dtype=bool)
    n_rel = per_user["n_rel"].to_numpy().astype(np.int64)
    out: FloatArray = metrics.ndcg_at_k(hits, n_rel, k).astype(np.float64)
    return out


def _train(
    params: dict[str, Any], train: pl.DataFrame, valid: pl.DataFrame, cfg: RankerConfig
) -> lgb.Booster:
    xt, yt, gt = _matrix(train)
    xv, yv, gv = _matrix(valid)
    dtrain = lgb.Dataset(xt, yt, group=gt, feature_name=list(FEATURES))
    dvalid = lgb.Dataset(xv, yv, group=gv, reference=dtrain)
    full = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "verbosity": -1,
        "seed": cfg.seed,
        "deterministic": True,
        "force_row_wise": True,
        **params,
    }
    return lgb.train(
        full,
        dtrain,
        num_boost_round=cfg.max_rounds,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
    )


def tune(train: pl.DataFrame, valid: pl.DataFrame, cfg: RankerConfig) -> dict[str, Any]:
    """Time-boxed Optuna search over LightGBM parameters on the inner validation users."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        booster = _train(params, train, valid, cfg)
        trial.set_user_attr("best_iteration", booster.best_iteration)
        return float(booster.best_score["valid_0"]["ndcg@10"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg.seed)
    )
    study.optimize(objective, n_trials=cfg.n_trials, timeout=cfg.timeout_minutes * 60)
    best = dict(study.best_params) | {"bagging_freq": 1}
    log.info(
        "tuned", extra={"trials": len(study.trials), "best_value": study.best_value, "params": best}
    )
    return best


def shap_importance(booster: lgb.Booster, x: npt.NDArray[np.float32]) -> dict[str, float]:
    """Mean absolute SHAP value per feature (LightGBM's built-in TreeSHAP)."""
    contrib = booster.predict(x, pred_contrib=True)
    values = np.abs(np.asarray(contrib)[:, :-1]).mean(axis=0)  # last column is the bias
    return dict(sorted(zip(FEATURES, values.tolist(), strict=True), key=lambda kv: -kv[1]))


def plot_shap(importance: dict[str, float], path: Path, top: int = 15) -> None:
    """Horizontal bar chart of the top features by mean |SHAP|."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    items = list(importance.items())[:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(items) + 1.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#4c72b0")
    ax.set_xlabel("mean |SHAP value| (ranker score units)")
    ax.set_title("Ranker feature importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(
    features_path: Path,
    out_dir: Path,
    cfg: RankerConfig,
    *,
    n_resamples: int = 1000,
    doc_dir: Path | None = None,
) -> dict[str, Any]:
    """Tune, train, and evaluate the ranker; write the model and a summary to `out_dir`."""
    df = pl.read_parquet(features_path).sort(["user_id", "retrieval_rank"])
    # Users whose labels early-stopped the retrieval model, carried from its artifacts.
    excluded = np.load(features_path.parent / "early_stop_users.npy")
    users = df["user_id"].to_numpy()
    train_users, eval_users = split_users(users, excluded, cfg.eval_share, cfg.seed)
    fit_users, inner_users = split_users(
        train_users, np.zeros(0, np.int64), cfg.inner_val_share, cfg.seed + 1
    )
    by_users = {
        name: df.filter(pl.col("user_id").is_in(u.tolist()))
        for name, u in (("fit", fit_users), ("inner", inner_users), ("eval", eval_users))
    }
    begin_stage(out_dir)
    t0 = time.perf_counter()
    params = tune(by_users["fit"], by_users["inner"], cfg)
    booster = _train(params, by_users["fit"], by_users["inner"], cfg)
    fit_s = time.perf_counter() - t0

    ev = by_users["eval"]
    x_eval, _, _ = _matrix(ev)
    ev = ev.with_columns(ranker_score=pl.Series(np.asarray(booster.predict(x_eval))))
    ranked = ndcg_by_user(ev, "ranker_score")
    retrieval = ndcg_by_user(ev.with_columns(neg_rank=-pl.col("retrieval_rank")), "neg_rank")
    lift = metrics.paired_difference(ranked, retrieval, n_resamples=n_resamples, seed=cfg.seed)
    importance = shap_importance(booster, x_eval[: min(50_000, x_eval.shape[0])])
    summary: dict[str, Any] = {
        "n_users": {k: int(v["user_id"].n_unique()) for k, v in by_users.items()},
        "n_excluded_users": int(np.intersect1d(np.unique(users), excluded).size),
        "params": params,
        "best_iteration": booster.best_iteration,
        "ndcg@10_ranker": asdict(metrics.bootstrap_mean(ranked, n_resamples, seed=cfg.seed)),
        "ndcg@10_retrieval": asdict(metrics.bootstrap_mean(retrieval, n_resamples, seed=cfg.seed)),
        "ndcg@10_lift": asdict(lift),
        "shap_importance": importance,
        "fit_seconds": round(fit_s, 1),
    }
    booster.save_model(str(out_dir / "ranker.txt"), num_iteration=booster.best_iteration)
    # Users the ranker never trained or tuned on: later stages (the online test simulation)
    # must evaluate on these only.
    np.save(out_dir / "eval_users.npy", eval_users)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if doc_dir is not None:
        doc_dir.mkdir(parents=True, exist_ok=True)
        plot_shap(importance, doc_dir / "ranker-shap.png")
    write_manifest(
        out_dir,
        stage="train_ranker",
        config={"ranker": asdict(cfg), "features": list(FEATURES), "input": str(features_path)},
        data_hash=str(verify_outputs(features_path.parent)["data_hash"]),
        seed=cfg.seed,
        outputs=[out_dir / "ranker.txt", out_dir / "summary.json", out_dir / "eval_users.npy"],
    )
    if mlflow.active_run() is not None:
        mlflow.log_params({f"lgb_{k}": v for k, v in params.items()})
        mlflow.log_metrics(
            {
                "ndcg_at_10_ranker": summary["ndcg@10_ranker"]["mean"],
                "ndcg_at_10_retrieval": summary["ndcg@10_retrieval"]["mean"],
                "ndcg_at_10_lift": lift.mean,
                "ndcg_at_10_lift_low": lift.low,
                "ndcg_at_10_lift_high": lift.high,
            }
        )
        mlflow.log_artifacts(str(out_dir))
    return summary


def main(argv: list[str] | None = None) -> None:
    """Tune and train the LambdaMART ranker and report NDCG@10 lift over retrieval order."""
    settings = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument(
        "--features",
        type=Path,
        default=settings.artifacts_dir / "ranker_features" / "ranker_features_val.parquet",
    )
    p.add_argument("--out-dir", type=Path, default=settings.artifacts_dir / "ranker")
    p.add_argument("--doc-dir", type=Path, default=Path("docs"))
    p.add_argument("--n-trials", type=int, default=RankerConfig.n_trials)
    p.add_argument("--timeout-minutes", type=float, default=RankerConfig.timeout_minutes)
    p.add_argument("--experiment", default="ranker")
    args = p.parse_args(argv)
    configure_logging(settings.log_level)
    cfg = RankerConfig(
        n_trials=args.n_trials, timeout_minutes=args.timeout_minutes, seed=settings.seed
    )
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="lambdamart"):
        summary = run(args.features, args.out_dir, cfg, doc_dir=args.doc_dir)
    log.info("done", extra={"lift": summary["ndcg@10_lift"]})


if __name__ == "__main__":
    main()
