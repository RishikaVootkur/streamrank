"""Train the two-tower model, evaluate it on the full catalog, and log everything to MLflow.

`TwoTowerRecommender` implements the same `fit` / `score` interface as the baselines, so it
is evaluated by the shared `fit_and_evaluate` path. The CLI also fits the best baseline
(EASE) on the same data and reports the paired per-user difference in Recall@100.
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.sparse as sp
import torch

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import begin_stage, verify_outputs, write_manifest
from streamrank.common.seeds import set_global_seed
from streamrank.eval import metrics
from streamrank.eval.dataset import EvalSetup, EvalTargets, TrainData, load_setup
from streamrank.eval.evaluate import EvalResult, evaluate_lists, recommend
from streamrank.models.baselines import EASE
from streamrank.models.sequences import (
    Batch,
    ItemContent,
    Sequences,
    build_item_content,
    build_sequences,
    query_batch,
    training_batch,
)
from streamrank.models.two_tower import (
    TensorBatch,
    TwoTower,
    TwoTowerConfig,
    cosine_lr,
    count_parameters,
    pick_device,
)

log = logging.getLogger(__name__)
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
EARLY_STOP_METRIC = "recall@100"


@dataclass(frozen=True)
class TrainConfig:
    """Optimization settings."""

    epochs: int = 8
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0
    patience: int = 2
    recent_window_prob: float = 0.0  # share of training windows ending at the latest event
    time_limit_minutes: float = 85.0
    seed: int = 42


def to_device(batch: Batch, device: torch.device) -> TensorBatch:
    def t(a: npt.NDArray[Any]) -> torch.Tensor:
        return torch.from_numpy(a).to(device)

    return TensorBatch(
        t(batch.inputs),
        t(batch.input_positive),
        t(batch.input_gaps),
        t(batch.targets),
        t(batch.target_mask),
    )


def _recall_at_100(model: "TwoTowerRecommender", targets: EvalTargets) -> float:
    recs = recommend(model, targets.user_rows, targets.exclude, 100)
    hits = metrics.hit_matrix(recs, targets.relevant)
    n_rel = np.diff(targets.relevant.indptr).astype(np.int64)
    return float(metrics.recall_at_k(hits, n_rel, 100).mean())


@dataclass
class TwoTowerRecommender:
    """Fit and score a two-tower model on `TrainData` (see the module docstring)."""

    model_cfg: TwoTowerConfig
    train_cfg: TrainConfig
    movies: pl.DataFrame
    tags: pl.DataFrame
    early_stop_targets: EvalTargets | None = None
    device: torch.device = field(default_factory=pick_device)
    name: str = "two_tower"
    history: list[dict[str, float]] = field(default_factory=list)
    model: TwoTower | None = None
    content: ItemContent | None = None
    _seqs: Sequences | None = None
    _items: torch.Tensor | None = None

    def fit(self, train: TrainData) -> None:
        cfg, tcfg = self.model_cfg, self.train_cfg
        set_global_seed(tcfg.seed)
        torch.manual_seed(tcfg.seed)
        rng = np.random.default_rng(tcfg.seed)
        gen = torch.Generator(device=self.device).manual_seed(tcfg.seed)
        self._seqs = seqs = build_sequences(train)
        self.content = build_item_content(
            train.item_ids, self.movies, self.tags, cutoff_ts=train.cutoff_ts
        )
        counts = torch.from_numpy(np.asarray(train.positives.sum(axis=0)).ravel()) + 1.0
        model = TwoTower(train.n_items, self.content, counts, cfg).to(self.device)
        self.model = model
        trainable = np.flatnonzero(seqs.lengths() >= 2)  # noqa: PLR2004 - input and target
        steps_per_epoch = -(-trainable.size // tcfg.batch_size)
        total = steps_per_epoch * tcfg.epochs
        opt, sched = _optimizer(model, tcfg, total)
        log.info(
            "training",
            extra={
                "params": count_parameters(model),
                "steps": total,
                "users": int(trainable.size),
                "device": str(self.device),
            },
        )
        best, best_state, bad = -1.0, None, 0
        start = time.perf_counter()
        for epoch in range(tcfg.epochs):
            losses = self._train_epoch(
                trainable, n_batches=steps_per_epoch, opt=opt, sched=sched, rng=rng, gen=gen
            )
            record = {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)),
                "minutes": (time.perf_counter() - start) / 60,
            }
            self._items = None
            if self.early_stop_targets is not None:
                record[f"val_{EARLY_STOP_METRIC}"] = _recall_at_100(self, self.early_stop_targets)
            self.history.append(record)
            log.info("epoch", extra=record)
            if mlflow.active_run() is not None:
                mlflow.log_metrics(
                    {_metric_name(k): v for k, v in record.items() if k != "epoch"},
                    step=epoch + 1,
                )
            score = record.get(f"val_{EARLY_STOP_METRIC}", -record["loss"])
            if score > best:
                best, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad > tcfg.patience or record["minutes"] > tcfg.time_limit_minutes:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._items = None

    def _train_epoch(
        self,
        users: IntArray,
        *,
        n_batches: int,
        opt: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler.LRScheduler,
        rng: np.random.Generator,
        gen: torch.Generator,
    ) -> list[float]:
        if self.model is None or self._seqs is None:
            raise RuntimeError("model not initialized")
        self.model.train()
        losses = []
        for batch_users in np.array_split(rng.permutation(users), n_batches):
            batch = training_batch(
                self._seqs,
                batch_users,
                self.model_cfg.max_len,
                rng,
                recent_prob=self.train_cfg.recent_window_prob,
            )
            if not batch.target_mask.any():
                continue
            loss = self.model.loss(to_device(batch, self.device), generator=gen)
            opt.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            opt.step()
            sched.step()
            losses.append(loss.item())
        return losses

    def item_vectors(self) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("call fit() first")
        if self._items is None:
            self.model.eval()
            with torch.no_grad():
                self._items = self.model.all_items()
        return self._items

    def user_vectors(self, user_rows: IntArray, batch_size: int = 1024) -> torch.Tensor:
        if self.model is None or self._seqs is None:
            raise RuntimeError("call fit() first")
        self.model.eval()
        parts = []
        with torch.no_grad():
            for s in range(0, user_rows.size, batch_size):
                q = to_device(
                    query_batch(self._seqs, user_rows[s : s + batch_size], self.model_cfg.max_len),
                    self.device,
                )
                parts.append(self.model.last_position(q.tokens, q.positive, q.gaps))
        return torch.cat(parts) if parts else torch.zeros(0, self.model_cfg.dim)

    def score(self, user_rows: IntArray) -> FloatArray:
        scores = self.user_vectors(user_rows) @ self.item_vectors().T
        out: FloatArray = scores.float().cpu().numpy()
        return out


def _optimizer(
    model: TwoTower, cfg: TrainConfig, total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """AdamW with weight decay on dense weights only, plus warmup and cosine decay."""
    named = list(model.named_parameters())
    decay = [p for n, p in named if p.ndim > 1 and "emb" not in n]
    other = [p for n, p in named if not (p.ndim > 1 and "emb" not in n)]
    opt = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": other, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: cosine_lr(s, total_steps, cfg.warmup_steps)
    )
    return opt, sched


def load_recommender(
    model_dir: Path,
    train: TrainData,
    movies: pl.DataFrame,
    tags: pl.DataFrame,
    device: torch.device | None = None,
) -> TwoTowerRecommender:
    """Rebuild a trained recommender from `model_dir` for the same training data.

    Sequences and item content are recomputed from `train`; the weights come from
    `model.pt` and the hyperparameters from the manifest.
    """
    manifest = verify_outputs(model_dir)
    model_cfg = TwoTowerConfig(**manifest["config"]["model"])
    train_cfg = TrainConfig(**manifest["config"]["train"])
    rec = TwoTowerRecommender(model_cfg, train_cfg, movies, tags, device=device or pick_device())
    rec._seqs = build_sequences(train)
    rec.content = build_item_content(train.item_ids, movies, tags, cutoff_ts=train.cutoff_ts)
    counts = torch.from_numpy(np.asarray(train.positives.sum(axis=0)).ravel()) + 1.0
    model = TwoTower(train.n_items, rec.content, counts, model_cfg)
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    rec.model = model.to(rec.device).eval()
    return rec


def split_targets(
    targets: EvalTargets, user_ids: IntArray, train: TrainData
) -> tuple[EvalTargets, EvalTargets]:
    """Split targets into the users whose IDs are in `user_ids` and everyone else."""
    mask = np.isin(train.user_ids[targets.user_rows], user_ids)

    def pick(m: npt.NDArray[np.bool_]) -> EvalTargets:
        return EvalTargets(
            user_rows=targets.user_rows[m],
            relevant=sp.csr_array(targets.relevant[m]),
            exclude=sp.csr_array(targets.exclude[m]),
            n_cold_users=targets.n_cold_users,
        )

    return pick(mask), pick(~mask)


def early_stop_users(setup: EvalSetup, early_stop_split: Path | None, seed: int) -> IntArray:
    """User IDs used for early stopping; metrics are also reported on everyone else.

    With `early_stop_split`, its validation users (for the full split, the 10% sample whose
    validation users also tuned the baselines). Otherwise a seeded half of the users.
    """
    if early_stop_split is not None:
        ids: IntArray = (
            pl.read_parquet(early_stop_split / "val.parquet")["user_id"].unique().to_numpy()
        )
        return ids
    users = setup.train.user_ids[setup.targets.user_rows]
    rng = np.random.default_rng(seed)
    out: IntArray = np.sort(rng.choice(users, size=users.size // 2, replace=False))
    return out


def run(
    split_dir: Path,
    out_dir: Path,
    model_cfg: TwoTowerConfig,
    train_cfg: TrainConfig,
    *,
    early_stop_split: Path | None = None,
    compare_ease: bool = True,
    n_resamples: int = 1000,
    run_name: str = "two_tower",
) -> dict[str, Any]:
    """Train on `split_dir` (validation partition), evaluate, and write results to `out_dir`."""
    settings = get_settings()
    setup = load_setup(split_dir, "val")
    processed = settings.data_dir / "processed"
    movies = pl.read_parquet(processed / "movies.parquet")
    tags = pl.read_parquet(processed / "tags.parquet")
    es_ids = early_stop_users(setup, early_stop_split, train_cfg.seed)
    es_targets, heldout_targets = split_targets(setup.targets, es_ids, setup.train)
    heldout = EvalSetup(train=setup.train, targets=heldout_targets, partition="val")
    begin_stage(out_dir)
    rec = TwoTowerRecommender(model_cfg, train_cfg, movies, tags, early_stop_targets=es_targets)

    t0 = time.perf_counter()
    rec.fit(setup.train)
    fit_s = time.perf_counter() - t0
    recs = recommend(rec, setup.targets.user_rows, setup.targets.exclude, 200)
    result = evaluate_lists(
        rec.name,
        {**asdict(model_cfg), **asdict(train_cfg)},
        recs,
        setup,
        n_resamples=n_resamples,
        seed=train_cfg.seed,
        fit_seconds=fit_s,
        recommend_seconds=time.perf_counter() - t0 - fit_s,
    )
    keep = np.isin(setup.targets.user_rows, heldout_targets.user_rows)
    held_result = evaluate_lists(
        rec.name,
        result.params,
        recs[keep],
        heldout,
        n_resamples=n_resamples,
        seed=train_cfg.seed,
    )
    summary: dict[str, Any] = {
        "run_name": run_name,
        "n_early_stop_users": int(es_targets.user_rows.size),
        "result": result.row(),
        "heldout": held_result.row(),
        "history": rec.history,
    }
    if compare_ease:
        ease_all, ease_held = _ease_results(setup, heldout, keep, n_resamples, train_cfg.seed)
        summary["ease"] = ease_all.row()
        summary["ease_heldout"] = ease_held.row()
        for label, ours, theirs in (("", result, ease_all), ("_heldout", held_result, ease_held)):
            diff = metrics.paired_difference(
                ours.per_user["recall@100"],
                theirs.per_user["recall@100"],
                n_resamples=n_resamples,
                seed=train_cfg.seed,
            )
            summary[f"recall@100_minus_ease{label}"] = asdict(diff)
    _save(out_dir, rec, summary, split_dir, setup.train.user_ids[es_targets.user_rows])
    if mlflow.active_run() is not None:
        mlflow.log_params({k: v for k, v in summary["result"]["params"].items()})
        mlflow.log_metrics(_flat_metrics(summary))
        mlflow.log_artifacts(str(out_dir))
    return summary


def _ease_results(
    setup: EvalSetup, heldout: EvalSetup, keep: npt.NDArray[np.bool_], n_resamples: int, seed: int
) -> tuple[EvalResult, EvalResult]:
    """EASE (tuned setting) on all users and on the held-out users, from one fit."""
    params = {"l2": 2000.0, "signal": "positive"}
    model = EASE(l2=2000.0, signal="positive")
    t0 = time.perf_counter()
    model.fit(setup.train)
    fit_s = time.perf_counter() - t0
    recs = recommend(model, setup.targets.user_rows, setup.targets.exclude, 200)
    full = evaluate_lists(
        model.name, params, recs, setup, n_resamples=n_resamples, seed=seed, fit_seconds=fit_s
    )
    held = evaluate_lists(
        model.name, params, recs[keep], heldout, n_resamples=n_resamples, seed=seed
    )
    return full, held


def _metric_name(name: str) -> str:
    """MLflow metric names cannot contain '@'."""
    return name.replace("@", "_at_")


def _flat_metrics(summary: dict[str, Any]) -> dict[str, float]:
    out = {}
    for section, prefix in (("result", ""), ("heldout", "heldout_")):
        for name, cell in summary[section].items():
            if isinstance(cell, dict) and "mean" in cell:
                out[prefix + _metric_name(name)] = float(cell["mean"])
    for key in ("recall@100_minus_ease", "recall@100_minus_ease_heldout"):
        if key in summary:
            out |= {f"{_metric_name(key)}_{k}": float(v) for k, v in summary[key].items()}
    return out


def _save(
    out_dir: Path,
    rec: TwoTowerRecommender,
    summary: dict[str, Any],
    split: Path,
    early_stop_ids: IntArray,
) -> None:
    if rec.model is None or rec.content is None:
        raise RuntimeError("model not trained")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    torch.save(rec.model.state_dict(), out_dir / "model.pt")
    np.save(out_dir / "item_vectors.npy", rec.item_vectors().cpu().numpy())
    # Users whose validation labels selected the epoch: later stages must not evaluate on them.
    np.save(out_dir / "early_stop_users.npy", np.asarray(early_stop_ids, dtype=np.int64))
    write_manifest(
        out_dir,
        stage="train_retrieval",
        config={
            "model": asdict(rec.model_cfg),
            "train": asdict(rec.train_cfg),
            "split": str(split),
        },
        data_hash=str(verify_outputs(split)["data_hash"]),
        seed=rec.train_cfg.seed,
        outputs=[
            out_dir / "summary.json",
            out_dir / "model.pt",
            out_dir / "item_vectors.npy",
            out_dir / "early_stop_users.npy",
        ],
    )


def main(argv: list[str] | None = None) -> None:
    """Train and evaluate the two-tower retrieval model."""
    settings = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--split-dir", type=Path, default=settings.data_dir / "split" / "sample10")
    p.add_argument(
        "--early-stop-split",
        type=Path,
        default=None,
        help="early-stop on the validation users of this (smaller) split",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-name", default="two_tower")
    p.add_argument("--experiment", default="retrieval")
    for f, default in asdict(TwoTowerConfig()).items():
        if isinstance(default, bool):
            p.add_argument(
                f"--{f.replace('_', '-')}", type=lambda s: s.lower() == "true", default=default
            )
        else:
            p.add_argument(f"--{f.replace('_', '-')}", type=type(default), default=default)
    for f, default in asdict(TrainConfig(seed=settings.seed)).items():
        p.add_argument(f"--{f.replace('_', '-')}", type=type(default), default=default)
    p.add_argument("--no-ease", action="store_true", help="skip the paired EASE comparison")
    p.add_argument("--n-resamples", type=int, default=1000)
    args = vars(p.parse_args(argv))
    configure_logging(settings.log_level)
    model_cfg = TwoTowerConfig(**{k: args[k] for k in asdict(TwoTowerConfig())})
    train_cfg = TrainConfig(**{k: args[k] for k in asdict(TrainConfig())})
    out_dir = args["out_dir"] or settings.artifacts_dir / "retrieval" / args["run_name"]
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(args["experiment"])
    with mlflow.start_run(run_name=args["run_name"]):
        summary = run(
            args["split_dir"],
            out_dir,
            model_cfg,
            train_cfg,
            early_stop_split=args["early_stop_split"],
            compare_ease=not args["no_ease"],
            n_resamples=args["n_resamples"],
            run_name=args["run_name"],
        )
    log.info(
        "done",
        extra={
            "heldout_recall@100": summary["heldout"]["recall@100"],
            "heldout_vs_ease": summary.get("recall@100_minus_ease_heldout"),
        },
    )


if __name__ == "__main__":
    main()
