"""Export the trained user tower to ONNX and check it matches PyTorch.

The exported graph takes left-padded `tokens`, `positive`, and `gaps` (int64, shape
[batch, length]) and returns the L2-normalized user vector at the last position.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import polars as pl
import torch
from torch import Tensor, nn

from streamrank.common.config import get_settings
from streamrank.eval.dataset import load_setup
from streamrank.models.sequences import query_batch
from streamrank.models.train_retrieval import TwoTowerRecommender, load_recommender
from streamrank.models.two_tower import TwoTower

INPUTS = ("tokens", "positive", "gaps")
MAX_ABS_DIFF = 1e-4  # parity tolerance between PyTorch and ONNX Runtime


class UserTowerExport(nn.Module):
    """Wraps the user tower so the graph has plain int64 inputs and one output."""

    def __init__(self, model: TwoTower) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: Tensor, positive: Tensor, gaps: Tensor) -> Tensor:
        return self.model.last_position(tokens, positive > 0, gaps)


def export(rec: TwoTowerRecommender, path: Path) -> Path:
    """Write the user tower as a single self-contained ONNX file."""
    if rec.model is None:
        raise RuntimeError("model not trained")
    module = UserTowerExport(rec.model.cpu().eval()).eval()
    length = rec.model_cfg.max_len
    example = (
        torch.randint(1, rec.model.n_items + 1, (2, length)),
        torch.randint(0, 2, (2, length)),
        torch.randint(1, 30, (2, length)),
    )
    batch = torch.export.Dim("batch", min=1, max=4096)
    path.parent.mkdir(parents=True, exist_ok=True)
    program = torch.onnx.export(
        module,
        example,
        dynamo=True,
        input_names=list(INPUTS),
        output_names=["user_vector"],
        dynamic_shapes={name: {0: batch} for name in INPUTS},
        optimize=True,
    )
    if program is None:
        raise RuntimeError("ONNX export returned no program")
    program.save(str(path), external_data=False)
    return path


def session(path: Path, threads: int = 1) -> ort.InferenceSession:
    """A low-latency CPU session: sequential execution, few threads, no spinning."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def parity(rec: TwoTowerRecommender, path: Path, user_rows: np.ndarray) -> float:
    """Largest absolute difference between PyTorch and ONNX Runtime user vectors."""
    if rec._seqs is None or rec.model is None:
        raise RuntimeError("model not trained")
    q = query_batch(rec._seqs, user_rows, rec.model_cfg.max_len)
    feeds = {
        "tokens": q.inputs,
        "positive": q.input_positive.astype(np.int64),
        "gaps": q.input_gaps,
    }
    got = session(path).run(None, feeds)[0]
    rec.model.cpu().eval()
    with torch.no_grad():
        want = rec.model.last_position(
            torch.from_numpy(q.inputs),
            torch.from_numpy(q.input_positive),
            torch.from_numpy(q.input_gaps),
        ).numpy()
    return float(np.abs(got - want).max())


def main(argv: list[str] | None = None) -> None:
    """Export the user tower to ONNX and verify parity with PyTorch."""
    settings = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, default=settings.data_dir / "split" / "full")
    p.add_argument("--out", type=Path, default=settings.artifacts_dir / "onnx" / "user_tower.onnx")
    p.add_argument("--check-users", type=int, default=512)
    args = p.parse_args(argv)
    setup = load_setup(args.split_dir, "val")
    processed = settings.data_dir / "processed"
    rec = load_recommender(
        args.model_dir,
        setup.train,
        pl.read_parquet(processed / "movies.parquet"),
        pl.read_parquet(processed / "tags.parquet"),
        torch.device("cpu"),
    )
    path = export(rec, args.out)
    worst = parity(rec, path, setup.targets.user_rows[: args.check_users])
    report = {"onnx": str(path), "max_abs_diff": worst, "checked_users": args.check_users}
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if worst > MAX_ABS_DIFF:
        raise SystemExit(f"ONNX output differs from PyTorch by {worst:.2e}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
