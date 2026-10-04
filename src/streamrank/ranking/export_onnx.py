"""Convert the LightGBM ranker to ONNX and check its scores match LightGBM.

Serving runs both models with ONNX Runtime, which uses its own thread pool instead of
OpenMP, so the API process can load FAISS without LightGBM's OpenMP runtime.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import onnxmltools
import polars as pl
from onnxmltools.convert.common.data_types import FloatTensorType

from streamrank.common.config import get_settings
from streamrank.ranking.features import FEATURES

TARGET_OPSET = 15  # highest opset the LightGBM converter supports
MAX_ABS_DIFF = 1e-5


def float32_thresholds(booster: lgb.Booster) -> lgb.Booster:
    """The same model with every split threshold rounded down to a float32 value.

    ONNX tree ensembles store thresholds as float32, and the converter rounds LightGBM's
    double thresholds to the nearest float32. A float32 feature equal to a threshold that
    rounded up would then branch the other way. Rounding each threshold t down to the
    largest float32 not above t changes nothing for float32 inputs (no float32 value lies
    between the two), and the float32 copy is exact.
    """
    lines = booster.model_to_string().splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("threshold="):
            continue
        values = np.array(line.split("=", 1)[1].split(), dtype=np.float64)
        f32 = values.astype(np.float32)
        above = f32.astype(np.float64) > values
        f32[above] = np.nextafter(f32[above], np.float32(-np.inf))
        lines[i] = "threshold=" + " ".join(repr(float(v)) for v in f32)
    # Tree byte offsets no longer match the rewritten text; LightGBM parses trees in order
    # when they are absent.
    lines = [line for line in lines if not line.startswith("tree_sizes=")]
    return lgb.Booster(model_str="\n".join(lines) + "\n")


def convert(booster: lgb.Booster, path: Path) -> Path:
    """Write the booster as an ONNX tree ensemble with one float input `features`."""
    model = onnxmltools.convert_lightgbm(
        float32_thresholds(booster),
        initial_types=[("features", FloatTensorType([None, len(FEATURES)]))],
        target_opset=TARGET_OPSET,
        zipmap=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(model.SerializeToString())
    return path


def max_difference(booster: lgb.Booster, path: Path, x: npt.NDArray[np.float32]) -> float:
    """Largest absolute score difference between LightGBM and ONNX Runtime on `x`."""
    import onnxruntime as ort  # noqa: PLC0415

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = np.asarray(sess.run(None, {"features": x})[0]).ravel()
    want = np.asarray(booster.predict(x, num_threads=1)).ravel()
    return float(np.abs(got - want).max())


def main(argv: list[str] | None = None) -> None:
    """Convert the trained ranker to ONNX and verify its scores."""
    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--ranker-dir", type=Path, default=s.artifacts_dir / "ranker")
    p.add_argument(
        "--features",
        type=Path,
        default=s.artifacts_dir / "ranker_features" / "ranker_features_val.parquet",
    )
    p.add_argument("--check-rows", type=int, default=50_000)
    args = p.parse_args(argv)
    booster = lgb.Booster(model_file=str(args.ranker_dir / "ranker.txt"))
    path = convert(booster, args.ranker_dir / "ranker.onnx")
    x = pl.read_parquet(args.features).select(FEATURES).head(args.check_rows).to_numpy()
    worst = max_difference(booster, path, x.astype(np.float32))
    report = {"onnx": str(path), "max_abs_diff": worst, "checked_rows": int(x.shape[0])}
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if worst > MAX_ABS_DIFF:
        raise SystemExit(f"ONNX ranker differs from LightGBM by {worst:.2e}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
