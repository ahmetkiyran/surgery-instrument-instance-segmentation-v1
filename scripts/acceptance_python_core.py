"""Direct, bounded Python-core smoke runner for the Windows acceptance script.

It deliberately uses the existing skeleton pipeline API; no algorithm code lives here.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

from surgical_pipeline.cli import _default_pose_path
from surgical_pipeline.config import load_config
from surgical_pipeline.model_loader import load_validated_yolo_models, resolve_model_file_paths
from surgical_pipeline.pose_estimator import PoseEstimator
from surgical_pipeline.pose_pipeline import run_pose_pilot
from surgical_pipeline.video_io import read_metadata


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-inventory", type=Path, required=True)
    parser.add_argument("--model-paths", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--enable-depth", action="store_true")
    return parser.parse_args()


def inventory(root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    health, instrument = resolve_model_file_paths(root, None, None, auto_download=False)
    pose = _default_pose_path(root)
    if not pose.is_file():
        raise FileNotFoundError(f"Pose model is missing: {pose.name}")
    health_model, instrument_model, health_info, instrument_info, health_class_id = load_validated_yolo_models(health, instrument)
    pose_info = PoseEstimator(pose, device="cpu", fp16=False).load()
    # The loaded models are intentionally discarded before the actual pipeline process.
    del health_model, instrument_model
    details = {
        "status": "PASSED",
        "health": {"filename": health.name, "sha256": health_info.sha256, "task": health_info.task, "classes": health_info.names},
        "instrument": {"filename": instrument.name, "sha256": instrument_info.sha256, "task": instrument_info.task, "classes": instrument_info.names},
        "pose": {"filename": pose.name, "sha256": pose_info.sha256, "task": pose_info.task, "classes": pose_info.names, "keypoint_shape": pose_info.keypoint_shape},
        "health_class_id": health_class_id,
        "distinct_model_hashes": len({health_info.sha256, instrument_info.sha256, pose_info.sha256}) == 3,
    }
    if not details["distinct_model_hashes"]:
        raise RuntimeError("Model roles do not resolve to three distinct weights.")
    return health, instrument, pose, details


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    health, instrument, pose, details = inventory(root)
    write_json(args.model_inventory, details)
    write_json(args.model_paths, {"health": str(health), "instrument": str(instrument), "pose": str(pose)})
    if not args.input:
        return 0
    if not args.output or not args.result:
        raise ValueError("--output and --result are required with --input")
    source = args.input.resolve()
    metadata = read_metadata(source)
    started = time.time()
    config = load_config(
        root,
        {
            "runtime": {"output_dir": str(args.output.resolve()), "save_audio": False, "max_video_seconds": 10},
            "depth": {"enabled": bool(args.enable_depth)},
            "cli": {"privacy_mode": "skeleton-only"},
        },
    )
    result = run_pose_pilot(
        source,
        health,
        instrument,
        args.output,
        pose,
        config,
        max_seconds=min(10.0, metadata.duration_s),
    )
    payload = {
        "status": "PASSED",
        "source_filename": source.name,
        "input_duration_seconds": metadata.duration_s,
        "processed_video": result.skeleton_video.name,
        "run_directory": result.run_dir.name,
        "elapsed_seconds": round(time.time() - started, 3),
        "artifacts": [item.name for item in result.files if item.is_file()],
        "privacy_report": json.loads(result.privacy_report.read_text(encoding="utf-8")),
        "manifest": json.loads(result.manifest.read_text(encoding="utf-8")),
    }
    write_json(args.result, payload)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
