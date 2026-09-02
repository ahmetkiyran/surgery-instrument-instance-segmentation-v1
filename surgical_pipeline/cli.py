"""Command-line entry point for preflight and local batch analysis."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

from .config import load_config
from .doctor import print_doctor, run_doctor
from .model_manager import ModelManager, ModelManagerError


def _path(value: str) -> Path:
    return Path(value).expanduser()


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="python -m surgical_pipeline", description="Yerel cerrahi video analiz sistemi")
    subcommands = command.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser("doctor", help="Kurulum ve model ön kontrolü")
    doctor.add_argument("--health-model", type=_path); doctor.add_argument("--instrument-model", type=_path)
    models = subcommands.add_parser("models", help="GitHub Release model ağırlıklarını yönet")
    models_subcommands = models.add_subparsers(dest="models_command", required=True)
    for name in ("list", "download", "verify", "redownload"):
        models_subcommands.add_parser(name)
    analyze = subcommands.add_parser("analyze", help="Bir videoyu yerelde analiz et")
    analyze.add_argument("--input", required=True, type=_path); analyze.add_argument("--health-model", type=_path); analyze.add_argument("--instrument-model", type=_path)
    analyze.add_argument("--output-dir", type=_path, default=None); analyze.add_argument("--blur-kernel", type=int, default=None); analyze.add_argument("--confidence", type=float, default=None); analyze.add_argument("--iou", type=float, default=None); analyze.add_argument("--tracker", choices=["botsort", "bytetrack"], default=None); analyze.add_argument("--depth-stride", type=int, default=None); analyze.add_argument("--disable-depth", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path.cwd()
    overrides: dict = {}
    if args.command == "analyze":
        runtime: dict = {}; privacy: dict = {}; tracking: dict = {}; depth: dict = {}
        if args.output_dir: runtime["output_dir"] = str(args.output_dir)
        if args.blur_kernel: privacy["blur_kernel"] = args.blur_kernel
        if args.confidence is not None:
            tracking["confidence"] = args.confidence
            tracking["health"] = {"confidence": args.confidence}
            tracking["instrument"] = {"confidence": args.confidence}
        if args.iou is not None:
            tracking["iou"] = args.iou
            tracking.setdefault("health", {})["iou"] = args.iou
            tracking.setdefault("instrument", {})["iou"] = args.iou
        if args.tracker: tracking["algorithm"] = args.tracker
        if args.depth_stride: depth["depth_stride"] = args.depth_stride
        if args.disable_depth: depth["enabled"] = False
        overrides = {key: value for key, value in {"runtime": runtime, "privacy": privacy, "tracking": tracking, "depth": depth}.items() if value}
    config = load_config(root, overrides)
    if args.command == "doctor":
        return print_doctor(run_doctor(root, config, args.health_model, args.instrument_model))
    if args.command == "models":
        try:
            manager = ModelManager(root)
            if args.models_command in {"list", "verify"}:
                statuses = manager.statuses()
                for status in statuses:
                    print(f"{'[OK]' if status.ready else '[MISSING]'} {status.key}: {status.detail}")
                return 0 if args.models_command == "list" or all(status.ready for status in statuses) else 2

            def download_progress(key: str, current: int, total: int | None) -> None:
                if total:
                    print(f"\r{key}: %{current / total * 100:5.1f}", end="", flush=True)
                else:
                    print(f"\r{key}: {current / (1024 * 1024):.1f} MiB", end="", flush=True)

            manager.download_all(force=args.models_command == "redownload", progress=download_progress)
            print("\nModeller SHA-256, task ve sınıf bilgileriyle doğrulandı.")
            return 0
        except ModelManagerError as error:
            print(f"Model işlemi başarısız: {error}")
            return 2
    checks = run_doctor(root, config, args.health_model, args.instrument_model)
    critical = [check for check in checks if not check.ok and check.label not in {"CUDA"}]
    if critical:
        print_doctor(checks)
        print("Kritik ön kontroller başarısız olduğu için analiz başlatılmadı.")
        return 2
    def progress(current: int, total: int, eta: float, stage: str) -> None:
        print(f"\r{stage}: {current}/{total} | ETA {eta:.1f}s", end="", flush=True)
    try:
        from .pipeline import AnalysisCancelled, AnalysisPipeline

        result = AnalysisPipeline(root, config).run(args.input, args.health_model, args.instrument_model, progress, threading.Event())
    except AnalysisCancelled as error:
        print(f"\nİptal edildi: {error}"); return 130
    except Exception as error:
        print(f"\nAnaliz başarısız: {error}"); return 1
    print(f"\nTamamlandı: {result.run_dir}")
    print(json.dumps({"summary": str(result.summary), "files": [str(path) for path in result.files]}, ensure_ascii=False, indent=2))
    return 0
