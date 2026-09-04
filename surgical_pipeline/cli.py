"""Production-oriented terminal adapter for the UI-independent pipeline APIs."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import __version__
from .config import AppConfig, load_config
from .doctor import print_doctor, run_doctor
from .model_loader import ModelValidationError, resolve_model_file_paths
from .model_manager import ModelManager, ModelManagerError
from .pose_estimator import PoseModelError
from .schemas import PIPELINE_SCHEMA_VERSION
from .video_io import VideoError, VideoMetadata, read_metadata


EXIT_OK = 0
EXIT_USAGE = 2
EXIT_MISSING_FILE = 3
EXIT_VALIDATION = 4
EXIT_ANALYSIS = 5
EXIT_OUTPUT = 6
EXIT_CANCELLED = 130


class CliError(RuntimeError):
    """A short, user-actionable error with the public command exit code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class AnalysisTargets:
    input_video: Path
    output_dir: Path
    health_model: Path
    instrument_model: Path
    pose_model: Path | None
    privacy_mode: str
    video_metadata: VideoMetadata


def _path(value: str) -> Path:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("Yol boş olamaz.")
    return Path(value).expanduser()


def _add_model_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("--health-model", type=_path, help="Health-personnel instance-segmentation ağırlığı (.pt).")
    command.add_argument("--instrument-model", type=_path, help="Cerrahi alet instance-segmentation ağırlığı (.pt).")
    command.add_argument("--pose-model", type=_path, help="İnsan pose modeli ağırlığı (.pt); skeleton modlarında zorunludur.")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="python -m surgical_pipeline",
        description="Yerel, gizlilik-öncelikli cerrahi video analizi.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subcommands = command.add_subparsers(dest="command", required=True, title="komutlar")

    doctor = subcommands.add_parser("doctor", help="Sistem, bağımlılık, model ve çıktı yolu ön kontrolü", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_model_options(doctor)
    doctor.add_argument("--output-dir", type=_path, help="Yazılabilirlik kontrolü yapılacak çıktı dizini.")
    doctor.add_argument("--config", type=_path, help="Varsayılanların üzerine uygulanacak YAML dosyası.")
    doctor.add_argument("--device", help="auto, cpu, cuda[:N] veya GPU indeksi.")
    doctor.add_argument("--privacy-mode", choices=("skeleton-only", "blur", "both"), help="Pose modelinin kritik sayılacağı analiz modu.")

    version = subcommands.add_parser("version", help="Paket ve çalışma zamanı sürümlerini göster")
    version.set_defaults(command="version")

    models = subcommands.add_parser("models", help="Doğrulanmış release model ağırlıklarını yönet")
    models_subcommands = models.add_subparsers(dest="models_command", required=True)
    for name in ("list", "download", "verify", "redownload"):
        models_subcommands.add_parser(name, help=f"Model {name} işlemi")

    analyze = subcommands.add_parser("analyze", help="Mevcut pipeline API'leriyle bir videoyu analiz et", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    analyze.add_argument("--input", required=True, type=_path, help="Okunabilir kaynak video dosyası. Kaynak asla değiştirilmez.")
    _add_model_options(analyze)
    analyze.add_argument("--output-dir", type=_path, help="Çalışma alt dizinlerinin yazılacağı kök dizin.")
    analyze.add_argument("--config", type=_path, help="Varsayılanların üzerine uygulanacak YAML dosyası.")
    analyze.add_argument("--privacy-mode", choices=("skeleton-only", "blur", "both"), help="skeleton-only RGB/sesi çıktı videosuna aktarmadan sentetik iskelet üretir; varsayılan config veya skeleton-only.")
    analyze.add_argument("--device", help="auto, cpu, cuda[:N] veya GPU indeksi. CPU'da FP16 kapatılır.")
    analyze.add_argument("--imgsz", type=int, help="YOLO çıkarım görüntü boyutu; en az 32 piksel.")
    analyze.add_argument("--health-conf", type=float, help="Health modeli güven eşiği (0, 1].")
    analyze.add_argument("--instrument-conf", type=float, help="Alet modeli güven eşiği (0, 1].")
    analyze.add_argument("--pose-conf", type=float, help="Pose/keypoint güven eşiği (0, 1]. Varsayılan: 0.25.")
    analyze.add_argument("--iou", type=float, help="YOLO IoU eşiği (0, 1].")
    analyze.add_argument("--tracker", choices=("botsort", "bytetrack"), help="Takip algoritması.")
    analyze.add_argument("--gap-tolerance", type=float, help="Aynı alet kullanım aralığı için izinli zaman boşluğu (saniye).")
    analyze.add_argument("--minimum-interval", type=float, help="Raporlanacak en kısa alet kullanım aralığı (saniye).")
    analyze.add_argument("--depth-interval", "--depth-stride", dest="depth_stride", type=int, help="Monoküler depth çıkarımı için kare aralığı; en az 1.")
    analyze.add_argument("--no-depth", "--disable-depth", dest="disable_depth", action="store_true", help="Göreli 3B/4B z_rel verisini üretmez; ilgili alanlar eksik kalır.")
    analyze.add_argument("--overwrite", action="store_true", help="Mevcut çıktı köküne yazılmasına izin verir; yine de benzersiz çalışma dizini oluşturulur.")
    analyze.add_argument("--quiet", action="store_true", help="İlerleme satırlarını gizler; hata ve nihai özet gösterilir.")
    analyze.add_argument("--debug", action="store_true", help="Hata oluşursa terminale ayrıntılı traceback yazdırır.")
    # Compatibility with the original local command surface.
    analyze.add_argument("--health-class", help=argparse.SUPPRESS)
    analyze.add_argument("--blur-kernel", type=int, help=argparse.SUPPRESS)
    analyze.add_argument("--confidence", type=float, help=argparse.SUPPRESS)
    return command


def _analyze_overrides(args: argparse.Namespace) -> dict:
    runtime: dict = {}
    privacy: dict = {}
    tracking: dict = {}
    depth: dict = {}
    analytics: dict = {}
    model: dict = {}
    cli: dict = {}
    reproducibility: dict = {}
    if getattr(args, "output_dir", None):
        runtime["output_dir"] = str(args.output_dir)
    if getattr(args, "device", None):
        runtime["device"] = args.device
    if getattr(args, "privacy_mode", None):
        cli["privacy_mode"] = args.privacy_mode
    if getattr(args, "blur_kernel", None) is not None:
        privacy["blur_kernel"] = args.blur_kernel
    if getattr(args, "confidence", None) is not None:
        tracking["confidence"] = args.confidence
        tracking["health"] = {"confidence": args.confidence}
        tracking["instrument"] = {"confidence": args.confidence}
    if getattr(args, "health_conf", None) is not None:
        tracking.setdefault("health", {})["confidence"] = args.health_conf
    if getattr(args, "instrument_conf", None) is not None:
        tracking.setdefault("instrument", {})["confidence"] = args.instrument_conf
    if getattr(args, "iou", None) is not None:
        tracking["iou"] = args.iou
        tracking.setdefault("health", {})["iou"] = args.iou
        tracking.setdefault("instrument", {})["iou"] = args.iou
    if getattr(args, "tracker", None):
        tracking["algorithm"] = args.tracker
    if getattr(args, "gap_tolerance", None) is not None:
        analytics["max_gap_seconds"] = args.gap_tolerance
    if getattr(args, "minimum_interval", None) is not None:
        analytics["minimum_interval_seconds"] = args.minimum_interval
    if getattr(args, "depth_stride", None) is not None:
        depth["depth_stride"] = args.depth_stride
    if getattr(args, "disable_depth", False):
        depth["enabled"] = False
    if getattr(args, "imgsz", None) is not None:
        model["image_size"] = args.imgsz
    for argument, config_key in (("health_model", "health_model_path"), ("instrument_model", "instrument_model_path"), ("pose_model", "pose_model_path")):
        value = getattr(args, argument, None)
        if value is not None:
            model[config_key] = str(value)
    health_class = getattr(args, "health_class", None)
    if health_class:
        if health_class.isdecimal():
            model["health_class_id"] = int(health_class)
        else:
            model["health_class_name"] = health_class
    if getattr(args, "debug", False):
        reproducibility["log_level"] = "DEBUG"
    values = {
        "runtime": runtime,
        "privacy": privacy,
        "tracking": tracking,
        "depth": depth,
        "analytics": analytics,
        "model": model,
        "cli": cli,
        "reproducibility": reproducibility,
    }
    return {name: value for name, value in values.items() if value}


def _safe_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _default_pose_path(root: Path) -> Path:
    for candidate in (root / "models" / "weights" / "yolo11m-pose.pt", root / "yolo11m-pose.pt"):
        if candidate.is_file():
            return candidate.resolve()
    return (root / "models" / "weights" / "yolo11m-pose.pt").resolve()


def _validate_pose_confidence(value: float | None) -> float:
    confidence = 0.25 if value is None else value
    if not 0 < confidence <= 1:
        raise CliError(EXIT_USAGE, "--pose-conf değeri 0 ile 1 arasında olmalıdır.")
    return confidence


def _resolve_targets(root: Path, args: argparse.Namespace, config: AppConfig) -> AnalysisTargets:
    input_video = _safe_path(args.input)
    if not input_video.is_file():
        raise CliError(EXIT_MISSING_FILE, f"Girdi videosu bulunamadı: {input_video.name}")
    output_dir = _safe_path(config.output_dir)
    if output_dir == input_video or (output_dir.exists() and not output_dir.is_dir()):
        raise CliError(EXIT_OUTPUT, "Kaynak video çıktı hedefi olarak kullanılamaz; --output-dir bir dizin olmalıdır.")
    try:
        health_model, instrument_model = resolve_model_file_paths(root, config.health_model_path, config.instrument_model_path, auto_download=False)
    except (ModelManagerError, ModelValidationError, OSError) as error:
        provided = config.health_model_path if config.health_model_path and not config.health_model_path.is_file() else config.instrument_model_path
        name = provided.name if provided is not None else "varsayılan health/instrument modeli"
        raise CliError(EXIT_MISSING_FILE, f"Model dosyası bulunamadı veya doğrulanmış varsayılan hazır değil: {name}") from error
    privacy_mode = args.privacy_mode or config.cli_privacy_mode
    pose_model = _safe_path(config.pose_model_path) if config.pose_model_path else _default_pose_path(root)
    if privacy_mode in {"skeleton-only", "both"} and not pose_model.is_file():
        raise CliError(EXIT_MISSING_FILE, f"{privacy_mode} modu için pose modeli bulunamadı: {pose_model.name}")
    protected = {input_video, health_model.resolve(), instrument_model.resolve(), pose_model.resolve() if pose_model.exists() else pose_model}
    if output_dir in protected or (output_dir.exists() and not output_dir.is_dir()):
        raise CliError(EXIT_OUTPUT, "Kaynak video veya model dosyası çıktı hedefi olarak kullanılamaz; --output-dir bir dizin olmalıdır.")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        probe = output_dir / ".surgical_pipeline_cli_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as error:
        raise CliError(EXIT_OUTPUT, f"Çıktı dizinine yazılamıyor: {output_dir.name}") from error
    try:
        metadata = read_metadata(input_video)
    except VideoError as error:
        raise CliError(EXIT_MISSING_FILE, str(error)) from error
    return AnalysisTargets(input_video, output_dir, health_model, instrument_model, pose_model if privacy_mode != "blur" else None, privacy_mode, metadata)


def _seconds(value: float) -> str:
    seconds = max(0, int(round(value)))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


class ProgressDisplay:
    """Throttle existing pipeline callbacks to readable terminal updates."""

    def __init__(self, quiet: bool) -> None:
        self.quiet = quiet
        self.started = time.perf_counter()
        self.last_update = 0.0

    def __call__(self, current: int, total: int, eta: float, stage: str) -> None:
        if self.quiet:
            return
        now = time.perf_counter()
        if current != 1 and total and current != total and now - self.last_update < 1.0:
            return
        self.last_update = now
        elapsed = _seconds(now - self.started)
        if total > 0:
            percent = current / total * 100
            print(f"{stage}: {current}/{total} kare (%{percent:.1f}) | Geçen: {elapsed} | Tahmini kalan: {_seconds(eta)}", flush=True)
        else:
            print(f"{stage}: {current} kare | Geçen: {elapsed}", flush=True)


@contextmanager
def _cancel_at_safe_boundary(event: threading.Event) -> Iterator[None]:
    """Convert Ctrl+C into the pipeline's existing cooperative cancellation token."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)

    def request_cancel(_signum, _frame) -> None:
        event.set()

    signal.signal(signal.SIGINT, request_cancel)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _print_result_summary(mode: str, blur_result: object | None, pose_result: object | None, elapsed: float) -> None:
    files: list[Path] = []
    run_ids: list[str] = []
    processed = 0
    privacy_status = "üretilmedi"
    if blur_result is not None:
        run_ids.append(blur_result.run_dir.name)
        files.extend(path for path in blur_result.files if path.is_file())
        processed = int(_read_json(blur_result.summary).get("video", {}).get("processed_frame_count", processed))
    if pose_result is not None:
        run_ids.append(pose_result.run_dir.name)
        files.extend(path for path in pose_result.files if path.is_file())
        manifest = _read_json(pose_result.manifest)
        processed = max(processed, int(manifest.get("processed_frame_count", 0)))
        privacy_status = "başarılı" if _read_json(pose_result.privacy_report).get("passed") else "uyarı"
    unique_files = list(dict.fromkeys(files))
    print("\nAnaliz tamamlandı.")
    print(f"Çalışma ID        : {', '.join(run_ids)}")
    print(f"Süre              : {_seconds(elapsed)}")
    print(f"İşlenen kare      : {processed}")
    print(f"Gizlilik modu     : {mode}")
    print("Health modeli     : hazır")
    print("Alet modeli       : hazır")
    print(f"Pose modeli       : {'hazır' if pose_result is not None else 'kullanılmadı'}")
    print(f"Privacy audit     : {privacy_status}")
    print("Çıktılar:")
    for path in unique_files:
        print(f"- {path.name}")


def _run_models(args: argparse.Namespace, root: Path) -> int:
    try:
        manager = ModelManager(root)
        if args.models_command in {"list", "verify"}:
            statuses = manager.statuses()
            for status in statuses:
                print(f"{'[OK]' if status.ready else '[MISSING]'} {status.key}: {status.detail}")
            return EXIT_OK if args.models_command == "list" or all(status.ready for status in statuses) else EXIT_VALIDATION

        def progress(key: str, current: int, total: int | None) -> None:
            if total:
                print(f"\r{key}: %{current / total * 100:5.1f}", end="", flush=True)
            else:
                print(f"\r{key}: {current / (1024 * 1024):.1f} MiB", end="", flush=True)

        manager.download_all(force=args.models_command == "redownload", progress=progress)
        print("\nModeller SHA-256, task ve sınıf bilgileriyle doğrulandı.")
        return EXIT_OK
    except ModelManagerError as error:
        print(f"Hata: Model işlemi başarısız: {error}", file=sys.stderr)
        return EXIT_VALIDATION


def _run_doctor(args: argparse.Namespace, root: Path, config: AppConfig) -> int:
    mode = args.privacy_mode or config.cli_privacy_mode
    return print_doctor(
        run_doctor(root, config, args.health_model, args.instrument_model, args.pose_model, args.output_dir, mode in {"skeleton-only", "both"}, args.config)
    )


def _run_analyze(args: argparse.Namespace, root: Path, config: AppConfig) -> int:
    targets = _resolve_targets(root, args, config)
    pose_confidence = _validate_pose_confidence(args.pose_conf)
    if not args.quiet:
        print("Modeller doğrulanıyor...")
        print("Video açılıyor...")
        print(f"Analiz başladı: {targets.video_metadata.width}x{targets.video_metadata.height}, {targets.video_metadata.fps:g} FPS")
    progress = ProgressDisplay(args.quiet)
    cancellation = threading.Event()
    started = time.perf_counter()
    blur_result = pose_result = None
    try:
        with _cancel_at_safe_boundary(cancellation):
            if targets.privacy_mode in {"blur", "both"}:
                from .pipeline import AnalysisPipeline

                blur_result = AnalysisPipeline(root, config).run(
                    targets.input_video, targets.health_model, targets.instrument_model, progress, cancellation,
                )
            if targets.privacy_mode in {"skeleton-only", "both"}:
                from .pose_pipeline import run_pose_pilot

                pose_result = run_pose_pilot(
                    targets.input_video,
                    targets.health_model,
                    targets.instrument_model,
                    targets.output_dir,
                    targets.pose_model,
                    config,
                    keypoint_confidence=pose_confidence,
                    progress=progress,
                    cancel_event=cancellation,
                )
    except KeyboardInterrupt:
        cancellation.set()
        print("\nİptal edildi; güvenli durma isteği kaydedildi.", file=sys.stderr)
        return EXIT_CANCELLED
    except Exception as error:
        return _report_runtime_error(error, args.debug)
    _print_result_summary(targets.privacy_mode, blur_result, pose_result, time.perf_counter() - started)
    return EXIT_OK


def _report_runtime_error(error: Exception, debug: bool) -> int:
    message = str(error)
    lowered = message.casefold()
    if isinstance(error, CliError):
        code = error.code
    elif isinstance(error, (ModelValidationError, PoseModelError)):
        code = EXIT_MISSING_FILE if "bulunamad" in lowered else EXIT_VALIDATION
    elif isinstance(error, (FileNotFoundError,)) or (isinstance(error, VideoError) and ("girdi" in lowered or "video aç" in lowered)):
        code = EXIT_MISSING_FILE
    elif isinstance(error, VideoError) or "çıktı" in lowered or "disk" in lowered:
        code = EXIT_OUTPUT
    elif "cuda cihazı istendi" in lowered:
        code = EXIT_VALIDATION
    else:
        code = EXIT_ANALYSIS
    print(f"Hata: {message}", file=sys.stderr)
    if debug:
        traceback.print_exc()
    else:
        print("Ayrıntılar varsa ilgili çalışma dizinindeki pipeline.log dosyasındadır.", file=sys.stderr)
    return code


def _print_version() -> int:
    try:
        import ultralytics

        ultralytics_version = str(ultralytics.__version__)
    except Exception:
        ultralytics_version = "yüklü değil"
    print(f"Surgical Video Analytics {__version__}")
    print(f"Pipeline schema: {PIPELINE_SCHEMA_VERSION}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Ultralytics: {ultralytics_version}")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    try:
        args = parser().parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    root = Path.cwd().resolve()
    if args.command == "version":
        return _print_version()
    if args.command == "models":
        return _run_models(args, root)
    try:
        config = load_config(root, _analyze_overrides(args), getattr(args, "config", None))
    except (OSError, TypeError, ValueError) as error:
        print(f"Hata: Yapılandırma geçersiz: {error}", file=sys.stderr)
        return EXIT_USAGE
    if args.command == "doctor":
        return _run_doctor(args, root, config)
    try:
        return _run_analyze(args, root, config)
    except CliError as error:
        return _report_runtime_error(error, args.debug)
