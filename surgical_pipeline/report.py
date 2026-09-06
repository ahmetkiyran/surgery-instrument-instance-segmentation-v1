"""Privacy-safe summary JSON, HTML report and downloadable result package."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

import pandas as pd

from .schemas import Detection, TrackPoint, UsageInterval
from .utils import write_json


DETECTION_COLUMNS = [
    "schema_version", "frame_index", "timestamp_s", "track_id", "class_id", "class_name", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "mask_area_px", "centroid_x", "centroid_y",
    "centroid_source", "mask_quality", "tracking_state",
]
TRACK_COLUMNS = [
    "schema_version", "frame_index", "timestamp_s", "track_id", "class_name", "confidence", "x", "y", "depth",
    "relative_x", "relative_y", "relative_depth", "depth_valid", "centroid_source", "depth_source", "depth_interpolated",
]
HEALTH_COLUMNS = ["schema_version", "frame_index", "timestamp_s", "active_health_person_count"]
INTERVAL_COLUMNS = ["schema_version", "class_name", "track_id", "start_s", "end_s", "duration_s", "source"]
USAGE_SUMMARY_COLUMNS = [
    "schema_version", "class_name", "first_seen_seconds", "last_seen_seconds", "raw_active_seconds", "merged_active_seconds",
    "union_usage_seconds", "instance_time_seconds", "not_visible_seconds", "usage_interval_count", "average_confidence",
    "maximum_concurrent_instances", "valid_track_count", "relative_2d_motion", "relative_3d_motion", "depth_validity_ratio",
]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> Path:
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def _detection_row(detection: Detection) -> dict:
    centroid = detection.centroid()
    bbox = detection.bbox_xyxy or (None, None, None, None)
    return {
        "schema_version": "1.0", "frame_index": detection.frame_index, "timestamp_s": detection.timestamp_s,
        "track_id": detection.track_id, "class_id": detection.class_id, "class_name": detection.class_name,
        "confidence": detection.confidence, "bbox_x1": bbox[0], "bbox_y1": bbox[1], "bbox_x2": bbox[2], "bbox_y2": bbox[3],
        "mask_area_px": int((detection.mask > 0).sum()), "centroid_x": centroid[0] if centroid else None,
        "centroid_y": centroid[1] if centroid else None, "centroid_source": detection.centroid_source(), "mask_quality": detection.mask_quality, "tracking_state": detection.tracking_state,
    }


def write_csvs(
    run_dir: Path,
    detections: list[Detection],
    track_points: list[TrackPoint],
    health_counts: list[dict],
    intervals: list[UsageInterval],
    instrument_summary: dict[str, dict],
) -> tuple[Path, ...]:
    """Write versioned, stable CSV contracts and legacy UI aliases."""
    detection_rows = [_detection_row(item) for item in detections]
    point_rows = [{"schema_version": "1.0", **point.row()} for point in track_points]
    health_rows = [{"schema_version": "1.0", **row} for row in health_counts]
    interval_rows = [{"schema_version": "1.0", **interval.row()} for interval in intervals]
    usage_rows = [{"schema_version": "1.0", "class_name": class_name, **{key: values.get(key) for key in USAGE_SUMMARY_COLUMNS if key not in {"schema_version", "class_name"}}} for class_name, values in sorted(instrument_summary.items())]
    detections_path = _write_csv(run_dir / "detections.csv", detection_rows, DETECTION_COLUMNS)
    tracks_path = _write_csv(run_dir / "tracks.csv", point_rows, TRACK_COLUMNS)
    health_path = _write_csv(run_dir / "health_person_count.csv", health_rows, HEALTH_COLUMNS)
    intervals_path = _write_csv(run_dir / "usage_intervals.csv", interval_rows, INTERVAL_COLUMNS)
    usage_summary_path = _write_csv(run_dir / "usage_summary.csv", usage_rows, USAGE_SUMMARY_COLUMNS)
    absent_rows = [
        {"schema_version": "1.0", "class_name": class_name, **interval, "source": "visible-table-complement"}
        for class_name, values in sorted(instrument_summary.items())
        for interval in values.get("not_visible_intervals", [])
    ]
    absent_path = _write_csv(
        run_dir / "not_visible_intervals.csv",
        absent_rows,
        ["schema_version", "class_name", "start_s", "end_s", "duration_s", "source"],
    )
    trajectories_path = _write_csv(run_dir / "trajectories_3d.csv", point_rows, TRACK_COLUMNS)
    # Existing Gradio releases expect these names; retain them as data-equivalent files.
    legacy_tracks = _write_csv(run_dir / "instrument_tracks.csv", point_rows, TRACK_COLUMNS)
    return detections_path, tracks_path, health_path, intervals_path, absent_path, usage_summary_path, trajectories_path, legacy_tracks


def write_summary(run_dir: Path, summary: dict) -> Path:
    """Write the versioned analysis contract and a legacy summary alias for the UI."""
    analysis_path = run_dir / "analysis.json"
    write_json(analysis_path, summary)
    write_json(run_dir / "summary.json", summary)
    return analysis_path


def write_html_report(run_dir: Path, summary: dict) -> Path:
    """No local path, patient name or original filename is emitted here."""
    instrument_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{values['union_usage_seconds']:.2f}</td><td>{values['instance_time_seconds']:.2f}</td><td>{values['relative_3d_motion']:.4f}</td></tr>" for name, values in summary.get("instrument_usage", {}).items())
    health_stats = summary.get("health_person_statistics", {})
    body = f"""<!doctype html><html lang='tr'><head><meta charset='utf-8'><title>Cerrahi video analiz raporu</title><style>body{{font-family:Arial,sans-serif;margin:36px;color:#102131}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cad5df;padding:8px;text-align:left}}th{{background:#e9f6f7}}.notice{{background:#fff8df;padding:12px}}</style></head><body><h1>Cerrahi video analiz raporu</h1><p class='notice'>Kullanım süresi, modelin aleti videoda görünür olarak algıladığı süredir; gerçek fiziksel kullanım süresi değildir. Göreli 3B sonuçlar metrik mesafe değildir.</p><h2>Video</h2><pre>{html.escape(str(summary.get('video', {})))}</pre><h2>Sağlık personeli</h2><p>Minimum: {health_stats.get('minimum', 0)} | Maksimum: {health_stats.get('maximum', 0)} | Ortalama: {health_stats.get('average', 0):.2f}</p><h2>Alet özetleri</h2><table><tr><th>Sınıf</th><th>Birleşik süre (s)</th><th>Instance-time (s)</th><th>Göreli 3B hareket (relative motion units)</th></tr>{instrument_rows}</table><h2>Uyarılar</h2><ul>{''.join('<li>'+html.escape(item)+'</li>' for item in summary.get('warnings', []))}</ul></body></html>"""
    path = run_dir / "analysis_report.html"; path.write_text(body, encoding="utf-8"); return path


def create_results_zip(run_dir: Path) -> Path:
    archive_base = run_dir / "results"
    # Exclude logs and temporary files because they may contain local diagnostic context.
    files = [path for path in run_dir.iterdir() if path.is_file() and path.suffix.lower() not in {".log", ".tmp"}]
    staging = run_dir / "_package"
    staging.mkdir(exist_ok=True)
    try:
        for source in files:
            shutil.copy2(source, staging / source.name)
        return Path(shutil.make_archive(str(archive_base), "zip", staging))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
