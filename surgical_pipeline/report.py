"""Privacy-safe summary JSON, HTML report and downloadable result package."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

import pandas as pd

from .schemas import TrackPoint, UsageInterval
from .utils import write_json


def write_csvs(run_dir: Path, track_points: list[TrackPoint], health_counts: list[dict], intervals: list[UsageInterval]) -> tuple[Path, Path, Path]:
    tracks = run_dir / "instrument_tracks.csv"; health = run_dir / "health_person_count.csv"; usage = run_dir / "usage_intervals.csv"
    pd.DataFrame([point.row() for point in track_points]).to_csv(tracks, index=False)
    pd.DataFrame(health_counts).to_csv(health, index=False)
    pd.DataFrame([interval.row() for interval in intervals]).to_csv(usage, index=False)
    return tracks, health, usage


def write_summary(run_dir: Path, summary: dict) -> Path:
    path = run_dir / "summary.json"; write_json(path, summary); return path


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
