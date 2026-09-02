"""Static scientific figures and an interactive relative-trajectory view."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .schemas import TrackPoint


def _mmss(seconds: float) -> str:
    minutes, seconds_int = divmod(round(seconds), 60)
    return f"{minutes:02d}:{seconds_int:02d}"


def usage_duration_chart(instruments: dict[str, dict], path: Path) -> Path:
    labels = list(instruments)
    values = [instruments[label]["union_usage_seconds"] for label in labels]
    fig, axis = plt.subplots(figsize=(9, max(3, 0.55 * len(labels) + 1.5)))
    bars = axis.barh(labels, values, color="#15c6d6")
    axis.set_xlabel("Kullanım süresi (model tarafından görünür olarak algılanan süre, saniye)")
    axis.set_title("Alet sınıfı bazında birleşik kullanım süresi")
    for bar, value in zip(bars, values, strict=False):
        axis.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, " " + _mmss(float(value)), va="center")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return path


def relative_motion_chart(instruments: dict[str, dict], path: Path) -> Path:
    labels = list(instruments)
    values = [instruments[label]["relative_3d_motion"] for label in labels]
    fig, axis = plt.subplots(figsize=(9, max(3, 0.55 * len(labels) + 1.5)))
    axis.barh(labels, values, color="#8b5cf6")
    axis.set_xlabel("Toplam göreli 3B hareket (relative motion units)")
    axis.set_title("Alet sınıfı bazında toplam göreli 3B hareket")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return path


def health_count_chart(rows: list[dict], path: Path) -> Path:
    x = [row["timestamp_s"] for row in rows]
    y = [row["active_health_person_count"] for row in rows]
    fig, axis = plt.subplots(figsize=(10, 4))
    axis.step(x, y, where="post", color="#14b8a6", linewidth=1.5)
    axis.set_xlabel("Video zamanı (s)"); axis.set_ylabel("Aktif sağlık personeli")
    axis.set_yticks(sorted(set(y)) if y else [0]); axis.grid(alpha=0.25)
    axis.set_title("Aktif sağlık personeli sayısı")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return path


def trajectories_charts(points: list[TrackPoint], html_path: Path, png_path: Path) -> tuple[Path, Path]:
    grouped: dict[tuple[str, int], list[TrackPoint]] = defaultdict(list)
    for point in points:
        if None not in (point.relative_x, point.relative_y, point.relative_depth):
            grouped[(point.class_name, point.track_id)].append(point)
    figure = go.Figure()
    for (class_name, track_id), track_points in grouped.items():
        figure.add_trace(go.Scatter3d(x=[p.relative_x for p in track_points], y=[p.relative_y for p in track_points], z=[p.relative_depth for p in track_points], mode="lines", name=f"{class_name} / Track {track_id}"))
    figure.update_layout(template="plotly_white", title="Göreli 3B yörüngeler / Relative 3D trajectory", scene={"xaxis_title": "x (normalized)", "yaxis_title": "y (normalized)", "zaxis_title": "relative depth"}, margin={"l": 0, "r": 0, "t": 50, "b": 0})
    figure.write_html(html_path, include_plotlyjs="inline", full_html=True)
    fig = plt.figure(figsize=(8, 6)); axis = fig.add_subplot(111, projection="3d")
    for (class_name, track_id), track_points in grouped.items():
        axis.plot([p.relative_x for p in track_points], [p.relative_y for p in track_points], [p.relative_depth for p in track_points], label=f"{class_name} / {track_id}")
    axis.set_xlabel("x (normalized)"); axis.set_ylabel("y (normalized)"); axis.set_zlabel("relative depth")
    if grouped: axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(png_path, dpi=180); plt.close(fig)
    return html_path, png_path
