"""Temporal usage analytics, independent from one-frame detection counts."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from .coordinates_3d import trajectory_lengths
from .schemas import Detection, TrackPoint, UsageInterval


def confirmed_health_count(history: dict[int, list[float]], current_ids: set[int], timestamp_s: float, min_frames: int, max_lost_seconds: float) -> int:
    """Count only temporally confirmed, recently active health-person tracks."""
    for track_id in current_ids:
        history.setdefault(track_id, []).append(timestamp_s)
    active = 0
    for observations in history.values():
        if len(observations) >= min_frames and timestamp_s - observations[-1] <= max_lost_seconds:
            active += 1
    return active


def usage_intervals(detections: list[Detection], max_gap_seconds: float, frame_step_s: float) -> list[UsageInterval]:
    """Bridge short misses within one track; each interval has a timestamp-based duration."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for detection in detections:
        if detection.track_id is not None:
            grouped[(detection.class_name, detection.track_id)].append(detection.timestamp_s)
    intervals: list[UsageInterval] = []
    for (class_name, track_id), times in grouped.items():
        times = sorted(set(times))
        if not times:
            continue
        start = previous = times[0]
        for timestamp in times[1:]:
            if timestamp - previous > max_gap_seconds + frame_step_s:
                end = round(previous + frame_step_s, 9)
                intervals.append(UsageInterval(class_name, track_id, start, end, round(max(0.0, end - start), 9)))
                start = timestamp
            previous = timestamp
        end = round(previous + frame_step_s, 9)
        intervals.append(UsageInterval(class_name, track_id, start, end, round(max(0.0, end - start), 9)))
    return sorted(intervals, key=lambda interval: (interval.class_name, interval.track_id, interval.start_s))


def union_duration(intervals: list[UsageInterval]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals, key=lambda interval: interval.start_s)
    total = 0.0
    start, end = ordered[0].start_s, ordered[0].end_s
    for interval in ordered[1:]:
        if interval.start_s <= end:
            end = max(end, interval.end_s)
        else:
            total += end - start
            start, end = interval.start_s, interval.end_s
    return total + end - start


def build_instrument_summary(detections: list[Detection], points: list[TrackPoint], max_gap_seconds: float, frame_step_s: float, max_jump: float) -> tuple[dict[str, dict], list[UsageInterval]]:
    intervals = usage_intervals(detections, max_gap_seconds, frame_step_s)
    lengths = trajectory_lengths(points, max_jump)
    by_class_detections: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_class_detections[detection.class_name].append(detection)
    result: dict[str, dict] = {}
    for class_name, class_detections in sorted(by_class_detections.items()):
        class_intervals = [interval for interval in intervals if interval.class_name == class_name]
        track_ids = sorted({detection.track_id for detection in class_detections if detection.track_id is not None})
        track_metrics = [lengths.get((class_name, track_id), {}) for track_id in track_ids]
        result[class_name] = {
            "union_usage_seconds": union_duration(class_intervals),
            "instance_time_seconds": sum(interval.duration_s for interval in class_intervals),
            "first_seen_seconds": min(detection.timestamp_s for detection in class_detections),
            "last_seen_seconds": max(detection.timestamp_s for detection in class_detections),
            "track_count": len(track_ids),
            "usage_interval_count": len(class_intervals),
            "average_confidence": mean(detection.confidence for detection in class_detections),
            "relative_2d_motion": sum(float(metric.get("relative_2d_motion", 0.0)) for metric in track_metrics),
            "relative_3d_motion": sum(float(metric.get("relative_3d_motion", 0.0)) for metric in track_metrics),
            "depth_validity_ratio": mean([float(metric.get("depth_validity_ratio", 0.0)) for metric in track_metrics]) if track_metrics else 0.0,
            "tracks": {str(track_id): lengths.get((class_name, track_id), {}) for track_id in track_ids},
        }
    return result, intervals


def health_statistics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    values = [int(row["active_health_person_count"]) for row in rows]
    return {"minimum": min(values) if values else 0, "maximum": max(values) if values else 0, "average": mean(values) if values else 0.0}
