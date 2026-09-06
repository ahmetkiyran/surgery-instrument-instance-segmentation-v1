"""Temporal usage analytics, independent from one-frame detection counts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
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


def usage_intervals(
    detections: list[Detection],
    max_gap_seconds: float,
    frame_step_s: float,
    minimum_interval_seconds: float = 0.0,
) -> list[UsageInterval]:
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
                duration = round(max(0.0, end - start), 9)
                if duration >= minimum_interval_seconds:
                    intervals.append(UsageInterval(class_name, track_id, start, end, duration))
                start = timestamp
            previous = timestamp
        end = round(previous + frame_step_s, 9)
        duration = round(max(0.0, end - start), 9)
        if duration >= minimum_interval_seconds:
            intervals.append(UsageInterval(class_name, track_id, start, end, duration))
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


def not_visible_intervals(intervals: list[UsageInterval], duration_s: float) -> list[dict[str, float]]:
    """Return the complement of visible class intervals over the processed timeline."""
    duration = max(0.0, float(duration_s))
    if duration == 0:
        return []
    merged: list[tuple[float, float]] = []
    for interval in sorted(intervals, key=lambda item: item.start_s):
        start = min(duration, max(0.0, interval.start_s))
        end = min(duration, max(start, interval.end_s))
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    result: list[dict[str, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            result.append({"start_s": cursor, "end_s": start, "duration_s": start - cursor})
        cursor = max(cursor, end)
    if cursor < duration:
        result.append({"start_s": cursor, "end_s": duration, "duration_s": duration - cursor})
    return result


def build_instrument_summary(
    detections: list[Detection],
    points: list[TrackPoint],
    max_gap_seconds: float,
    frame_step_s: float,
    max_jump: float,
    minimum_interval_seconds: float = 0.0,
    video_duration_s: float | None = None,
    expected_classes: Iterable[str] | None = None,
) -> tuple[dict[str, dict], list[UsageInterval]]:
    intervals = usage_intervals(detections, max_gap_seconds, frame_step_s, minimum_interval_seconds)
    lengths = trajectory_lengths(points, max_jump)
    by_class_detections: dict[str, list[Detection]] = defaultdict(list)
    for detection in detections:
        by_class_detections[detection.class_name].append(detection)
    result: dict[str, dict] = {}
    class_names = set(by_class_detections)
    class_names.update(str(name) for name in (expected_classes or ()))
    for class_name in sorted(class_names):
        class_detections = by_class_detections[class_name]
        class_intervals = [interval for interval in intervals if interval.class_name == class_name]
        absent = not_visible_intervals(class_intervals, video_duration_s) if video_duration_s is not None else []
        track_ids = sorted({detection.track_id for detection in class_detections if detection.track_id is not None})
        track_metrics = [lengths.get((class_name, track_id), {}) for track_id in track_ids]
        timestamps: dict[float, set[int]] = defaultdict(set)
        for detection in class_detections:
            if detection.track_id is not None:
                timestamps[detection.timestamp_s].add(detection.track_id)
        raw_active = len(timestamps) * frame_step_s
        merged = union_duration(class_intervals)
        result[class_name] = {
            "raw_active_seconds": raw_active,
            "merged_active_seconds": merged,
            "union_usage_seconds": merged,
            "instance_time_seconds": sum(interval.duration_s for interval in class_intervals),
            "first_seen_seconds": min((detection.timestamp_s for detection in class_detections), default=None),
            "last_seen_seconds": max((detection.timestamp_s for detection in class_detections), default=None),
            "track_count": len(track_ids),
            "valid_track_count": len(track_ids),
            "usage_interval_count": len(class_intervals),
            "usage_intervals": [{"track_id": item.track_id, "start_s": item.start_s, "end_s": item.end_s, "duration_s": item.duration_s} for item in class_intervals],
            "not_visible_seconds": sum(item["duration_s"] for item in absent),
            "not_visible_intervals": absent,
            "average_confidence": mean(detection.confidence for detection in class_detections) if class_detections else 0.0,
            "maximum_concurrent_instances": max((len(ids) for ids in timestamps.values()), default=0),
            "relative_2d_motion": sum(float(metric.get("relative_2d_motion", 0.0)) for metric in track_metrics),
            "relative_3d_motion": sum(float(metric.get("relative_3d_motion", 0.0)) for metric in track_metrics),
            "depth_validity_ratio": mean([float(metric.get("depth_validity_ratio", 0.0)) for metric in track_metrics]) if track_metrics else 0.0,
            "tracks": {str(track_id): lengths.get((class_name, track_id), {}) for track_id in track_ids},
        }
    return result, intervals


def health_statistics(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    values = [int(row["active_health_person_count"]) for row in rows]
    return {"minimum": min(values) if values else 0, "maximum": max(values) if values else 0, "average": mean(values) if values else 0.0}
