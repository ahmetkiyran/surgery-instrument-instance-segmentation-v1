import numpy as np

from surgical_pipeline.analytics import build_instrument_summary, not_visible_intervals, usage_intervals
from surgical_pipeline.schemas import Detection


def detection(timestamp: float, track_id: int, class_name: str = "scissors") -> Detection:
    return Detection(0, class_name, 0.8, track_id, np.ones((4, 4), dtype=np.uint8), int(timestamp * 10), timestamp)


def test_short_gap_is_bridged_and_duration_uses_fps() -> None:
    intervals = usage_intervals([detection(0.0, 1), detection(0.2, 1), detection(0.7, 1)], max_gap_seconds=0.5, frame_step_s=0.1)
    assert len(intervals) == 1
    assert intervals[0].duration_s == 0.8


def test_same_class_union_duration_differs_from_instance_time() -> None:
    items = [detection(0.0, 1), detection(0.1, 1), detection(0.0, 2), detection(0.1, 2)]
    summary, _ = build_instrument_summary(items, [], max_gap_seconds=0.5, frame_step_s=0.1, max_jump=1.0)
    assert summary["scissors"]["union_usage_seconds"] == 0.2
    assert summary["scissors"]["instance_time_seconds"] == 0.4


def test_not_visible_intervals_are_visible_timeline_complement() -> None:
    visible = usage_intervals([detection(0.2, 1), detection(0.3, 1)], max_gap_seconds=0.1, frame_step_s=0.1)
    assert not_visible_intervals(visible, 0.6) == [
        {"start_s": 0.0, "end_s": 0.2, "duration_s": 0.2},
        {"start_s": 0.4, "end_s": 0.6, "duration_s": 0.19999999999999996},
    ]


def test_expected_but_undetected_class_is_not_visible_for_full_video() -> None:
    summary, _ = build_instrument_summary([], [], 0.5, 0.1, 1.0, video_duration_s=2.0, expected_classes=["scissors"])
    assert summary["scissors"]["union_usage_seconds"] == 0.0
    assert summary["scissors"]["not_visible_seconds"] == 2.0
