from surgical_pipeline.coordinates_3d import robust_mask_depth, trajectory_lengths
from surgical_pipeline.schemas import TrackPoint


def point(x: float, y: float, depth: float | None, index: int) -> TrackPoint:
    return TrackPoint(index, index / 10, 7, "scissors", 0.9, x, y, depth, x, y, depth, depth is not None)


def test_relative_3d_distance_and_large_jump_filtering() -> None:
    metrics = trajectory_lengths([point(0.0, 0.0, 0.0, 0), point(0.1, 0.0, 0.1, 1), point(0.9, 0.9, 0.9, 2)], max_jump=0.3)
    assert metrics[("scissors", 7)]["relative_2d_motion"] == 0.1
    assert metrics[("scissors", 7)]["relative_3d_motion"] > 0.14


def test_missing_depth_and_outlier_values_are_handled() -> None:
    import numpy as np

    depth = np.array([[1.0, 1.0, 999.0], [1.1, 1.0, 1.0]], dtype=np.float32)
    mask = np.ones_like(depth, dtype=np.uint8)
    assert 0.9 < robust_mask_depth(depth, mask) < 1.2
    metrics = trajectory_lengths([point(0, 0, None, 0), point(0.1, 0, None, 1)], 1.0)
    assert metrics[("scissors", 7)]["relative_3d_motion"] == 0.0
