from pathlib import Path

import cv2
import numpy as np

from surgical_pipeline.config import AppConfig
from surgical_pipeline.pipeline import AnalysisPipeline
from surgical_pipeline.schemas import Detection


def test_mock_models_process_a_synthetic_video(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for index in range(5):
        frame = np.full((48, 64, 3), 35 + index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    def fake_detector(role: str, frame: np.ndarray, index: int, timestamp: float) -> list[Detection]:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if role == "health":
            mask[5:25, 5:20] = 1
            return [Detection(0, "health_personel", 0.92, 101, mask, index, timestamp)]
        mask[25:35, 20 + index:35 + index] = 1
        return [Detection(0, "scissors", 0.87, 201, mask, index, timestamp)]

    config = AppConfig(output_dir=tmp_path / "outputs", depth_enabled=False, min_confirm_frames=1, blur_kernel=9).validate()
    result = AnalysisPipeline(tmp_path, config, fake_detector).run(source)
    expected = {"processed_video.mp4", "summary.json", "instrument_tracks.csv", "health_person_count.csv", "usage_intervals.csv", "usage_duration_chart.png", "relative_3d_motion_chart.png", "health_person_count_chart.png", "trajectories_3d.html", "trajectories_3d.png", "analysis_report.html", "run_config.yaml", "results.zip"}
    assert expected <= {path.name for path in result.run_dir.iterdir()}
    assert result.processed_video.stat().st_size > 0
