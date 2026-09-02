"""Opt-in clean-machine test; skipped until authorised Release URLs and a local smoke video exist."""

import os
from dataclasses import replace
from pathlib import Path

import pytest

from surgical_pipeline.config import load_config
from surgical_pipeline.model_manager import ModelManager
from surgical_pipeline.pipeline import AnalysisPipeline


@pytest.mark.integration
def test_published_release_downloads_and_starts_pipeline(tmp_path: Path) -> None:
    if os.environ.get("MODEL_RELEASE_INTEGRATION") != "1":
        pytest.skip("Requires an authorised, published GitHub Release.")
    video = os.environ.get("MODEL_RELEASE_SMOKE_VIDEO")
    if not video or not Path(video).is_file():
        pytest.skip("MODEL_RELEASE_SMOKE_VIDEO must point to a local, permitted smoke video.")
    root = Path(__file__).resolve().parents[1]
    manager = ModelManager(root, weights_dir=tmp_path / "models" / "weights")
    if not manager.release_published:
        pytest.skip("Manifest has no authorised published release URL yet.")
    health, instrument = manager.download_all()
    assert all(status.ready for status in manager.statuses())
    config = replace(load_config(root), output_dir=tmp_path / "outputs", max_video_seconds=3.0)
    result = AnalysisPipeline(root, config).run(Path(video), health, instrument)
    assert result.processed_video.is_file() and result.summary.is_file()
