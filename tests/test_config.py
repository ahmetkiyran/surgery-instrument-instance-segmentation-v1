from pathlib import Path

import pytest

from surgical_pipeline.config import AppConfig, load_config


def test_even_blur_kernel_is_normalised() -> None:
    assert AppConfig(blur_kernel=50).validate().blur_kernel == 51


def test_invalid_tracker_is_rejected() -> None:
    with pytest.raises(ValueError, match="botsort"):
        AppConfig(tracker_algorithm="unknown").validate()


def test_config_loads_without_local_env(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text("runtime:\n  output_dir: results\n", encoding="utf-8")
    (tmp_path / "configs" / "instrument_names.yaml").write_text("instrument_display_names: {}\n", encoding="utf-8")
    assert load_config(tmp_path).output_dir == Path("results")
