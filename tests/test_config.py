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


def test_alternative_model_roles_and_relative_paths_are_generic(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        "model:\n  health_model_path: weights/operator.pt\n  instrument_model_path: weights/needle.pt\n  instrument_display_names:\n    needle: Needle\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "instrument_names.yaml").write_text("instrument_display_names: {scissors: scissors}\n", encoding="utf-8")
    config = load_config(tmp_path)
    assert config.health_model_path == (tmp_path / "weights" / "operator.pt").resolve()
    assert config.instrument_model_path == (tmp_path / "weights" / "needle.pt").resolve()
    assert config.instrument_display_names == {"needle": "Needle"}
