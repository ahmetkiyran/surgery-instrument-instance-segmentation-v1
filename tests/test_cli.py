"""Fast, model-free tests for the terminal adapter contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from surgical_pipeline import cli
from surgical_pipeline.config import load_config
from surgical_pipeline.doctor import Check, print_doctor


@pytest.fixture
def cli_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        "runtime:\n  output_dir: outputs\n  device: auto\nprivacy:\n  mode: gaussian\nmodel:\n  fp16: true\ncli:\n  privacy_mode: skeleton-only\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "instrument_names.yaml").write_text("instrument_display_names: {}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "read_metadata", lambda _path: SimpleNamespace(width=64, height=48, fps=10.0, frame_count=3))
    return tmp_path


def _files(project: Path) -> tuple[Path, Path, Path]:
    source = project / "video with spaces.mp4"
    health = project / "health.pt"
    instrument = project / "instrument.pt"
    for path in (source, health, instrument):
        path.write_bytes(b"placeholder")
    return source, health, instrument


def _mock_model_resolution(monkeypatch: pytest.MonkeyPatch, health: Path, instrument: Path) -> None:
    monkeypatch.setattr(cli, "resolve_model_file_paths", lambda *_args, **_kwargs: (health, instrument))


def _success_result(output: Path) -> SimpleNamespace:
    run_dir = output / "run_20260101_010101"
    run_dir.mkdir(parents=True)
    summary = run_dir / "analysis.json"
    summary.write_text(json.dumps({"video": {"processed_frame_count": 3}}), encoding="utf-8")
    manifest = run_dir / "run_manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    existing = run_dir / "processed_video.mp4"
    existing.write_bytes(b"video")
    return SimpleNamespace(run_dir=run_dir, files=[summary, manifest, existing, run_dir / "not-created.csv"], summary=summary, manifest=manifest)


def _mock_pipeline(monkeypatch: pytest.MonkeyPatch, result: SimpleNamespace | None = None, error: Exception | None = None) -> list[SimpleNamespace]:
    calls: list[SimpleNamespace] = []

    class FakePipeline:
        def __init__(self, _root: Path, config) -> None:
            self.config = config

        def run(self, *_args) -> SimpleNamespace:
            calls.append(SimpleNamespace(config=self.config, args=_args))
            if error is not None:
                raise error
            assert result is not None
            return result

    import surgical_pipeline.pipeline as pipeline

    monkeypatch.setattr(pipeline, "AnalysisPipeline", FakePipeline)
    return calls


def test_help_screens_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--help"]) == 0
    assert "komutlar" in capsys.readouterr().out
    assert cli.main(["analyze", "--help"]) == 0
    assert "skeleton-only RGB" in capsys.readouterr().out
    assert cli.main(["doctor", "--help"]) == 0
    assert "Pose modelinin" in capsys.readouterr().out
    assert cli.main(["version"]) == 0
    assert "Pipeline schema" in capsys.readouterr().out


def test_required_input_and_invalid_arguments_return_usage_errors(cli_project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["analyze"]) == 2
    assert cli.main(["analyze", "--input", "missing.mp4", "--privacy-mode", "invalid"]) == 2
    assert cli.main(["analyze", "--input", "missing.mp4", "--health-conf", "1.2"]) == 2
    assert cli.main(["analyze", "--input", "missing.mp4", "--device", "quantum"]) == 2
    assert "Hata:" in capsys.readouterr().err


def test_missing_video_and_models_have_documented_exit_code(cli_project: Path) -> None:
    assert cli.main(["analyze", "--input", "missing.mp4", "--privacy-mode", "blur"]) == 3
    source, _health, instrument = _files(cli_project)
    assert cli.main(["analyze", "--input", str(source), "--health-model", "missing health.pt", "--instrument-model", str(instrument), "--privacy-mode", "blur"]) == 3
    health = cli_project / "health.pt"
    assert cli.main(["analyze", "--input", str(source), "--health-model", str(health), "--instrument-model", "missing instrument.pt", "--privacy-mode", "blur"]) == 3


def test_skeleton_requires_pose_but_blur_does_not(cli_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, health, instrument = _files(cli_project)
    _mock_model_resolution(monkeypatch, health, instrument)
    assert cli.main(["analyze", "--input", str(source), "--privacy-mode", "skeleton-only", "--pose-model", "missing pose.pt"]) == 3
    result = _success_result(cli_project / "blur outputs")
    _mock_pipeline(monkeypatch, result)
    assert cli.main(["analyze", "--input", str(source), "--privacy-mode", "blur", "--output-dir", str(cli_project / "blur outputs")]) == 0


def test_windows_style_space_path_and_no_depth_are_forwarded(cli_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, health, instrument = _files(cli_project)
    _mock_model_resolution(monkeypatch, health, instrument)
    output = cli_project / "output with spaces"
    result = _success_result(output)
    calls = _mock_pipeline(monkeypatch, result)
    assert cli.main(["analyze", "--input", str(source), "--privacy-mode", "blur", "--output-dir", str(output), "--no-depth"]) == 0
    assert calls and calls[0].config.depth_enabled is False


def test_config_and_environment_precedence(cli_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = cli_project / "custom.yaml"
    custom.write_text(
        "runtime:\n  output_dir: config-output\n  device: cpu\nmodel:\n  health_model_path: config-health.pt\n  instrument_model_path: config-instrument.pt\n  pose_model_path: config-pose.pt\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SURGICAL_OUTPUT_DIR", "environment-output")
    monkeypatch.setenv("SURGICAL_DEVICE", "cuda")
    monkeypatch.setenv("HEALTH_MODEL_PATH", "environment-health.pt")
    configured = load_config(cli_project, config_path=custom)
    assert configured.output_dir == Path("config-output")
    assert configured.device == "cpu"
    assert configured.health_model_path == (cli_project / "config-health.pt").resolve()
    overridden = load_config(cli_project, {"runtime": {"output_dir": "cli-output"}, "model": {"health_model_path": "cli-health.pt"}}, custom)
    assert overridden.output_dir == Path("cli-output")
    assert overridden.health_model_path == (cli_project / "cli-health.pt").resolve()
    custom.unlink()
    environment = load_config(cli_project)
    assert environment.output_dir == Path("environment-output")
    assert environment.health_model_path == (cli_project / "environment-health.pt").resolve()


def test_source_cannot_be_output_target(cli_project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source, health, instrument = _files(cli_project)
    _mock_model_resolution(monkeypatch, health, instrument)
    assert cli.main(["analyze", "--input", str(source), "--privacy-mode", "blur", "--output-dir", str(source)]) == 6


def test_pipeline_error_cancellation_quiet_debug_and_safe_summary(cli_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    source, health, instrument = _files(cli_project)
    _mock_model_resolution(monkeypatch, health, instrument)
    output = cli_project / "out"
    _mock_pipeline(monkeypatch, error=RuntimeError("inference failed"))
    assert cli.main(["analyze", "--input", str(source), "--privacy-mode", "blur", "--output-dir", str(output)]) == 5
    assert "Traceback" not in capsys.readouterr().err
    _mock_pipeline(monkeypatch, error=RuntimeError("debug failure"))
    assert cli.main(["analyze", "--input", str(source), "--privacy-mode", "blur", "--output-dir", str(output), "--debug"]) == 5
    assert "Traceback" in capsys.readouterr().err
    _mock_pipeline(monkeypatch, error=KeyboardInterrupt())
    assert cli.main(["analyze", "--input", str(source), "--privacy-mode", "blur", "--output-dir", str(output)]) == 130
    capsys.readouterr()
    result = _success_result(output)
    _mock_pipeline(monkeypatch, result)
    assert cli.main(["analyze", "--input", str(source), "--privacy-mode", "blur", "--output-dir", str(output), "--quiet"]) == 0
    printed = capsys.readouterr().out
    assert "Analiz tamamlandı" in printed
    assert "not-created.csv" not in printed
    assert "Modeller doğrulanıyor" not in printed


def test_doctor_result_and_ascii_fallback(cli_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "run_doctor", lambda *_args, **_kwargs: [Check("ready", True, "ok")])
    assert cli.main(["doctor", "--privacy-mode", "blur"]) == 0
    assert "Sistem analize hazır" in capsys.readouterr().out
    monkeypatch.setattr(cli, "run_doctor", lambda *_args, **_kwargs: [Check("broken", False, "no", "critical")])
    assert cli.main(["doctor", "--privacy-mode", "blur"]) == 4
    assert print_doctor([Check("warn", False, "optional", "warning")], ascii_only=True) == 0
    assert "[WARN]" in capsys.readouterr().out


def test_legacy_doctor_json_is_machine_readable(cli_project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "run_doctor", lambda *_args, **_kwargs: [])
    assert cli.main(["doctor", "--json", "--privacy-mode", "blur"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ready"


def test_console_script_and_module_help_subprocess() -> None:
    root = Path(__file__).resolve().parents[1]
    assert 'surgery-analyze = "surgical_pipeline.cli:main"' in (root / "pyproject.toml").read_text(encoding="utf-8")
    completed = subprocess.run([sys.executable, "-m", "surgical_pipeline", "--help"], cwd=root, capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    assert "analyze" in completed.stdout
