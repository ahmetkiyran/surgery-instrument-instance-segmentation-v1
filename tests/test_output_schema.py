import json
from pathlib import Path

from surgical_pipeline.report import write_summary
from surgical_pipeline.schemas import SummarySchema
from surgical_pipeline.utils import safe_run_dir


def test_safe_run_directory_is_contained_and_summary_has_no_source_path(tmp_path: Path) -> None:
    run_dir = safe_run_dir(tmp_path / "outputs")
    assert run_dir.parent == (tmp_path / "outputs").resolve()
    path = write_summary(run_dir, {"video": {"source_filename": "redacted"}, "warnings": []})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "C:\\" not in json.dumps(payload)


def test_completed_summary_contract_requires_core_sections() -> None:
    payload = {
        "schema_version": "1.0",
        "video": {},
        "models": {},
        "health_person_statistics": {},
        "instrument_usage": {},
        "relative_3d_note": "relative",
        "processing": {},
    }
    assert SummarySchema.model_validate(payload).schema_version == "1.0"
