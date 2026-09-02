import pytest

from surgical_pipeline.model_loader import ModelValidationError, resolve_health_class_id


@pytest.mark.parametrize("name", ["health_personel", "health_personnel", "health_person", "healthcare_personnel"])
def test_health_class_is_resolved_from_metadata_not_position(name: str) -> None:
    assert resolve_health_class_id({3: "monitor", 9: name}) == 9


def test_invalid_health_class_lists_available_model_classes() -> None:
    with pytest.raises(ModelValidationError, match="scissors"):
        resolve_health_class_id({0: "scissors", 4: "forceps"})
