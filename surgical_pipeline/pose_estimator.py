"""One-load Ultralytics pose adapter with explicit keypoint validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .pose_schemas import COCO17_KEYPOINT_NAMES, PoseModelInfo, PoseObservation
from .utils import sha256_file


class PoseModelError(RuntimeError):
    """The selected pose model is unavailable or does not expose usable keypoints."""


def _as_numpy(value):
    if value is None:
        return None
    for method in ("detach", "cpu"):
        if hasattr(value, method):
            value = getattr(value, method)()
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


class PoseEstimator:
    """Loads an official YOLO pose model once and converts results to typed records."""

    def __init__(self, model_path: Path | str = "yolo11m-pose.pt", device: str | int = "auto", fp16: bool = True, confidence: float = 0.25) -> None:
        self.path = Path(model_path).expanduser().resolve()
        self.device = 0 if device == "auto" else device
        self.fp16 = fp16 and self.device != "cpu"
        self.confidence = confidence
        self.model = None
        self.info: PoseModelInfo | None = None

    def load(self) -> PoseModelInfo:
        if self.model is not None and self.info is not None:
            return self.info
        try:
            from ultralytics import YOLO

            # Ultralytics' official resolver may download an explicitly requested
            # pretrained filename. It is never substituted with an unrelated model.
            model = YOLO(str(self.path if self.path.is_file() else self.path.name))
        except Exception as error:
            raise PoseModelError(f"Pose modeli yüklenemedi: {self.path.name}. Manuel olarak bu adı yerleştirin: {error}") from error
        actual_path = Path(getattr(model, "ckpt_path", self.path)).expanduser().resolve()
        task = str(getattr(model, "task", "unknown"))
        shape = tuple(int(item) for item in getattr(model.model, "kpt_shape", ()))
        names = {int(key): str(value) for key, value in dict(getattr(model, "names", {})).items()}
        if task != "pose":
            raise PoseModelError(f"Pose modeli bekleniyordu, ancak task={task} döndü.")
        if len(shape) != 2 or shape[0] <= 0 or shape[1] < 3:
            raise PoseModelError("Pose modeli geçerli bir keypoint şekli bildirmedi.")
        if not names or "person" not in {name.casefold() for name in names.values()}:
            raise PoseModelError(f"Model insan pose modeli gibi görünmüyor: sınıflar={names}")
        keypoint_names = COCO17_KEYPOINT_NAMES if shape[0] == len(COCO17_KEYPOINT_NAMES) else tuple(f"keypoint_{index}" for index in range(shape[0]))
        self.model = model
        self.path = actual_path
        self.info = PoseModelInfo(str(actual_path), sha256_file(actual_path), task, names, shape, keypoint_names)
        return self.info

    def estimate(self, frame_bgr: np.ndarray, frame_index: int, timestamp_s: float) -> list[PoseObservation]:
        self.load()
        assert self.model is not None
        results = self.model.predict(source=frame_bgr, conf=self.confidence, device=self.device, half=self.fp16, verbose=False)
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        data = _as_numpy(getattr(keypoints, "data", None))
        xyxy = _as_numpy(getattr(boxes, "xyxy", None))
        scores = _as_numpy(getattr(boxes, "conf", None))
        if data is None or xyxy is None or scores is None:
            return []
        observations: list[PoseObservation] = []
        for index in range(min(len(data), len(xyxy), len(scores))):
            points = np.asarray(data[index], dtype=np.float32)
            if points.ndim != 2 or points.shape[1] < 3:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy[index][:4])
            observations.append(PoseObservation((x1, y1, x2, y2), points[:, :3], float(scores[index]), frame_index, timestamp_s))
        return observations
