"""Extract auditable point-prompt candidates from real V1 model masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def _frame(source: Path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(source))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {index}")
        return frame
    finally:
        capture.release()


def _candidates(model: YOLO, frame: np.ndarray, frame_index: int, output: Path, kind: str) -> list[dict[str, object]]:
    result = model.track(frame, persist=False, verbose=False, conf=0.20, iou=0.5)[0]
    preview = frame.copy()
    values: list[dict[str, object]] = []
    if result.boxes is None or result.masks is None:
        return values
    names = result.names
    ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else list(range(1, len(result.boxes) + 1))
    classes = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    boxes = result.boxes.xyxy.cpu().tolist()
    for candidate_index, (track_id, class_id, confidence, box, polygon) in enumerate(
        zip(ids, classes, confidences, boxes, result.masks.xy, strict=True), start=1
    ):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        points = np.rint(polygon).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)
        if not np.any(mask):
            continue
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        x1, y1, x2, y2 = (int(value) for value in box)
        colour = ((37 * candidate_index) % 255, (97 * candidate_index) % 255, (173 * candidate_index) % 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, colour, 2, cv2.LINE_AA)
        cv2.rectangle(preview, (x1, y1), (x2, y2), colour, 2)
        label = f"#{candidate_index} id={track_id} {names[class_id]} {confidence:.3f}"
        cv2.putText(preview, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 2, cv2.LINE_AA)
        cv2.circle(preview, (int(x), int(y)), 6, (0, 255, 255), -1, cv2.LINE_AA)
        values.append({
            "candidate_index": candidate_index, "kind": kind, "frame_index": frame_index,
            "timestamp": frame_index / 30.0, "v1_track_id": int(track_id),
            "class_id": int(class_id), "class_name": str(names[class_id]), "confidence": float(confidence),
            "bbox_xyxy": [x1, y1, x2, y2], "mask_area_pixels": int(np.count_nonzero(mask)),
            "point_x": int(x), "point_y": int(y),
            "normalized_x": float(x / (frame.shape[1] - 1)), "normalized_y": float(y / (frame.shape[0] - 1)),
        })
    cv2.imwrite(str(output / f"detections_{kind}_{frame_index:04d}.png"), preview)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--health-model", type=Path, required=True)
    parser.add_argument("--instrument-model", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    health, instrument = YOLO(args.health_model), YOLO(args.instrument_model)
    results = {
        "instrument_frame_300": _candidates(instrument, _frame(args.source, 300), 300, args.output, "instrument"),
        "person_frame_900": _candidates(health, _frame(args.source, 900), 900, args.output, "person_1"),
        "person_frame_1500": _candidates(health, _frame(args.source, 1500), 1500, args.output, "person_2"),
    }
    (args.output / "v1_candidates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
