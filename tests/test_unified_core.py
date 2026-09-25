from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import cv2

from surgical_pipeline.schemas import Detection
from surgical_pipeline.unified import PrivacyXRayRenderer, SelectionEvent, UnifiedTrackMatcher
from surgical_pipeline.unified import SAM3Bridge, V1SelectionRecovery, _write_comparison_video


def test_normalized_selection_maps_to_original_source_coordinates() -> None:
    event = SelectionEvent(.5, .25, 7, .7, 200, 100, displayed_width=100, displayed_height=50)
    assert event.pixel(200, 100) == (100, 25)


def test_unified_matcher_links_v1_and_sam3_using_mask_overlap() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 3:8] = 1
    person = Detection(0, "health_personnel", .9, 12, mask, 0, 0.0, (3, 2, 8, 7))
    tracks = UnifiedTrackMatcher().match([person], [{"sam3_track_id": 3, "category": "person", "mask": mask, "bounding_box": (3, 2, 8, 7)}], 0)
    assert len(tracks) == 1
    assert tracks[0].v1_track_id == 12
    assert tracks[0].sam3_track_id == 3


def test_privacy_renderer_cannot_receive_original_rgb_frame() -> None:
    assert "original_frame" not in inspect.signature(PrivacyXRayRenderer.render).parameters


def test_privacy_renderer_has_clean_clone_builtin_fallback(tmp_path: Path) -> None:
    points = np.zeros((17, 3), dtype=float)
    points[:, :2] = (16, 12)
    points[:, 2] = .9
    pose = SimpleNamespace(pose_track_id=1, keypoints=points, v1_track_id=None)
    renderer = PrivacyXRayRenderer(tmp_path)
    canvas = renderer.render((32, 24), [pose], [], set(), [], 0, [])
    assert renderer.backend == "builtin"
    assert canvas.shape == (24, 32, 3)
    assert np.all(canvas[:, :, 0] >= 0)


def test_session_api_records_normalized_selection(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from surgical_pipeline.server.app import create_app
    from surgical_pipeline.server.config import ServerSettings

    app = create_app(ServerSettings(project_root=Path.cwd(), data_root_override=tmp_path, token="x" * 32))
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer " + "x" * 32}
        created = client.post("/api/v1/sessions", headers=headers)
        assert created.status_code == 201
        session_id = created.json()["session_id"]
        payload = {"normalized_x": .5, "normalized_y": .5, "displayed_width": 320, "displayed_height": 180, "source_width": 1920, "source_height": 1080, "frame_index": 12, "timestamp": .4}
        assert client.post(f"/api/v1/sessions/{session_id}/selections", json=payload, headers=headers).status_code == 200
        events = client.get(f"/api/v1/sessions/{session_id}/events", headers=headers)
        assert events.status_code == 200
        assert events.json()[0]["frame_index"] == 12


def test_session_clean_preview_and_selection_do_not_draw_automatic_overlay(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from surgical_pipeline.server.app import create_app
    from surgical_pipeline.server.config import ServerSettings

    video = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    writer.write(np.full((24, 32, 3), (30, 80, 140), dtype=np.uint8)); writer.release()
    token = "t" * 32; headers = {"Authorization": f"Bearer {token}"}
    app = create_app(ServerSettings(project_root=Path.cwd(), data_root_override=tmp_path / "server", token=token))
    with TestClient(app) as client:
        session = client.post("/api/v1/sessions", headers=headers).json()["session_id"]
        upload = client.post("/api/v1/uploads", content=video.read_bytes(), headers=headers | {"X-File-Name": "fixture.mp4", "Content-Type": "application/octet-stream"}).json()
        attached = client.post(f"/api/v1/sessions/{session}/video", json={"upload_id": upload["upload_id"], "options": {}}, headers=headers)
        assert attached.status_code == 200
        clean = client.get(f"/api/v1/sessions/{session}/preview?frame_index=0&overlay=clean", headers=headers)
        assert clean.status_code == 200 and clean.headers["content-type"] == "image/jpeg"
        payload = {"normalized_x": .5, "normalized_y": .5, "displayed_width": 32, "displayed_height": 24, "source_width": 32, "source_height": 24, "frame_index": 0, "timestamp": 0}
        selection = client.post(f"/api/v1/sessions/{session}/selections", json=payload, headers=headers)
        assert selection.status_code == 200
        assert selection.json()["status"] == "selection_recorded"


def test_interactive_selection_is_non_blocking_and_assigns_stable_target_id(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from surgical_pipeline.server.app import create_app
    from surgical_pipeline.server.config import ServerSettings

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("SAM3 propagation must start only with the session job")

    monkeypatch.setattr("surgical_pipeline.unified.SAM3Bridge.track", fail_if_called)
    video = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    writer.write(np.zeros((24, 32, 3), dtype=np.uint8)); writer.release()
    token = "i" * 32; headers = {"Authorization": f"Bearer {token}"}
    app = create_app(ServerSettings(project_root=Path.cwd(), data_root_override=tmp_path / "server", token=token))
    with TestClient(app) as client:
        session = client.post("/api/v1/sessions", headers=headers).json()["session_id"]
        upload = client.post("/api/v1/uploads", content=video.read_bytes(), headers=headers | {"X-File-Name": "fixture.mp4", "Content-Type": "application/octet-stream"}).json()
        assert client.post(f"/api/v1/sessions/{session}/video", json={"upload_id": upload["upload_id"], "options": {}}, headers=headers).status_code == 200
        payload = {"normalized_x": .25, "normalized_y": .5, "displayed_width": 32, "displayed_height": 24, "source_width": 32, "source_height": 24, "frame_index": 3, "timestamp": .6}
        response = client.post(f"/api/v1/sessions/{session}/selections", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "selection_recorded"
        assert response.json()["active_tracks"][0]["sam3_track_id"] == 1


def test_session_privacy_preview_does_not_open_source_rgb(tmp_path: Path, monkeypatch) -> None:
    from fastapi.testclient import TestClient
    from surgical_pipeline.server.app import create_app
    from surgical_pipeline.server.config import ServerSettings

    video = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    writer.write(np.full((24, 32, 3), (30, 80, 140), dtype=np.uint8)); writer.release()
    token = "p" * 32; headers = {"Authorization": f"Bearer {token}"}
    app = create_app(ServerSettings(project_root=Path.cwd(), data_root_override=tmp_path / "server", token=token))
    with TestClient(app) as client:
        session = client.post("/api/v1/sessions", headers=headers).json()["session_id"]
        upload = client.post("/api/v1/uploads", content=video.read_bytes(), headers=headers | {"X-File-Name": "fixture.mp4", "Content-Type": "application/octet-stream"}).json()
        assert client.post(f"/api/v1/sessions/{session}/video", json={"upload_id": upload["upload_id"], "options": {}}, headers=headers).status_code == 200
        original_capture = cv2.VideoCapture
        monkeypatch.setattr(cv2, "VideoCapture", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("privacy preview opened source RGB")))
        response = client.get(f"/api/v1/sessions/{session}/preview?frame_index=0&overlay=privacy-xray", headers=headers)
        monkeypatch.setattr(cv2, "VideoCapture", original_capture)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"


def test_job_pause_resume_blocks_checkpoint_and_cancel_is_idempotent(tmp_path: Path) -> None:
    import threading
    import time
    from surgical_pipeline.server.config import ServerSettings
    from surgical_pipeline.server.jobs import JobManager, JobRecord
    from surgical_pipeline.server.schemas import JobCreateRequest

    manager = JobManager(ServerSettings(project_root=Path.cwd(), data_root_override=tmp_path / "server"))
    request = JobCreateRequest(local_path=str(tmp_path / "video.mp4"))
    record = JobRecord("job-test", tmp_path / "video.mp4", request, tmp_path / "job")
    manager._jobs[record.job_id] = record
    manager.pause(record.job_id)
    finished = threading.Event()

    def checkpoint() -> None:
        manager._checkpoint(record)
        finished.set()

    worker = threading.Thread(target=checkpoint)
    worker.start()
    time.sleep(0.05)
    assert not finished.is_set()
    manager.resume(record.job_id)
    worker.join(timeout=1)
    assert finished.is_set()
    manager.cancel(record.job_id)
    manager.cancel(record.job_id)
    assert record.cancel_event.is_set()


def test_sam3_cache_preserves_two_targets_and_replaces_only_same_target() -> None:
    cache: dict[int, list[dict]] = {}
    SAM3Bridge._merge(cache, 100, {"sam3_track_id": 1, "category": "person"})
    SAM3Bridge._merge(cache, 200, {"sam3_track_id": 1, "category": "person"})
    SAM3Bridge._merge(cache, 200, {"sam3_track_id": 2, "category": "instrument"})
    SAM3Bridge._merge(cache, 200, {"sam3_track_id": 1, "category": "person-updated"})
    assert cache[100] == [{"sam3_track_id": 1, "category": "person"}]
    assert {item["sam3_track_id"] for item in cache[200]} == {1, 2}
    assert next(item for item in cache[200] if item["sam3_track_id"] == 1)["category"] == "person-updated"


def test_sam3_chunk_ranges_cover_first_minute_once_with_overlap() -> None:
    ranges = SAM3Bridge.chunk_ranges(1800, 300, 30)
    assert ranges == [(0, 300, 0), (270, 570, 300), (540, 840, 570), (810, 1110, 840), (1080, 1380, 1110), (1350, 1650, 1380), (1620, 1800, 1650)]
    emitted = [frame for start, end, write_start in ranges for frame in range(max(start, write_start), end)]
    assert emitted == list(range(1800))


def test_sam3_prompt_point_is_inside_mask() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:16, 8:24] = 1
    point = SAM3Bridge._prompt_point(mask)
    assert point is not None
    assert mask[point[1], point[0]] == 1


def test_comparison_video_stays_frame_synchronized(tmp_path: Path) -> None:
    left, right = tmp_path / "left.mp4", tmp_path / "right.mp4"
    for path, colour in ((left, (20, 40, 60)), (right, (70, 90, 110))):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        for _ in range(7):
            writer.write(np.full((48, 64, 3), colour, dtype=np.uint8))
        writer.release()
    destination = _write_comparison_video(left, right, tmp_path / "comparison.mp4")
    capture = cv2.VideoCapture(str(destination))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 128
    capture.release()


def test_v1_recovery_replaces_obvious_instrument_drift() -> None:
    event = SelectionEvent(.5, .5, 10, 1.0, 100, 100, target_category="instrument", target_name="selected_instrument", sam3_track_id=1, unified_track_id="u-sam3-1", initial_class_name="ring_instrument")
    candidate_mask = np.zeros((100, 100), dtype=np.uint8); candidate_mask[45:55, 45:60] = 1
    candidate = Detection(0, "ring_instrument", .9, 4, candidate_mask, 10, 1.0, (45, 45, 60, 55))
    drift_mask = np.zeros((100, 100), dtype=np.uint8); drift_mask[10:90, 10:90] = 1
    raw = [{"sam3_track_id": 1, "mask": drift_mask, "mask_area_pixels": 6400, "center": (20., 20.), "visible": True}]
    recovered = V1SelectionRecovery([event], 100, 100).apply(raw, [candidate], 10)
    assert recovered[0]["recovered"] is True
    assert recovered[0]["recovery_method"] == "v1_instance_mask"
    assert recovered[0]["unified_track_id"] == "u-sam3-1"
    assert recovered[0]["mask_area_pixels"] == 150


def test_server_rejects_uploads_longer_than_configured_duration(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient
    from surgical_pipeline.server.app import create_app
    from surgical_pipeline.server.config import ServerSettings

    video = tmp_path / "long.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 5, (32, 24))
    for _ in range(10):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    token = "d" * 32
    app = create_app(ServerSettings(project_root=Path.cwd(), data_root_override=tmp_path / "server", token=token, max_upload_duration_seconds=1.0))
    with TestClient(app) as client:
        response = client.post("/api/v1/uploads", content=video.read_bytes(), headers={"Authorization": f"Bearer {token}", "X-File-Name": "long.mp4"})
    assert response.status_code == 413
    assert not list((tmp_path / "server" / "uploads").glob("*"))
