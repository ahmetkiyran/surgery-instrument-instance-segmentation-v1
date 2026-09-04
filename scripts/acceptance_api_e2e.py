"""Bounded local API acceptance checks used by the Windows runner.

Desktop and mobile modes exercise their respective API contracts, but deliberately do
not claim native Tauri or Android UI automation.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class JobPollingTimeout(TimeoutError):
    """Keep the last safe job response when a bounded E2E poll expires."""

    def __init__(self, last_job: dict[str, Any], observed_states: list[str]) -> None:
        super().__init__("Analysis job reached its timeout")
        self.last_job = last_job
        self.observed_states = observed_states


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def request(port: int, path: str, token: str | None, method: str = "GET", body: bytes | None = None, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict[str, str]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    active_headers = dict(headers or {})
    if token:
        active_headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        active_headers.setdefault("Content-Length", str(len(body)))
    connection.request(method, path, body=body, headers=active_headers)
    response = connection.getresponse()
    data = response.read()
    status = response.status
    response_headers = {key.lower(): value for key, value in response.getheaders()}
    connection.close()
    return status, data, response_headers


def request_json(port: int, path: str, token: str | None, method: str = "GET", payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any] | list[Any]]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    status, raw, _ = request(port, path, token, method, body, {"Content-Type": "application/json"} if body else None)
    try:
        return status, json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return status, {"raw": raw.decode("utf-8", "replace")[:500]}


def upload_stream(port: int, token: str, source: Path) -> tuple[int, dict[str, Any] | list[Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    connection.putrequest("POST", "/api/v1/uploads")
    connection.putheader("Authorization", f"Bearer {token}")
    connection.putheader("Content-Type", "application/octet-stream")
    connection.putheader("X-File-Name", source.name)
    connection.putheader("Content-Length", str(source.stat().st_size))
    connection.endheaders()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            connection.send(chunk)
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    connection.close()
    return status, json.loads(raw.decode("utf-8"))


def wait_for_health(port: int, token: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = request_json(port, "/api/v1/health", token)
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.25)
    raise TimeoutError("Local API did not become ready")


def poll_job(port: int, token: str, job_id: str, timeout_seconds: int) -> tuple[dict[str, Any], list[str]]:
    deadline = time.monotonic() + timeout_seconds
    observed: list[str] = []
    last_job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status, data = request_json(port, f"/api/v1/jobs/{job_id}", token)
        if status != 200 or not isinstance(data, dict):
            raise RuntimeError(f"Job polling returned HTTP {status}")
        last_job = data
        state = str(data.get("status", "unknown"))
        if not observed or observed[-1] != state:
            observed.append(state)
        if state in {"completed", "failed", "cancelled"}:
            return data, observed
        time.sleep(2)
    raise JobPollingTimeout(last_job, observed)


def job_options(output_directory: Path | None) -> dict[str, Any]:
    options: dict[str, Any] = {
        "privacy_mode": "skeleton-only",
        "device": "auto",
        "depth_enabled": False,
        "tracker": "botsort",
        "health_confidence": 0.35,
        "instrument_confidence": 0.25,
        "pose_confidence": 0.25,
        "iou": 0.45,
    }
    if output_directory:
        options["output_directory"] = str(output_directory.resolve())
    return options


def artifacts_and_privacy(port: int, token: str, job_id: str) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    status, data = request_json(port, f"/api/v1/jobs/{job_id}/artifacts", token)
    if status != 200 or not isinstance(data, list):
        raise RuntimeError("Artifact listing failed")
    artifacts = [item for item in data if isinstance(item, dict)]
    privacy: dict[str, Any] | None = None
    for item in artifacts:
        if item.get("filename") == "privacy_report.json":
            code, raw, _ = request(port, f"/api/v1/jobs/{job_id}/artifacts/{item['artifact_id']}", token)
            if code == 200:
                privacy = json.loads(raw.decode("utf-8"))
            break
    return artifacts, privacy


def static_checks(port: int, token: str) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for endpoint in ("/api/v1/health", "/api/v1/system/doctor", "/api/v1/models", "/api/v1/config/defaults"):
        status, data = request_json(port, endpoint, token)
        checks[endpoint] = {"status": status, "passed": status == 200, "summary_type": type(data).__name__}
    invalid_status, _ = request_json(port, "/api/v1/health", "invalid-token")
    checks["invalid_token"] = {"status": invalid_status, "passed": invalid_status == 401}
    missing_status, _ = request_json(port, "/api/v1/jobs/not-a-real-job", token)
    checks["invalid_job"] = {"status": missing_status, "passed": missing_status == 404}
    traversal_status, _ = request_json(
        port,
        "/api/v1/jobs",
        token,
        "POST",
        {"local_path": "..\\not-a-video.mp4", "options": job_options(None)},
    )
    checks["path_traversal"] = {"status": traversal_status, "passed": traversal_status in {403, 422}}
    return checks


def run_desktop(port: int, token: str, source: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    status, first = request_json(port, "/api/v1/jobs", token, "POST", {"local_path": str(source.resolve()), "options": job_options(output)})
    if status != 202 or not isinstance(first, dict):
        raise RuntimeError(f"Desktop job creation failed: HTTP {status}")
    first_id = str(first["job_id"])
    queue_status, second = request_json(port, "/api/v1/jobs", token, "POST", {"local_path": str(source.resolve()), "options": job_options(output)})
    queue_cancelled = False
    if queue_status == 202 and isinstance(second, dict):
        cancel_status, cancelled = request_json(port, f"/api/v1/jobs/{second['job_id']}/cancel", token, "POST")
        queue_cancelled = cancel_status == 200 and isinstance(cancelled, dict) and cancelled.get("status") == "cancelled"
    try:
        completed, observed = poll_job(port, token, first_id, 40 * 60)
    except JobPollingTimeout as error:
        return {
            "status": "FAILED",
            "test_kind": "FastAPI desktop-contract integration; native Tauri UI automation was not run.",
            "job_status": error.last_job.get("status", "timed_out"),
            "observed_states": error.observed_states,
            "queue_second_job_cancelled": queue_cancelled,
            "error": str(error),
            "last_job": error.last_job,
        }
    artifacts, privacy = artifacts_and_privacy(port, token, first_id) if completed.get("status") == "completed" else ([], None)
    invalid_artifact_status, _, _ = request(port, f"/api/v1/jobs/{first_id}/artifacts/not-a-real-artifact", token)
    return {
        "status": "PASSED" if completed.get("status") == "completed" and any(item.get("kind") == "skeleton_video" for item in artifacts) else "FAILED",
        "test_kind": "FastAPI desktop-contract integration; native Tauri UI automation was not run.",
        "job_status": completed.get("status"),
        "observed_states": observed,
        "queue_second_job_cancelled": queue_cancelled,
        "invalid_artifact_status": invalid_artifact_status,
        "artifacts": artifacts,
        "privacy_report": privacy,
    }


def run_mobile(port: int, token: str, source: Path) -> dict[str, Any]:
    upload_status, upload = upload_stream(port, token, source)
    if upload_status != 201 or not isinstance(upload, dict):
        raise RuntimeError(f"Streaming upload failed: HTTP {upload_status}")
    status, job = request_json(port, "/api/v1/jobs", token, "POST", {"upload_id": upload["upload_id"], "options": job_options(None)})
    if status != 202 or not isinstance(job, dict):
        raise RuntimeError(f"Mobile job creation failed: HTTP {status}")
    completed, observed = poll_job(port, token, str(job["job_id"]), 45 * 60)
    artifacts, privacy = artifacts_and_privacy(port, token, str(job["job_id"])) if completed.get("status") == "completed" else ([], None)
    return {
        "status": "PASSED" if completed.get("status") == "completed" and any(item.get("kind") == "skeleton_video" for item in artifacts) else "FAILED",
        "test_kind": "Streaming-upload API integration representing the mobile contract; Android UI automation was not run.",
        "upload_size_bytes": upload.get("size_bytes"),
        "job_status": completed.get("status"),
        "observed_states": observed,
        "artifacts": artifacts,
        "privacy_report": privacy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--mode", choices=("static", "desktop", "mobile"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-log", type=Path, help="Optional local Uvicorn/pipeline diagnostic log.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = secrets.token_urlsafe(36)
    port = choose_port()
    environment = os.environ.copy()
    environment["SURGICAL_API_DATA_ROOT"] = str(args.data_root.resolve())
    command = [sys.executable, "-m", "surgical_pipeline.server", "--host", "127.0.0.1", "--port", str(port), "--token", token, "--log-level", "warning"]
    server_log_handle = None
    if args.server_log:
        args.server_log.parent.mkdir(parents=True, exist_ok=True)
        server_log_handle = args.server_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=args.project_root,
        env=environment,
        stdout=server_log_handle or subprocess.DEVNULL,
        stderr=subprocess.STDOUT if server_log_handle else subprocess.DEVNULL,
    )
    started = time.time()
    payload: dict[str, Any] = {"mode": args.mode, "status": "FAILED", "started_at": started}
    try:
        wait_for_health(port, token)
        payload["api_checks"] = static_checks(port, token)
        if not all(check["passed"] for check in payload["api_checks"].values()):
            raise RuntimeError("One or more static API checks failed")
        if args.mode == "desktop":
            if not args.input or not args.output:
                raise ValueError("Desktop mode requires --input and --output")
            payload["e2e"] = run_desktop(port, token, args.input.resolve(), args.output.resolve())
            payload["status"] = payload["e2e"]["status"]
        elif args.mode == "mobile":
            if not args.input:
                raise ValueError("Mobile mode requires --input")
            payload["e2e"] = run_mobile(port, token, args.input.resolve())
            payload["status"] = payload["e2e"]["status"]
        else:
            payload["status"] = "PASSED"
    except Exception as error:
        payload["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        payload["finished_at"] = time.time()
        payload["elapsed_seconds"] = round(payload["finished_at"] - started, 3)
        write_json(args.result, payload)
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        if server_log_handle:
            server_log_handle.close()

    return 0 if payload["status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
