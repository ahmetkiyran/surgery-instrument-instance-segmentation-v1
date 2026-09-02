import hashlib
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from surgical_pipeline.model_manager import ModelManager
from surgical_pipeline.schemas import ModelInfo


def manifest(path: Path, url: str, sha256: str, published: bool = True) -> Path:
    payload = {
        "version": "test",
        "release_published": published,
        "models": {
            "health_personnel": {
                "filename": "health.pt",
                "download_url": url,
                "sha256": sha256,
                "task": "segment",
                "expected_classes": ["health_personel"],
            },
            "surgical_instruments": {
                "filename": "instrument.pt",
                "download_url": url,
                "sha256": sha256,
                "task": "segment",
                "expected_classes": ["tool"],
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_uses_real_discovered_class_lists() -> None:
    root = Path(__file__).resolve().parents[1]
    manager = ModelManager(root)
    specs = manager.specs()
    assert specs["health_personnel"].expected_classes == ("health_personel", "light", "monitor")
    assert specs["surgical_instruments"].expected_classes == ("skin_stapler", "thumb_forceps", "ring_instrument", "scissors")


def test_manager_detects_missing_and_partial_download(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"; source.write_bytes(b"model")
    manifest_path = manifest(tmp_path / "manifest.json", "https://example.invalid/model.pt", hashlib.sha256(b"model").hexdigest())
    manager = ModelManager(tmp_path, manifest_path, tmp_path / "weights")
    spec = manager.specs()["health_personnel"]
    assert manager.verify_spec(spec).state == "missing"
    manager.local_path(spec).parent.mkdir()
    manager.local_path(spec).with_suffix(".pt.part").write_bytes(b"partial")
    assert manager.verify_spec(spec).state == "partial"


def test_download_is_hashed_then_atomically_validated(tmp_path: Path, monkeypatch) -> None:
    payload = b"verified-release-asset"
    (tmp_path / "asset.pt").write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A003
            pass

    previous = Path.cwd()
    try:
        # The HTTP server exposes only the test directory and never an external network resource.
        import os

        os.chdir(tmp_path)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        url = f"http://127.0.0.1:{server.server_port}/asset.pt"
        manager = ModelManager(tmp_path, manifest(tmp_path / "manifest.json", url, digest), tmp_path / "weights")
        monkeypatch.setattr(
            "surgical_pipeline.model_manager.inspect_model",
            lambda path: ModelInfo(str(path), digest, {0: "health_personel"} if path.name == "health.pt" else {0: "tool"}, "segment"),
        )
        assert manager.download_all()[0].read_bytes() == payload
        assert all(status.ready for status in manager.statuses())
    finally:
        if "server" in locals():
            server.shutdown(); server.server_close()
        os.chdir(previous)
