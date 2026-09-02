"""Build untracked GitHub Release assets after an authorised distribution review."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from surgical_pipeline.model_loader import inspect_model, resolve_health_class_id


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Yetkili GitHub Release model paketi oluşturur.")
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--instrument", type=Path, required=True)
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--confirm-distribution-rights", action="store_true")
    args = parser.parse_args()
    if not args.confirm_distribution_rights:
        parser.error("Ağırlık dağıtım hakkını doğrulamadan önce --confirm-distribution-rights kullanılamaz.")
    health, instrument = args.health.resolve(), args.instrument.resolve()
    health_info, instrument_info = inspect_model(health), inspect_model(instrument)
    resolve_health_class_id(health_info.names)
    if health_info.task != "segment" or instrument_info.task != "segment":
        parser.error("İki model de Ultralytics segmentation modeli olmalıdır.")
    root = Path(__file__).resolve().parents[1]
    destination = root / "release_assets" / f"v{args.version}-models"
    destination.mkdir(parents=True, exist_ok=True)
    health_out = destination / "health_personnel_segmentation.pt"
    instrument_out = destination / "surgical_instrument_segmentation.pt"
    shutil.copy2(health, health_out)
    shutil.copy2(instrument, instrument_out)
    manifest = {
        "version": args.version,
        "models": {
            "health_personnel": {"filename": health_out.name, "download_url": "GITHUB_RELEASE_DOWNLOAD_URL", "sha256": checksum(health_out), "task": health_info.task, "expected_classes": [health_info.names[index] for index in sorted(health_info.names)]},
            "surgical_instruments": {"filename": instrument_out.name, "download_url": "GITHUB_RELEASE_DOWNLOAD_URL", "sha256": checksum(instrument_out), "task": instrument_info.task, "expected_classes": [instrument_info.names[index] for index in sorted(instrument_info.names)]},
        },
    }
    (destination / "model_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "SHA256SUMS.txt").write_text(f"{manifest['models']['health_personnel']['sha256']}  {health_out.name}\n{manifest['models']['surgical_instruments']['sha256']}  {instrument_out.name}\n", encoding="utf-8")
    (destination / "MODEL_CARD.md").write_text("# Model Card\n\nBu dosyayı yayın öncesinde eğitim verisi, lisans, amaç, sınıflar, sınırlar ve gizlilik riskleriyle tamamlayın.\n", encoding="utf-8")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
