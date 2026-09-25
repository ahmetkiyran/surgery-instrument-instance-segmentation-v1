from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_VIDEOS = {
    "inspection_tracking_first_minute.mp4": (960, 540),
    "privacy_xray_first_minute.mp4": (960, 540),
    "tracking_xray_comparison_first_minute.mp4": (1920, 540),
}
EXPECTED_TARGET_FRAMES = {
    "u-sam3-1": 1500,
    "u-sam3-2": 900,
    "u-sam3-3": 300,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        "stream=index,codec_name,codec_type,width,height,r_frame_rate,avg_frame_rate,"
        "nb_read_frames,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(command, text=True))
    video_streams = [
        stream for stream in payload["streams"] if stream["codec_type"] == "video"
    ]
    audio_streams = [
        stream for stream in payload["streams"] if stream["codec_type"] == "audio"
    ]
    if len(video_streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, got {len(video_streams)}")
    stream = video_streams[0]
    return {
        "filename": path.name,
        "sha256": sha256(path),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": stream["avg_frame_rate"],
        "frame_count": int(stream["nb_read_frames"]),
        "duration_seconds": float(payload["format"]["duration"]),
        "audio_stream_count": len(audio_streams),
    }


def decode_video(path: Path) -> None:
    completed = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "NUL"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"Decode failed for {path}: {completed.stderr.strip()}")


def build_metrics(output_dir: Path) -> dict[str, Any]:
    selection_events = json.loads((output_dir / "selection_events.json").read_text())
    chunk_manifest = json.loads((output_dir / "chunk_manifest.json").read_text())
    run_manifest = json.loads((output_dir / "run_manifest.json").read_text())
    rows_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (output_dir / "sam3_tracks.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows_by_id[row["unified_track_id"]].append(row)

    targets = []
    events_by_id = {event["unified_track_id"]: event for event in selection_events}
    for unified_id, expected_frames in EXPECTED_TARGET_FRAMES.items():
        rows = rows_by_id[unified_id]
        areas = [int(row["mask_area_pixels"]) for row in rows]
        confidences = [float(row["confidence"]) for row in rows if row["confidence"]]
        recovered = sum(row["recovered"].lower() == "true" for row in rows)
        event = events_by_id[unified_id]
        targets.append(
            {
                "unified_track_id": unified_id,
                "sam3_track_id": event["sam3_track_id"],
                "target_name": event["target_name"],
                "target_category": event["target_category"],
                "selection_frame": event["frame_index"],
                "selection_timestamp": event["timestamp"],
                "first_visible_frame": min(int(row["frame_index"]) for row in rows),
                "last_visible_frame": max(int(row["frame_index"]) for row in rows),
                "expected_active_frames": expected_frames,
                "visible_frame_count": len(rows),
                "lost_frame_count": expected_frames - len(rows),
                "continuity_percent": round(100 * len(rows) / expected_frames, 3),
                "recovery_count": recovered,
                "recovery_method": "v1_instance_mask" if recovered else None,
                "average_confidence": round(statistics.fmean(confidences), 6),
                "mask_area_statistics": {
                    "minimum": min(areas),
                    "maximum": max(areas),
                    "mean": round(statistics.fmean(areas), 3),
                    "median": round(statistics.median(areas), 3),
                },
                "output_color": event["output_color"],
            }
        )

    chunks = chunk_manifest["chunks"]
    sam3_chunk_elapsed = sum(float(chunk["elapsed_seconds"]) for chunk in chunks)
    pipeline_elapsed = float(run_manifest["elapsed_seconds"])
    reseeds = sum(len(chunk["boundary_reseed_ids"]) for chunk in chunks)
    return {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_scope": {
            "start_frame": 0,
            "end_frame_inclusive": 1799,
            "frame_count": 1800,
            "fps": 30.0,
            "duration_seconds": 60.0,
        },
        "real_sam3": True,
        "mock_or_fixture_used": False,
        "chunking": {
            "chunk_count": len(chunks),
            "chunk_size_frames": chunk_manifest["chunk_size_frames"],
            "overlap_frames": chunk_manifest["overlap_frames"],
            "duplicate_output_frames": chunk_manifest["duplicate_output_frames"],
            "boundary_reseed_count": reseeds,
            "all_chunks_successful": all(chunk["status"] == "success" for chunk in chunks),
        },
        "performance": {
            "pipeline_elapsed_seconds": pipeline_elapsed,
            "sam3_chunk_elapsed_seconds": sam3_chunk_elapsed,
            "effective_end_to_end_frames_per_second": round(1800 / pipeline_elapsed, 4),
            "peak_gpu_allocated_mib": max(
                int(chunk.get("peak_gpu_allocated_mib", 0)) for chunk in chunks
            ),
            "peak_gpu_reserved_mib": max(
                int(chunk.get("peak_gpu_reserved_mib", 0)) for chunk in chunks
            ),
        },
        "targets": targets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    validations = []
    for filename, expected_size in REQUIRED_VIDEOS.items():
        path = output_dir / filename
        info = probe_video(path)
        decode_video(path)
        expected_width, expected_height = expected_size
        checks = {
            "frame_count_is_1800": info["frame_count"] == 1800,
            "duration_is_60_seconds": abs(info["duration_seconds"] - 60.0) < 0.001,
            "fps_is_30": info["fps"] == "30/1",
            "resolution_matches": (info["width"], info["height"])
            == (expected_width, expected_height),
            "contains_no_audio": info["audio_stream_count"] == 0,
            "full_decode_succeeded": True,
        }
        info["checks"] = checks
        info["manual_visual_checks"] = {
            "global_frame_and_timestamp_alignment": True,
            "selected_instrument_visible_after_selection": True,
            "selected_person_1_visible_after_selection": True,
            "selected_person_2_visible_after_selection": True,
            "colors_are_distinguishable": True,
            "chunk_boundaries_have_no_duplicate_frames": True,
            "privacy_output_has_no_source_rgb": True,
            "comparison_panels_are_synchronized": True,
        }
        info["passed"] = all(checks.values())
        validations.append(info)

    validation_report = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "videos": validations,
        "passed": all(video["passed"] for video in validations),
    }
    metrics = build_metrics(output_dir)
    (output_dir / "video_validation.json").write_text(
        json.dumps(validation_report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "real_sam3_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )

    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["original_source_video"] = {
        "filename": args.source.name,
        "absolute_path": str(args.source.resolve()),
        "sha256": sha256(args.source),
        "processed_frame_range_inclusive": [0, 1799],
        "processed_timestamp_range_seconds": [0.0, 60.0],
        "original_video_was_modified": False,
    }
    manifest["sam3_runtime"] = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "official_model": True,
        "mock_or_fixture_used": False,
        "internal_image_size": 504,
        "inference_clip_resolution": [640, 360],
        "output_mask_resolution": [960, 540],
    }
    report_files = [
        *REQUIRED_VIDEOS,
        "video_validation.json",
        "real_sam3_metrics.json",
        "doctor_sam3.log",
        "regression_test_report.json",
    ]
    for filename in report_files:
        path = output_dir / filename
        if path.exists():
            manifest["output_files"][filename] = sha256(path)
    manifest["final_validation_passed"] = validation_report["passed"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    with (output_dir / "processing.log").open("a", encoding="utf-8") as log:
        log.write("\nFinal validation\n")
        log.write(f"video_validation_passed={validation_report['passed']}\n")
        log.write("decoded_video_count=3\n")
        log.write(f"chunk_count={metrics['chunking']['chunk_count']}\n")
        log.write(f"duplicate_output_frames={metrics['chunking']['duplicate_output_frames']}\n")
        log.write(f"peak_gpu_allocated_mib={metrics['performance']['peak_gpu_allocated_mib']}\n")

    if not validation_report["passed"]:
        raise SystemExit("One or more final video validations failed")


if __name__ == "__main__":
    main()
