"""Video metadata, timestamp-aware iteration and safe H.264/audio output."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2

from .utils import tool_available


class VideoError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    variable_timestamps: bool

    def safe_dict(self) -> dict[str, int | float | bool]:
        return {"width": self.width, "height": self.height, "fps": self.fps, "frame_count": self.frame_count, "duration_seconds": self.duration_s, "variable_timestamps_detected": self.variable_timestamps}


def read_metadata(path: Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError("Video açılamadı veya desteklenmeyen bir codec kullanıyor.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise VideoError("Video metadata bilgisi geçersiz (FPS veya çözünürlük okunamadı).")
    return VideoMetadata(width, height, fps, frame_count, frame_count / fps if frame_count else 0.0, False)


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise VideoError("Video açılamadı.")
        self.metadata = read_metadata(path)
        self._last_timestamp = -1.0
        self.variable_timestamps = False

    def __iter__(self):
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            pos_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            fallback = index / self.metadata.fps
            timestamp = pos_ms / 1000.0 if pos_ms > 0 else fallback
            if index > 0 and abs((timestamp - self._last_timestamp) - 1 / self.metadata.fps) > 0.01:
                self.variable_timestamps = True
            self._last_timestamp = timestamp
            yield index, timestamp, frame
            index += 1

    def close(self) -> None:
        self.capture.release()


class VideoWriter:
    """Write an intermediate stream, then rely on FFmpeg to make browser-ready H.264."""

    def __init__(self, temporary_path: Path, metadata: VideoMetadata) -> None:
        self.temporary_path = temporary_path
        self.metadata = metadata
        self.writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), metadata.fps, (metadata.width, metadata.height))
        if not self.writer.isOpened():
            raise VideoError("Geçici video yazıcısı açılamadı.")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    def finalise(self, source_video: Path, destination: Path, keep_audio: bool) -> list[str]:
        warnings: list[str] = []
        if not tool_available("ffmpeg"):
            self.temporary_path.replace(destination)
            return ["FFmpeg bulunamadı; H.264 ve ses mux işlemi yapılamadı. MP4V fallback kullanıldı."]
        command = ["ffmpeg", "-y", "-i", str(self.temporary_path)]
        if keep_audio:
            command += ["-i", str(source_video), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", "-shortest", str(destination)]
        else:
            command += ["-map", "0:v:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            # A documented, safe fallback; still leaves the source untouched.
            self.temporary_path.replace(destination)
            warnings.append("FFmpeg H.264 mux başarısız oldu; MP4V fallback üretildi. Ayrıntı log dosyasındadır.")
        elif self.temporary_path.exists():
            self.temporary_path.unlink()
        return warnings
