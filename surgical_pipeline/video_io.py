"""Video metadata, timestamp-aware iteration and safe H.264/audio output."""

from __future__ import annotations

import os
import shutil
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
    if not path.is_file():
        raise VideoError("Girdi videosu bulunamadı.")
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


def ensure_output_space(output_dir: Path, source_size_bytes: int, metadata: VideoMetadata) -> None:
    """Conservative early warning before a long run fills the output volume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    # Re-encoding normally needs one intermediate plus one final stream. Source size
    # is a conservative portable proxy when an exact bitrate is unavailable.
    required = max(source_size_bytes * 2, metadata.width * metadata.height * 3 * max(metadata.frame_count, 1) // 40)
    if free < required:
        raise VideoError(f"Çıktı diski için yeterli alan yok (yaklaşık {required} bayt gerekli, {free} bayt boş).")


class VideoReader:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise VideoError("Video açılamadı.")
        self.metadata = read_metadata(path)
        self._last_timestamp = -1.0
        self.variable_timestamps = False
        self.timestamp_method = "opencv_pts"

    def __iter__(self):
        index = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            pos_ms = float(self.capture.get(cv2.CAP_PROP_POS_MSEC))
            fallback = index / self.metadata.fps
            if pos_ms > 0 or index == 0:
                timestamp = pos_ms / 1000.0
            else:
                timestamp = fallback
                self.timestamp_method = "opencv_pts_with_fps_fallback"
            if index > 0 and abs((timestamp - self._last_timestamp) - 1 / self.metadata.fps) > 0.01:
                self.variable_timestamps = True
            self._last_timestamp = timestamp
            yield index, timestamp, frame
            index += 1

    def close(self) -> None:
        self.capture.release()


class VideoWriter:
    """Write an intermediate stream, then rely on FFmpeg to make browser-ready H.264."""

    def __init__(self, temporary_path: Path, metadata: VideoMetadata, codec: str = "mp4v") -> None:
        self.temporary_path = temporary_path
        self.metadata = metadata
        self.frame_count = 0
        fourcc = "avc1" if codec.casefold() in {"avc1", "h264"} else "mp4v"
        self.writer = cv2.VideoWriter(str(temporary_path), cv2.VideoWriter_fourcc(*fourcc), metadata.fps, (metadata.width, metadata.height))
        if not self.writer.isOpened():
            raise VideoError("Geçici video yazıcısı açılamadı.")

    def write(self, frame) -> None:
        if frame is None or frame.shape[:2] != (self.metadata.height, self.metadata.width):
            raise VideoError("Yazılacak frame boyutu video metadata'sıyla uyumlu değil.")
        self.writer.write(frame)
        self.frame_count += 1

    def close(self) -> None:
        self.writer.release()

    @staticmethod
    def _verify(path: Path, metadata: VideoMetadata, minimum_frames: int) -> None:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise VideoError("Yazılan çıktı videosu tekrar açılamadı.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if (width, height) != (metadata.width, metadata.height) or count < minimum_frames:
            raise VideoError("Yazılan çıktı videosunun frame sayısı veya çözünürlüğü doğrulanamadı.")

    def abort(self) -> None:
        self.close()
        self.temporary_path.unlink(missing_ok=True)

    def finalise(self, source_video: Path, destination: Path, keep_audio: bool) -> list[str]:
        warnings: list[str] = []
        if self.frame_count == 0:
            raise VideoError("Hiç frame yazılmadığı için çıktı videosu tamamlanamadı.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate = destination.with_name(destination.stem + ".partial" + destination.suffix)
        candidate.unlink(missing_ok=True)
        if not tool_available("ffmpeg"):
            os.replace(self.temporary_path, candidate)
            warnings.append("FFmpeg bulunamadı; H.264 ve ses mux işlemi yapılamadı. MP4V fallback kullanıldı.")
        else:
            command = ["ffmpeg", "-y", "-i", str(self.temporary_path)]
            if keep_audio:
                command += ["-i", str(source_video), "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", "-shortest", str(candidate)]
            else:
                command += ["-map", "0:v:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(candidate)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
            except subprocess.TimeoutExpired as error:
                if self.temporary_path.exists():
                    os.replace(self.temporary_path, candidate)
                else:
                    raise VideoError("FFmpeg çıktı dönüştürme zaman aşımına uğradı.") from error
                warnings.append("FFmpeg H.264 mux zaman aşımına uğradı; MP4V fallback kullanıldı.")
                result = None
            if result is not None and result.returncode != 0:
                os.replace(self.temporary_path, candidate)
                warnings.append("FFmpeg H.264 mux başarısız oldu; MP4V fallback üretildi. Ayrıntı pipeline.log dosyasındadır.")
            elif self.temporary_path.exists():
                self.temporary_path.unlink()
        try:
            self._verify(candidate, self.metadata, self.frame_count)
            os.replace(candidate, destination)
        except Exception:
            candidate.unlink(missing_ok=True)
            raise
        return warnings
