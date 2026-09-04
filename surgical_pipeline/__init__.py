"""Privacy-first surgical-video analytics package."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import AnalysisPipeline, AnalysisResult, analyze_video
    from .pose_pipeline import PosePilotResult, run_pose_pilot

__all__ = ["AnalysisPipeline", "AnalysisResult", "analyze_video", "PosePilotResult", "run_pose_pilot"]
try:
    __version__ = version("surgical-video-analytics")
    if not __version__:
        raise PackageNotFoundError("surgical-video-analytics")
except PackageNotFoundError:
    try:
        import tomllib

        __version__ = str(tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"])
    except (FileNotFoundError, KeyError, ModuleNotFoundError, ValueError):
        __version__ = "development"


def __getattr__(name: str):
    """Keep `doctor` usable before optional chart/UI dependencies are installed."""
    if name in {"AnalysisPipeline", "AnalysisResult", "analyze_video"}:
        from .pipeline import AnalysisPipeline, AnalysisResult, analyze_video

        return {"AnalysisPipeline": AnalysisPipeline, "AnalysisResult": AnalysisResult, "analyze_video": analyze_video}[name]
    if name in {"PosePilotResult", "run_pose_pilot"}:
        from .pose_pipeline import PosePilotResult, run_pose_pilot

        return {"PosePilotResult": PosePilotResult, "run_pose_pilot": run_pose_pilot}[name]
    raise AttributeError(name)
