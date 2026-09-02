"""Privacy-first surgical-video analytics package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import AnalysisPipeline, AnalysisResult

__all__ = ["AnalysisPipeline", "AnalysisResult"]
__version__ = "0.1.0"


def __getattr__(name: str):
    """Keep `doctor` usable before optional chart/UI dependencies are installed."""
    if name in {"AnalysisPipeline", "AnalysisResult"}:
        from .pipeline import AnalysisPipeline, AnalysisResult

        return {"AnalysisPipeline": AnalysisPipeline, "AnalysisResult": AnalysisResult}[name]
    raise AttributeError(name)
