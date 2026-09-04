"""Local-only HTTP adapter for the surgical analysis core."""

from .app import create_app

__all__ = ["create_app"]
