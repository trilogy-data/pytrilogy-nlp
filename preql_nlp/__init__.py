from .monkeypatch import patch_promptimize

patch_promptimize()

from .main import build_query  # noqa: E402


__version__ = "0.0.10"

__all__ = ["build_query"]
