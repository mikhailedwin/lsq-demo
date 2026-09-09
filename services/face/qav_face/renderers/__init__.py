from __future__ import annotations

from ..avatars import AvatarStyle
from .base import FaceRenderer
from .procedural import ProceduralFaceRenderer

_REGISTRY: dict[str, type[FaceRenderer]] = {
    "procedural": ProceduralFaceRenderer,
}


def register_renderer(name: str, cls: type[FaceRenderer]) -> None:
    """Plug in a new face model (e.g. a GPU neural renderer) under `QAV_RENDERER=<name>`."""
    _REGISTRY[name] = cls


def get_renderer(name: str, *, width: int, height: int, style: AvatarStyle, seed: int | None = None) -> FaceRenderer:
    try:
        cls = _REGISTRY[name]
    except KeyError as e:
        raise ValueError(f"unknown renderer {name!r}; available: {sorted(_REGISTRY)}") from e
    return cls(width=width, height=height, style=style, seed=seed)


__all__ = ["FaceRenderer", "ProceduralFaceRenderer", "get_renderer", "register_renderer"]
