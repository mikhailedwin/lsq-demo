from __future__ import annotations

import importlib
from collections.abc import Callable

from ..avatars import AvatarSpec
from .base import FaceBackend

# name -> "module:Class". GPU backends import torch lazily so the CPU path stays light.
_REGISTRY: dict[str, str | Callable[[AvatarSpec], FaceBackend]] = {
    "procedural": "qav_face.backends.procedural:ProceduralBackend",
    "musetalk": "qav_face.backends.musetalk:MuseTalkBackend",
    "liveavatar": "qav_face.backends.liveavatar:LiveAvatarBackend",
}


def register_backend(name: str, factory: str | Callable[[AvatarSpec], FaceBackend]) -> None:
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def create_backend(spec: AvatarSpec) -> FaceBackend:
    try:
        target = _REGISTRY[spec.renderer]
    except KeyError as e:
        raise ValueError(f"unknown face renderer {spec.renderer!r}; available: {available_backends()}") from e
    if callable(target):
        return target(spec)
    mod_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(spec)


__all__ = ["FaceBackend", "available_backends", "create_backend", "register_backend"]
