"""Avatar descriptions: what the API's avatar catalog entry means to a backend."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AvatarStyle:
    """Palette for the procedural (CPU placeholder) face."""

    skin: str = "#f1c9a5"
    hair: str = "#3b2a1f"
    eyes: str = "#3f6d8e"
    lips: str = "#c46a6a"
    shirt: str = "#2f4858"
    background: str = "#e9eef5"
    hair_style: str = "short"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> AvatarStyle:
        if not d:
            return cls()
        return cls(
            skin=d.get("skin", cls.skin),
            hair=d.get("hair", cls.hair),
            eyes=d.get("eyes", cls.eyes),
            lips=d.get("lips", cls.lips),
            shirt=d.get("shirt", cls.shirt),
            background=d.get("background", cls.background),
            hair_style=d.get("hairStyle", d.get("hair_style", cls.hair_style)),
        )


def avatar_root() -> str:
    """Where prepared avatar assets live (``QAV_AVATAR_DIR``, default ``./avatars``)."""
    return os.path.abspath(os.getenv("QAV_AVATAR_DIR", "avatars"))


@dataclass(frozen=True)
class AvatarSpec:
    id: str
    name: str
    renderer: str
    """Backend name: procedural | musetalk | liveavatar."""
    width: int
    height: int
    fps: float
    style: AvatarStyle = field(default_factory=AvatarStyle)
    assets: dict[str, Any] = field(default_factory=dict)
    """Backend-specific: ``{"avatarDir": "yongen"}`` for MuseTalk, ``{"image": "...", "prompt": "..."}`` for Live Avatar."""

    @classmethod
    def from_catalog(
        cls,
        avatar: dict[str, Any] | None,
        *,
        avatar_id: str = "",
        width: int,
        height: int,
        fps: float,
        default_renderer: str = "procedural",
    ) -> AvatarSpec:
        avatar = avatar or {}
        renderer = str(avatar.get("renderer") or default_renderer).lower()
        return cls(
            id=avatar.get("id") or avatar_id or "qav-nova",
            name=avatar.get("name", "QAV"),
            renderer=renderer,
            width=width,
            height=height,
            fps=fps,
            style=AvatarStyle.from_dict(avatar.get("style")),
            assets=dict(avatar.get("assets") or {}),
        )

    def asset_path(self, key: str, default: str | None = None) -> str | None:
        """Resolve an asset entry to an absolute path under the avatar root (absolute paths pass through)."""
        v = self.assets.get(key, default)
        if not v:
            return None
        v = os.path.expanduser(str(v))
        return v if os.path.isabs(v) else os.path.join(avatar_root(), v)
