"""Built-in CPU face: a stylised 2D character with audio-driven mouth, blinks and head motion.

Design goals: zero model downloads, a few ms/frame at 512², looks like a person
rather than a waveform, and every avatar gets a distinct look from its
`AvatarStyle` palette.

Rendering strategy: everything static (background, body, head, hair, soft
shading) is rasterised once at 2× into cached layers cropped to their bounding
boxes. Per frame we copy the background, composite the head + hair layers at a
small offset (head motion), draw the eyes/brows/mouth with opaque primitives,
and box-downsample for anti-aliasing. Note `ImageDraw` *replaces* pixels rather
than blending, so anything translucent lives in the cached layers only.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..avatars import AvatarStyle
from ..lipsync import FaceState
from .base import FaceRenderer

RGBA = tuple[int, int, int, int]


def _hex(color: str, alpha: int = 255) -> RGBA:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha


def _mix(a: RGBA, b: RGBA, t: float) -> RGBA:
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(4))  # type: ignore[return-value]


def _shade(c: RGBA, f: float) -> RGBA:
    """f < 1 darkens, f > 1 lightens (towards white)."""
    if f <= 1.0:
        return int(c[0] * f), int(c[1] * f), int(c[2] * f), c[3]
    return _mix(c, (255, 255, 255, c[3]), min(f - 1.0, 1.0))


class _Layer:
    """An RGBA image cropped to its content plus the offset it was cropped from."""

    def __init__(self, img: Image.Image) -> None:
        bbox = img.getbbox() or (0, 0, 1, 1)
        self.x, self.y = bbox[0], bbox[1]
        self.img = img.crop(bbox)


class ProceduralFaceRenderer(FaceRenderer):
    def __init__(self, *, width: int, height: int, style: AvatarStyle, seed: int | None = None) -> None:
        super().__init__(width=width, height=height, style=style, seed=seed)
        self.ss = 2 if max(width, height) <= 720 else 1
        self.W, self.H = width * self.ss, height * self.ss
        self._bg: Image.Image | None = None
        self._head: _Layer | None = None
        self._hair: _Layer | None = None
        self._g: dict[str, float] = {}

        s = style
        self.c_skin = _hex(s.skin)
        self.c_skin_dark = _shade(self.c_skin, 0.82)
        self.c_hair = _hex(s.hair)
        self.c_eyes = _hex(s.eyes)
        self.c_lips = _hex(s.lips)
        self.c_shirt = _hex(s.shirt)
        self.c_bg = _hex(s.background)
        self.c_line = _shade(self.c_skin, 0.55)
        self.c_blush = _mix(self.c_skin, _hex("#e0707a"), 0.4)

    # ------------------------------------------------------------------ setup
    def warmup(self) -> None:
        if self._bg is None:
            self._build_static()

    def _build_static(self) -> None:
        W, H = self.W, self.H
        u = min(W, H) / 512.0  # 1u == 1 output px at 512, scaled by supersample
        cx, cy = W / 2, H * 0.47
        head_rx, head_ry = 118 * u, 150 * u
        eye_y = cy - 18 * u
        eye_dx = 46 * u
        self._g = {"u": u, "cx": cx, "cy": cy, "head_rx": head_rx, "head_ry": head_ry, "eye_y": eye_y, "eye_dx": eye_dx}

        # --- background + body -------------------------------------------
        bg = Image.new("RGBA", (W, H), self.c_bg)
        d = ImageDraw.Draw(bg)
        top, bottom = _shade(self.c_bg, 1.06), _shade(self.c_bg, 0.9)
        for y in range(0, H, 4):
            d.rectangle([0, y, W, y + 4], fill=_mix(top, bottom, y / H))
        sh_y = cy + head_ry * 0.95
        d.rounded_rectangle([cx - 205 * u, sh_y + 40 * u, cx + 205 * u, H + 100 * u], radius=120 * u, fill=self.c_shirt)
        d.polygon(
            [(cx - 60 * u, sh_y + 40 * u), (cx + 60 * u, sh_y + 40 * u), (cx, sh_y + 110 * u)],
            fill=_shade(self.c_shirt, 0.85),
        )
        d.rounded_rectangle(
            [cx - 42 * u, cy + head_ry * 0.55, cx + 42 * u, sh_y + 60 * u], radius=30 * u, fill=self.c_skin_dark
        )
        if self.style.hair_style == "long":
            # hair falling behind the shoulders: two curtains, so the neck stays visible under the chin
            for sx in (-1, 1):
                x0 = cx + sx * head_rx * 0.55
                x1 = cx + sx * head_rx * 1.2
                d.rounded_rectangle(
                    [min(x0, x1), cy - head_ry * 0.9, max(x0, x1), cy + head_ry * 1.45],
                    radius=40 * u,
                    fill=self.c_hair,
                )
        self._bg = bg

        # --- head layer (moves as a unit) ----------------------------------
        head = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        hd = ImageDraw.Draw(head)
        for sx in (-1, 1):
            ex = cx + sx * head_rx * 0.98
            hd.ellipse([ex - 22 * u, cy - 20 * u, ex + 22 * u, cy + 34 * u], fill=self.c_skin_dark)
        hd.ellipse([cx - head_rx, cy - head_ry, cx + head_rx, cy + head_ry], fill=self.c_skin)

        # translucent shading goes through alpha_composite so it blends
        shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        sd = ImageDraw.Draw(shade)
        sd.ellipse(
            [cx - head_rx * 0.9, cy + head_ry * 0.45, cx + head_rx * 0.9, cy + head_ry * 1.02], fill=(0, 0, 0, 22)
        )
        for sx in (-1, 1):
            ex = cx + sx * eye_dx
            sd.ellipse([ex - 30 * u, eye_y - 20 * u, ex + 30 * u, eye_y + 22 * u], fill=(0, 0, 0, 16))
            bx = cx + sx * 66 * u
            sd.ellipse([bx - 30 * u, cy + 26 * u, bx + 30 * u, cy + 56 * u], fill=self.c_blush[:3] + (85,))
        shade = shade.filter(ImageFilter.GaussianBlur(10 * u))
        # keep the shading inside the face silhouette
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).ellipse([cx - head_rx, cy - head_ry, cx + head_rx, cy + head_ry], fill=255)
        shade.putalpha(Image.fromarray(np.minimum(np.asarray(shade.split()[3]), np.asarray(mask))))
        head = Image.alpha_composite(head, shade)

        hd = ImageDraw.Draw(head)
        nx, ny = cx, cy + 22 * u
        hd.line(
            [(nx - 2 * u, ny - 34 * u), (nx - 10 * u, ny + 8 * u), (nx + 6 * u, ny + 12 * u)],
            fill=self.c_line,
            width=int(3 * u),
            joint="curve",
        )
        self._head = _Layer(head)

        # --- hair front (over the face, moves with the head) --------------
        hair = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        hh = ImageDraw.Draw(hair)
        style = self.style.hair_style
        if style != "bald":
            # rounded cap: top half of an ellipse whose centre line sits above the brows
            hh.chord(
                [cx - head_rx * 1.05, cy - head_ry * 1.08, cx + head_rx * 1.05, cy - head_ry * 0.15],
                start=180,
                end=360,
                fill=self.c_hair,
            )
            # asymmetric fringe sweep, kept inside the silhouette so no corners poke out
            hh.polygon(
                [
                    (cx - head_rx * 0.97, cy - head_ry * 0.62),
                    (cx - head_rx * 0.5, cy - head_ry * 0.84),
                    (cx + head_rx * 0.25, cy - head_ry * 0.74),
                    (cx + head_rx * 0.97, cy - head_ry * 0.48),
                    (cx + head_rx * 0.97, cy - head_ry * 0.62),
                ],
                fill=self.c_hair,
            )
            if style == "short":
                for sx in (-1, 1):  # sideburns
                    hh.rounded_rectangle(
                        [cx + sx * head_rx * 0.95 - 12 * u, cy - head_ry * 0.6, cx + sx * head_rx * 0.95 + 12 * u, cy - head_ry * 0.05],
                        radius=8 * u,
                        fill=self.c_hair,
                    )
            if style == "bun":
                hh.ellipse([cx - 40 * u, cy - head_ry * 1.32, cx + 40 * u, cy - head_ry * 0.95], fill=self.c_hair)
            if style == "long":
                for sx in (-1, 1):  # strands framing the face, ending at cheek level
                    hh.rounded_rectangle(
                        [
                            cx + sx * head_rx * 0.92 - 26 * u,
                            cy - head_ry * 0.65,
                            cx + sx * head_rx * 0.92 + 26 * u,
                            cy + head_ry * 0.35,
                        ],
                        radius=26 * u,
                        fill=self.c_hair,
                    )
            hh.arc(
                [cx - head_rx * 0.8, cy - head_ry * 1.02, cx + head_rx * 0.4, cy - head_ry * 0.3],
                start=200,
                end=300,
                fill=_shade(self.c_hair, 1.25),
                width=int(6 * u),
            )
        self._hair = _Layer(hair)

    # ----------------------------------------------------------------- render
    def render(self, state: FaceState) -> np.ndarray:
        if self._bg is None:
            self._build_static()
        assert self._bg is not None and self._head is not None and self._hair is not None
        g = self._g
        u, cx, cy = g["u"], g["cx"], g["cy"]

        # head offset from yaw/pitch (+ a breathing bob); features shift a bit more for parallax
        dx = state.head_yaw * 9 * u + state.head_roll * 2 * u
        dy = state.head_pitch * 6 * u + math.sin(state.t * 1.4) * 1.2 * u
        fdx = dx * 1.6
        fdy = dy * 1.3

        frame = self._bg.copy()
        frame.alpha_composite(self._head.img, (max(0, self._head.x + int(dx)), max(0, self._head.y + int(dy))))
        d = ImageDraw.Draw(frame)

        # --- eyes ------------------------------------------------------------
        eye_y = g["eye_y"] + fdy
        eye_dx = g["eye_dx"]
        eye_rx, eye_ry = 24 * u, 15 * u
        open_amt = 1.0 - state.blink
        for sx in (-1, 1):
            ex = cx + sx * eye_dx + fdx
            ry = max(1.0, eye_ry * open_amt)
            d.ellipse([ex - eye_rx, eye_y - ry, ex + eye_rx, eye_y + ry], fill=(250, 250, 250, 255))
            if open_amt > 0.12:
                ix = ex + state.gaze_x * 9 * u
                iy = eye_y + state.gaze_y * 5 * u
                ir = 10.5 * u
                d.ellipse([ix - ir, iy - ir * open_amt, ix + ir, iy + ir * open_amt], fill=self.c_eyes)
                d.ellipse(
                    [ix - ir * 0.5, iy - ir * 0.5 * open_amt, ix + ir * 0.5, iy + ir * 0.5 * open_amt],
                    fill=(20, 18, 22, 255),
                )
                d.ellipse(
                    [ix - ir * 0.15 + 3 * u, iy - ir * 0.55 * open_amt, ix + ir * 0.35 + 3 * u, iy - ir * 0.05],
                    fill=(255, 255, 255, 255),
                )
            d.arc(
                [ex - eye_rx, eye_y - ry, ex + eye_rx, eye_y + ry],
                start=190,
                end=350,
                fill=self.c_line,
                width=int(3.2 * u),
            )
            if open_amt < 0.12:
                d.line([(ex - eye_rx, eye_y), (ex + eye_rx, eye_y)], fill=self.c_line, width=int(3 * u))

        # --- brows -----------------------------------------------------------
        brow_y = eye_y - 28 * u - state.brow_raise * 8 * u
        for sx in (-1, 1):
            bx = cx + sx * eye_dx + fdx
            inner = (bx - sx * 24 * u, brow_y + 4 * u)
            mid = (bx, brow_y - 4 * u - state.brow_raise * 2 * u)
            outer = (bx + sx * 26 * u, brow_y + 2 * u)
            d.line([inner, mid, outer], fill=_shade(self.c_hair, 0.9), width=int(6 * u), joint="curve")

        # --- mouth -----------------------------------------------------------
        mx = cx + fdx
        my = cy + 72 * u + fdy
        base_w = 44 * u
        w = base_w * (0.78 + 0.5 * state.mouth_width) * (1.0 - 0.18 * state.mouth_open)
        h = 3 * u + state.mouth_open * 40 * u
        if state.mouth_open < 0.06:
            d.line(
                [(mx - w, my - 2 * u), (mx, my + 3 * u), (mx + w, my - 2 * u)],
                fill=self.c_lips,
                width=int(7 * u),
                joint="curve",
            )
            d.line(
                [(mx - w * 0.9, my - 1 * u), (mx, my + 2 * u), (mx + w * 0.9, my - 1 * u)],
                fill=_shade(self.c_lips, 0.8),
                width=int(2 * u),
                joint="curve",
            )
        else:
            d.ellipse([mx - w - 5 * u, my - h - 5 * u, mx + w + 5 * u, my + h + 5 * u], fill=self.c_lips)
            d.ellipse([mx - w, my - h, mx + w, my + h], fill=(48, 18, 24, 255))
            if state.mouth_open > 0.18 or state.teeth > 0.3:
                th = min(h * 0.55, (6 + 8 * state.teeth) * u)
                d.rounded_rectangle(
                    [mx - w * 0.85, my - h * 0.92, mx + w * 0.85, my - h * 0.92 + th],
                    radius=3 * u,
                    fill=(245, 242, 236, 255),
                )
            if state.mouth_open > 0.45:
                d.ellipse([mx - w * 0.6, my + h * 0.15, mx + w * 0.6, my + h * 1.05], fill=(190, 80, 95, 255))
            d.arc(
                [mx - w - 5 * u, my - h - 5 * u, mx + w + 5 * u, my + h + 5 * u],
                start=200,
                end=340,
                fill=_shade(self.c_lips, 1.25),
                width=int(2 * u),
            )

        # --- hair over everything on the head ------------------------------
        frame.alpha_composite(self._hair.img, (max(0, self._hair.x + int(dx)), max(0, self._hair.y + int(dy))))

        if self.ss != 1:
            frame = frame.reduce(self.ss)
        return np.asarray(frame, dtype=np.uint8)
