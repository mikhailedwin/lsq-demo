"use client";

import { type RefObject, useEffect, useState } from "react";

/**
 * Samples the avatar video and returns three dominant colours.
 *
 * Liquid Glass only reads as glass when there is something behind it to
 * refract, so the page's ambient wash is derived from the call itself rather
 * than hard-coded — the same idea as the now-playing background in Music.
 *
 * WebRTC MediaStream frames do not taint a canvas, so this stays same-origin
 * safe; if a browser disagrees we simply stop sampling and keep the fallback.
 */
export function useAmbient(videoRef: RefObject<HTMLVideoElement | null>, active: boolean) {
  const [colors, setColors] = useState<[string, string, string] | null>(null);

  useEffect(() => {
    if (!active) {
      setColors(null);
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = 24;
    canvas.height = 24;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    let stop = false;
    let timer: number;

    const sample = () => {
      const v = videoRef.current;
      if (stop) return;
      if (v && v.videoWidth > 0) {
        try {
          ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
          const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
          // three bands — top, middle, bottom — so the wash has some structure
          const bands: [number, number][] = [
            [0, 8],
            [8, 16],
            [16, 24],
          ];
          const next = bands.map(([y0, y1], bi) => {
            let r = 0;
            let g = 0;
            let b = 0;
            let n = 0;
            for (let y = y0; y < y1; y++) {
              for (let x = 0; x < canvas.width; x++) {
                const i = (y * canvas.width + x) * 4;
                r += data[i];
                g += data[i + 1];
                b += data[i + 2];
                n++;
              }
            }
            return mixWithBase(r / n, g / n, b / n, BASE[bi]);
          }) as [string, string, string];
          setColors(next);
        } catch {
          stop = true; // tainted canvas or decode failure: keep the fallback wash
          return;
        }
      }
      timer = window.setTimeout(sample, 900);
    };

    sample();
    return () => {
      stop = true;
      window.clearTimeout(timer);
    };
  }, [videoRef, active]);

  return colors;
}

/** The wash's resting palette, matched to the CSS fallbacks. */
const BASE: [number, number, number][] = [
  [143, 182, 255],
  [201, 162, 255],
  [127, 220, 255],
];

/**
 * Tint the base palette toward the frame rather than using the frame directly.
 *
 * A camera frame is usually close to neutral, and you cannot boost chroma that
 * isn't there — saturating grey just yields grey, which leaves the glass with
 * nothing to refract. So the sampled colour steers the wash while the base
 * keeps it alive, weighted by how much colour the frame actually has.
 */
function mixWithBase(r: number, g: number, b: number, base: [number, number, number]): string {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const chroma = (max - min) / 255; // 0 for grey, ~1 for vivid
  const w = Math.min(0.65, 0.18 + chroma * 1.6); // how much the frame steers
  const mix = (c: number, bc: number) => Math.round(c * w + bc * (1 - w));
  return `rgb(${mix(r, base[0])} ${mix(g, base[1])} ${mix(b, base[2])})`;
}
