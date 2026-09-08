"""qav-face command line.

    qav-face selftest --renderer procedural --out /tmp/qav-selftest
    qav-face selftest --renderer musetalk --avatar yongen --out /tmp/qav-selftest
    qav-face prepare --renderer musetalk --avatar yongen --video data/video/yongen.mp4
    qav-face demo-avatars           # prepare the bundled demo avatars for every installed backend
    qav-face start                  # run the worker (same as python -m qav_face.worker start)

`selftest` is the no-GPU-in-CI escape hatch: it drives a backend with a WAV (or
synthetic speech) through the exact frame loop the worker uses and writes
PNG frames, a contact sheet and — if imageio is installed — an MP4 with audio,
so you can eyeball lip-sync on the GPU box before wiring up a session.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import wave

import numpy as np

from .avatars import AvatarSpec, AvatarStyle, avatar_root


def _read_wav(path: str, target_sr: int) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise SystemExit("selftest needs 16-bit PCM WAV")
    pcm = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch)[:, 0]
    if sr != target_sr:
        x = np.arange(len(pcm)) / sr
        xi = np.arange(int(len(pcm) * target_sr / sr)) / target_sr
        pcm = np.interp(xi, x, pcm.astype(np.float32)).astype(np.int16)
    return pcm


def cmd_selftest(a: argparse.Namespace) -> int:
    from livekit import rtc
    from livekit.agents.voice.avatar import AudioSegmentEnd, AvatarOptions

    from .backends import create_backend
    from .protocol import AUDIO_SAMPLE_RATE
    from .stream import BackendVideoGenerator
    from .testing import synth_speech

    spec = AvatarSpec(
        id=a.avatar or "selftest",
        name=a.avatar or "selftest",
        renderer=a.renderer,
        width=a.width,
        height=a.height,
        fps=a.fps,
        style=AvatarStyle(hair_style="long"),
        assets={"avatarDir": a.avatar} if a.avatar else {},
    )
    if a.image:
        spec = AvatarSpec(**{**spec.__dict__, "assets": {"image": a.image, "prompt": a.prompt or ""}})
    backend = create_backend(spec)
    options = AvatarOptions(
        video_width=a.width, video_height=a.height, video_fps=a.fps, audio_sample_rate=AUDIO_SAMPLE_RATE, audio_channels=1
    )
    gen = BackendVideoGenerator(options, backend)

    pcm = _read_wav(a.wav, AUDIO_SAMPLE_RATE) if a.wav else synth_speech(a.text, sample_rate=AUDIO_SAMPLE_RATE)
    os.makedirs(a.out, exist_ok=True)

    async def run() -> tuple[list[np.ndarray], np.ndarray]:
        # feed audio in 20 ms frames, like the data stream does
        step = AUDIO_SAMPLE_RATE // 50
        for i in range(0, len(pcm), step):
            chunk = pcm[i : i + step]
            await gen.push_audio(
                rtc.AudioFrame(data=chunk.tobytes(), sample_rate=AUDIO_SAMPLE_RATE, num_channels=1, samples_per_channel=len(chunk))
            )
        await gen.push_audio(AudioSegmentEnd())

        frames: list[np.ndarray] = []
        audio_out: list[np.ndarray] = []
        idle_after = int(a.fps * a.idle_seconds)
        idle_count = 0
        async for item in gen:
            if isinstance(item, rtc.VideoFrame):
                buf = np.frombuffer(item.data, dtype=np.uint8).reshape(item.height, item.width, 4)
                frames.append(buf.copy())
                if segment_done[0]:
                    idle_count += 1
                    if idle_count >= idle_after:
                        break
            elif isinstance(item, rtc.AudioFrame):
                audio_out.append(np.frombuffer(item.data, dtype=np.int16).copy())
            elif isinstance(item, AudioSegmentEnd):
                segment_done[0] = True
                if idle_after == 0:
                    break
        return frames, np.concatenate(audio_out) if audio_out else np.zeros(0, dtype=np.int16)

    segment_done = [False]
    frames, audio_out = asyncio.run(run())
    backend.close()

    from PIL import Image

    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(a.out, f"{i:05d}.png"))
    # contact sheet of 12 evenly spaced frames
    picks = [frames[int(i * (len(frames) - 1) / 11)] for i in range(12)] if len(frames) >= 12 else frames
    tile = 256
    sheet = Image.new("RGBA", (4 * tile, ((len(picks) + 3) // 4) * tile))
    for i, f in enumerate(picks):
        sheet.paste(Image.fromarray(f).resize((tile, tile)), ((i % 4) * tile, (i // 4) * tile))
    sheet.save(os.path.join(a.out, "contact-sheet.png"))

    mp4 = os.path.join(a.out, "selftest.mp4")
    try:
        import imageio.v3 as iio  # type: ignore

        wav_path = os.path.join(a.out, "audio.wav")
        with wave.open(wav_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(AUDIO_SAMPLE_RATE)
            w.writeframes(audio_out.tobytes())
        iio.imwrite(mp4.replace(".mp4", "-silent.mp4"), [f[..., :3] for f in frames], fps=a.fps, codec="libx264")
        import subprocess

        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp4.replace(".mp4", "-silent.mp4"), "-i", wav_path, "-c:v", "copy", "-c:a", "aac", "-shortest", mp4],
            check=False,
        )
    except Exception as e:  # noqa: BLE001
        print(f"(no MP4: {e}; PNG frames + contact sheet written)")

    print(f"{len(frames)} frames, backend {type(backend).__name__}, avg {gen.render_ms:.1f} ms/frame → {a.out}")
    return 0


def cmd_prepare(a: argparse.Namespace) -> int:
    if a.renderer == "musetalk":
        from .backends.musetalk import prepare_avatar

        out = prepare_avatar(avatar_id=a.avatar, video_path=a.video, bbox_shift=a.bbox_shift)
        print(f"prepared MuseTalk avatar {a.avatar!r} → {out}")
        return 0
    if a.renderer == "liveavatar":
        from .backends.liveavatar import prepare_avatar

        out = prepare_avatar(avatar_id=a.avatar, image_path=a.video, prompt=a.prompt or "")
        print(f"prepared Live Avatar {a.avatar!r} → {out}")
        return 0
    print("nothing to prepare for the procedural renderer")
    return 0


def cmd_demo_avatars(a: argparse.Namespace) -> int:
    """Prepare the demo avatars shipped with the model repos (MIT / Apache-2.0 sample assets)."""
    from .backends import available_backends

    done = 0
    mt_root = os.getenv("MUSETALK_ROOT")
    if "musetalk" in available_backends() and mt_root and (a.only in (None, "musetalk")):
        from .backends.musetalk import prepare_avatar

        for name in ("yongen", "sun"):
            src = os.path.join(mt_root, "data", "video", f"{name}.mp4")
            if os.path.exists(src):
                print(f"[musetalk] preparing {name} from {src}")
                prepare_avatar(avatar_id=name, video_path=src, bbox_shift=0)
                done += 1
    la_root = os.getenv("LIVEAVATAR_ROOT")
    if "liveavatar" in available_backends() and la_root and (a.only in (None, "liveavatar")):
        from .backends.liveavatar import DEMO_PROMPTS, prepare_avatar

        for name, prompt in DEMO_PROMPTS.items():
            src = os.path.join(la_root, "examples", f"{name}.jpg")
            if os.path.exists(src):
                print(f"[liveavatar] preparing {name} from {src}")
                prepare_avatar(avatar_id=name, image_path=src, prompt=prompt)
                done += 1
    print(f"{done} demo avatar(s) prepared under {avatar_root()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qav-face", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="render a WAV / synthetic speech through a backend, offline")
    s.add_argument("--renderer", default=os.getenv("QAV_RENDERER", "procedural"))
    s.add_argument("--avatar", help="prepared avatar id (MuseTalk) — see `prepare`")
    s.add_argument("--image", help="reference image (Live Avatar)")
    s.add_argument("--prompt", help="scene prompt (Live Avatar)")
    s.add_argument("--wav", help="16-bit PCM WAV to lip-sync; default: synthetic speech")
    s.add_argument("--text", default="Hello! I am your QAV avatar. This is a self test of the face renderer.")
    s.add_argument("--width", type=int, default=int(os.getenv("QAV_VIDEO_WIDTH", "512")))
    s.add_argument("--height", type=int, default=int(os.getenv("QAV_VIDEO_HEIGHT", "512")))
    s.add_argument("--fps", type=float, default=float(os.getenv("QAV_VIDEO_FPS", "25")))
    s.add_argument("--idle-seconds", type=float, default=1.0, help="idle frames to render after speech")
    s.add_argument("--out", default="qav-selftest")
    s.set_defaults(fn=cmd_selftest)

    pr = sub.add_parser("prepare", help="preprocess an avatar source (video clip / image) for a backend")
    pr.add_argument("--renderer", required=True, choices=["musetalk", "liveavatar", "procedural"])
    pr.add_argument("--avatar", required=True, help="avatar id (directory name under QAV_AVATAR_DIR)")
    pr.add_argument("--video", required=True, help="source video (MuseTalk) or image (Live Avatar)")
    pr.add_argument("--prompt", help="scene prompt (Live Avatar)")
    pr.add_argument("--bbox-shift", type=int, default=0)
    pr.set_defaults(fn=cmd_prepare)

    d = sub.add_parser("demo-avatars", help="prepare the demo avatars bundled with the model repos")
    d.add_argument("--only", choices=["musetalk", "liveavatar"])
    d.set_defaults(fn=cmd_demo_avatars)

    st = sub.add_parser("start", help="run the face worker")
    st.set_defaults(fn=None)

    a = p.parse_args(argv)
    if a.cmd == "start":
        from .worker import main as worker_main

        sys.argv = [sys.argv[0], "start"]
        worker_main()
        return 0
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
