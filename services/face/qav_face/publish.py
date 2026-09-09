"""AvatarRunner with explicit video publish settings."""

from __future__ import annotations

from livekit import rtc
from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner


def video_bitrate_for(width: int, height: int, fps: float) -> int:
    """Bits/s that keep a talking head sharp. libwebrtc's default table for
    small frames (~225 kbps at 384²) makes the encoder *downscale* the picture;
    a face is mostly static so this is still cheap on the wire."""
    pixels = width * height
    bps = int(pixels * fps * 0.22)  # ≈ 0.22 bits per pixel per frame
    return max(600_000, min(bps, 6_000_000))


def video_publish_options(options: AvatarOptions) -> rtc.TrackPublishOptions:
    return rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        simulcast=False,  # one avatar per session; always serve the full layer
        video_encoding=rtc.VideoEncoding(
            max_bitrate=video_bitrate_for(options.video_width, options.video_height, options.video_fps),
            max_framerate=int(options.video_fps),
        ),
        degradation_preference=rtc.DegradationPreference.MAINTAIN_RESOLUTION,
    )


class QavAvatarRunner(AvatarRunner):
    """The base class publishes with LiveKit's defaults (auto bitrate, simulcast on,
    balanced degradation). Overriding the private ``_publish_track`` is the only hook
    it offers (livekit-agents 1.8); revisit if a public option appears."""

    def __init__(self, room: rtc.Room, *, video_options: rtc.TrackPublishOptions | None = None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(room, **kwargs)
        self._qav_video_options = video_options or video_publish_options(self._options)

    async def _publish_track(self) -> None:
        async with self._lock:
            await self._room_connected_fut

            audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self._audio_source)
            audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            self._audio_publication = await self._room.local_participant.publish_track(audio_track, audio_options)
            await self._audio_publication.wait_for_subscription()

            video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self._video_source)
            self._video_publication = await self._room.local_participant.publish_track(
                video_track, self._qav_video_options
            )
