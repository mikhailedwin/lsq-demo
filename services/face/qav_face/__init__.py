"""QAV face renderer.

Everything that turns speech audio into a talking face lives here, behind
two interfaces:

* :class:`qav_face.backends.FaceBackend` — a face model (procedural CPU
  placeholder, MuseTalk, Live Avatar, ...).
* :class:`qav_face.stream.BackendVideoGenerator` — feeds any backend from a
  LiveKit audio stream and emits AV-synced frames for ``AvatarRunner``.

The engine runs a backend in-process for the CPU placeholder; GPU backends
run in the ``qav-face`` worker (:mod:`qav_face.worker`) that joins the room
as a second participant.
"""

__version__ = "0.1.0"
