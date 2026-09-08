export interface QavClientOptions {
  /**
   * Base URL of the QAV API. Defaults to the `api` claim baked into the
   * session token by the server, so you normally don't set this.
   */
  apiBaseUrl?: string;
  /** Don't capture or publish the microphone. */
  disableInputAudio?: boolean;
  /** Capture this input device instead of the default. */
  audioDeviceId?: string;
  /** Extra ICE servers (TURN) merged into the WebRTC config. */
  iceServers?: RTCIceServer[];
  /** Full RTCConfiguration passthrough, e.g. `{ iceTransportPolicy: "relay" }`. */
  rtcConfiguration?: RTCConfiguration;
  /** Free-form label stored on the session (shows up in `GET /v1/sessions`). */
  clientLabel?: string;
  /** Free-form JSON handed to the engine at session start. */
  clientMetadata?: Record<string, unknown>;
  /** Milliseconds to wait for the avatar's video track before giving up. Default 30 000. */
  avatarJoinTimeoutMs?: number;
}

export interface InputAudioState {
  isMuted: boolean;
  deviceId?: string;
}

/** Response of `POST /v1/engine/session`. */
export interface EngineSessionInfo {
  sessionId: string;
  livekitUrl: string;
  livekitToken: string;
  roomName: string;
  engineIdentity: string;
  avatarIdentity: string;
  persona: { name: string; avatarId: string };
}

export class QavError extends Error {
  constructor(
    message: string,
    public readonly code:
      | "invalid_token"
      | "session_start_failed"
      | "connect_failed"
      | "avatar_timeout"
      | "not_connected"
      | "mic_denied"
      | "rpc_failed",
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = "QavError";
  }
}
