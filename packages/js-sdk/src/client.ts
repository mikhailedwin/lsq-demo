import {
  ConnectionState,
  DisconnectReason,
  type LocalTrackPublication,
  type Participant,
  ParticipantKind,
  type RemoteParticipant,
  type RemoteTrack,
  type RemoteTrackPublication,
  Room,
  RoomEvent,
  Track,
  type TextStreamReader,
} from "livekit-client";
import {
  ConnectionClosedCode,
  type Message,
  type PersonaState,
  QavEvent,
  TypedEmitter,
} from "./events.js";
import { type EngineSessionInfo, type InputAudioState, QavClientOptions, QavError } from "./types.js";

// Wire constants shared with the engine (services/engine/qav_engine/protocol.py).
const TOPIC_TRANSCRIPTION = "lk.transcription";
const ATTR_SEGMENT_ID = "lk.segment_id";
const ATTR_AGENT_STATE = "lk.agent.state";
const RPC_TALK = "qav.talk";
const RPC_USER_MESSAGE = "qav.user_message";
const RPC_INTERRUPT = "qav.interrupt";

function decodeTokenClaims(token: string): { sid?: string; api?: string; exp?: number } {
  try {
    const payload = token.split(".")[1] ?? "";
    const json = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(json) as { sid?: string; api?: string; exp?: number };
  } catch {
    return {};
  }
}

/**
 * One QAV persona session: fetches LiveKit credentials with the session
 * token, joins the room, pipes the avatar's tracks into your `<video>`,
 * publishes the mic, and surfaces transcripts/state as events.
 */
export class QavClient extends TypedEmitter {
  private readonly options: QavClientOptions;
  private readonly apiBaseUrl: string;
  private room: Room | null = null;
  private info: EngineSessionInfo | null = null;
  private videoEl: HTMLVideoElement | null = null;
  private audioEl: HTMLAudioElement | null = null;
  private videoStream: MediaStream | null = null;
  private audioStream: MediaStream | null = null;
  private micPublication: LocalTrackPublication | null = null;
  private inputMuted = false;
  private messages: Message[] = [];
  private localSeq = 0;
  private personaState: PersonaState = "initializing";
  private closed = false;
  private sessionReadyFired = false;

  constructor(
    private readonly sessionToken: string,
    options: QavClientOptions = {},
  ) {
    super();
    this.options = options;
    const claims = decodeTokenClaims(sessionToken);
    const base = options.apiBaseUrl ?? claims.api;
    if (!base) {
      throw new QavError("session token has no API URL; pass options.apiBaseUrl", "invalid_token");
    }
    this.apiBaseUrl = base.replace(/\/+$/, "");
    if (claims.exp && claims.exp * 1000 < Date.now()) {
      throw new QavError("session token is expired", "invalid_token");
    }
  }

  // ---------------------------------------------------------------------------
  // Lifecycle
  // ---------------------------------------------------------------------------

  /**
   * Start the session and attach the avatar to a `<video>` element (by id or
   * reference). Audio plays through the same element unless you pass a
   * separate `<audio>` element.
   */
  async streamToVideoElement(
    video: string | HTMLVideoElement,
    audio?: string | HTMLAudioElement,
  ): Promise<void> {
    const videoEl = typeof video === "string" ? document.getElementById(video) : video;
    if (!(videoEl instanceof HTMLVideoElement)) {
      throw new QavError(`video element not found: ${String(video)}`, "connect_failed");
    }
    this.videoEl = videoEl;
    if (audio) {
      const audioEl = typeof audio === "string" ? document.getElementById(audio) : audio;
      if (audioEl instanceof HTMLAudioElement) this.audioEl = audioEl;
    }
    videoEl.autoplay = true;
    videoEl.playsInline = true;
    await this.stream();
  }

  /**
   * Start the session and return the avatar's media streams without touching
   * the DOM. Resolves once the face renderer is publishing video.
   */
  async stream(): Promise<MediaStream[]> {
    if (this.room) throw new QavError("already streaming", "connect_failed");
    this.closed = false;

    const info = await this.startEngineSession();
    this.info = info;

    // One avatar track per session: always take the full-resolution layer rather than
    // letting adaptive streaming downgrade it to the <video> element's CSS size.
    const room = new Room({ adaptiveStream: false, dynacast: false });
    this.room = room;
    this.bindRoomEvents(room, info);

    try {
      await room.connect(info.livekitUrl, info.livekitToken, {
        autoSubscribe: true,
        rtcConfig: this.buildRtcConfig(),
      });
    } catch (err) {
      this.room = null;
      throw new QavError(`could not join session room: ${String(err)}`, "connect_failed", err);
    }
    this.emit(QavEvent.CONNECTION_ESTABLISHED);

    // Tracks that were already published before we joined won't re-fire TrackSubscribed.
    for (const p of room.remoteParticipants.values()) this.adoptExistingTracks(p, info);

    if (!this.options.disableInputAudio) {
      await this.publishMicrophone(room);
    }

    await this.waitForAvatar(info);
    return [this.videoStream, this.audioStream].filter((s): s is MediaStream => s !== null);
  }

  /** Leave the room and end the session server-side. Safe to call twice. */
  async stopStreaming(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const room = this.room;
    this.room = null;
    if (this.videoEl) this.videoEl.srcObject = null;
    if (this.audioEl) this.audioEl.srcObject = null;
    this.videoStream = null;
    this.audioStream = null;
    try {
      await room?.disconnect(true);
    } finally {
      // Fire-and-forget: the room's empty timeout cleans up even if this fails.
      void fetch(`${this.apiBaseUrl}/v1/engine/session/stop`, {
        method: "POST",
        headers: { authorization: `Bearer ${this.sessionToken}` },
        keepalive: true,
      }).catch(() => undefined);
      this.emit(QavEvent.CONNECTION_CLOSED, ConnectionClosedCode.NORMAL);
    }
  }

  getActiveSessionId(): string | null {
    return this.info?.sessionId ?? null;
  }

  getPersonaState(): PersonaState {
    return this.personaState;
  }

  getMessageHistory(): Message[] {
    return [...this.messages];
  }

  // ---------------------------------------------------------------------------
  // Commands (RPC to the engine participant)
  // ---------------------------------------------------------------------------

  /** Make the persona speak this text verbatim. */
  async talk(text: string): Promise<void> {
    await this.rpc(RPC_TALK, { text });
  }

  /** Inject a user message as if it had been spoken; the persona replies. */
  async sendUserMessage(text: string): Promise<void> {
    await this.rpc(RPC_USER_MESSAGE, { text });
    // Show it straight away. A typed turn only reaches the transcript as a text
    // stream when a real LLM is driving the reply; with an echo/mock brain the
    // engine never creates a user turn at all, so without this the message
    // vanishes and typing looks broken. Any stream that does arrive for the
    // same text is folded into this entry rather than duplicating it.
    this.addLocalUserMessage(text);
  }

  private addLocalUserMessage(text: string): void {
    this.messages.push({
      id: `local-${++this.localSeq}`,
      role: "user",
      content: text,
      final: true,
      createdAt: Date.now(),
    });
    this.emit(QavEvent.MESSAGE_HISTORY_UPDATED, this.getMessageHistory());
  }

  /** Cut the persona off mid-sentence. */
  async interruptPersona(): Promise<void> {
    await this.rpc(RPC_INTERRUPT, {});
  }

  // ---------------------------------------------------------------------------
  // Microphone
  // ---------------------------------------------------------------------------

  getInputAudioState(): InputAudioState {
    return { isMuted: this.inputMuted, deviceId: this.options.audioDeviceId };
  }

  async muteInputAudio(): Promise<InputAudioState> {
    this.inputMuted = true;
    await this.micPublication?.mute();
    return this.getInputAudioState();
  }

  async unmuteInputAudio(): Promise<InputAudioState> {
    this.inputMuted = false;
    if (this.room && !this.micPublication) {
      await this.publishMicrophone(this.room);
    } else {
      await this.micPublication?.unmute();
    }
    return this.getInputAudioState();
  }

  async setInputAudioDevice(deviceId: string): Promise<void> {
    if (!this.room) throw new QavError("not connected", "not_connected");
    await this.room.switchActiveDevice("audioinput", deviceId);
    this.options.audioDeviceId = deviceId;
    this.emit(QavEvent.INPUT_AUDIO_DEVICE_CHANGED, deviceId);
  }

  // ---------------------------------------------------------------------------
  // Internals
  // ---------------------------------------------------------------------------

  /** The engine normally joins as `info.engineIdentity`; fall back to "the agent that isn't the face". */
  private isEngine(p: Participant): boolean {
    const info = this.info;
    if (!info) return false;
    if (p.identity === info.engineIdentity) return true;
    return p.kind === ParticipantKind.AGENT && p.identity !== info.avatarIdentity;
  }

  private isPersona(identity: string): boolean {
    const info = this.info;
    if (!info) return false;
    if (identity === info.engineIdentity || identity === info.avatarIdentity) return true;
    const p = this.room?.remoteParticipants.get(identity);
    return p ? this.isEngine(p) : false;
  }

  private engineIdentity(): string {
    const room = this.room;
    const info = this.info;
    if (!room || !info) return info?.engineIdentity ?? "";
    if (room.remoteParticipants.has(info.engineIdentity)) return info.engineIdentity;
    for (const p of room.remoteParticipants.values()) if (this.isEngine(p)) return p.identity;
    return info.engineIdentity;
  }

  private async startEngineSession(): Promise<EngineSessionInfo> {
    let res: Response;
    try {
      res = await fetch(`${this.apiBaseUrl}/v1/engine/session`, {
        method: "POST",
        headers: { authorization: `Bearer ${this.sessionToken}`, "content-type": "application/json" },
        body: JSON.stringify({
          clientLabel: this.options.clientLabel,
          clientMetadata: this.options.clientMetadata,
        }),
      });
    } catch (err) {
      throw new QavError(`could not reach QAV API at ${this.apiBaseUrl}`, "session_start_failed", err);
    }
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new QavError(`engine session failed (${res.status}): ${body}`, "session_start_failed");
    }
    return (await res.json()) as EngineSessionInfo;
  }

  private buildRtcConfig(): RTCConfiguration | undefined {
    const { iceServers, rtcConfiguration } = this.options;
    if (!iceServers && !rtcConfiguration) return undefined;
    const merged = [...(iceServers ?? []), ...(rtcConfiguration?.iceServers ?? [])];
    return { ...rtcConfiguration, ...(merged.length ? { iceServers: merged } : {}) };
  }

  private bindRoomEvents(room: Room, info: EngineSessionInfo): void {
    room.on(RoomEvent.TrackSubscribed, (track, pub, participant) => {
      this.onTrack(track, pub, participant, info);
    });

    room.on(RoomEvent.ParticipantAttributesChanged, (changed, participant) => {
      if (participant.identity === info.engineIdentity && ATTR_AGENT_STATE in changed) {
        this.setPersonaState(changed[ATTR_AGENT_STATE]);
      }
    });

    room.on(RoomEvent.ParticipantConnected, (p) => {
      if (p.identity === info.engineIdentity) this.setPersonaState(p.attributes[ATTR_AGENT_STATE]);
    });

    room.on(RoomEvent.ParticipantDisconnected, (p) => {
      if (p.identity === info.engineIdentity && !this.closed) {
        this.emit(QavEvent.SERVER_WARNING, "engine left the session");
        void this.teardown(ConnectionClosedCode.ENGINE_LEFT);
      }
    });

    room.on(RoomEvent.ActiveSpeakersChanged, (speakers: Participant[]) => {
      const me = room.localParticipant.identity;
      this.emit(
        QavEvent.USER_SPEAKING_CHANGED,
        speakers.some((s) => s.identity === me),
      );
    });

    room.on(RoomEvent.Disconnected, (reason) => {
      if (this.closed) return;
      void this.teardown(mapDisconnectReason(reason));
    });

    room.on(RoomEvent.MediaDevicesError, (err) => {
      this.emit(QavEvent.SERVER_WARNING, `media device error: ${err.message}`);
    });

    room.registerTextStreamHandler(TOPIC_TRANSCRIPTION, (reader, participantInfo) => {
      void this.consumeTranscription(reader, participantInfo.identity, room, info);
    });
  }

  private adoptExistingTracks(p: RemoteParticipant, info: EngineSessionInfo): void {
    for (const pub of p.trackPublications.values()) {
      if (pub.track) this.onTrack(pub.track, pub, p, info);
    }
    if (p.identity === info.engineIdentity) this.setPersonaState(p.attributes[ATTR_AGENT_STATE]);
  }

  private onTrack(
    track: RemoteTrack,
    _pub: RemoteTrackPublication,
    participant: RemoteParticipant,
    info: EngineSessionInfo,
  ): void {
    // The renderer publishes on behalf of the engine; accept either identity.
    if (participant.identity !== info.avatarIdentity && participant.identity !== info.engineIdentity) return;

    if (track.kind === Track.Kind.Video) {
      this.videoStream = new MediaStream([track.mediaStreamTrack]);
      this.emit(QavEvent.VIDEO_STREAM_STARTED, this.videoStream);
      if (this.videoEl) {
        track.attach(this.videoEl);
        const onPlaying = () => {
          this.videoEl?.removeEventListener("playing", onPlaying);
          this.emit(QavEvent.VIDEO_PLAY_STARTED);
        };
        this.videoEl.addEventListener("playing", onPlaying);
      }
    } else if (track.kind === Track.Kind.Audio) {
      this.audioStream = new MediaStream([track.mediaStreamTrack]);
      this.emit(QavEvent.AUDIO_STREAM_STARTED, this.audioStream);
      if (this.audioEl) track.attach(this.audioEl);
      else if (this.videoEl) track.attach(this.videoEl);
      else track.attach(); // creates a detached <audio> so the persona is audible
    }
    this.maybeSessionReady();
  }

  private maybeSessionReady(): void {
    if (this.sessionReadyFired || !this.videoStream || !this.audioStream || !this.info) return;
    this.sessionReadyFired = true;
    this.emit(QavEvent.SESSION_READY, this.info.sessionId);
  }

  private async publishMicrophone(room: Room): Promise<void> {
    this.emit(QavEvent.MIC_PERMISSION_PENDING);
    try {
      const pub = await room.localParticipant.setMicrophoneEnabled(true, {
        deviceId: this.options.audioDeviceId,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      });
      this.micPublication = pub ?? null;
      this.emit(QavEvent.MIC_PERMISSION_GRANTED);
      const msTrack = pub?.track?.mediaStreamTrack;
      if (msTrack) this.emit(QavEvent.INPUT_AUDIO_STREAM_STARTED, new MediaStream([msTrack]));
      if (this.inputMuted) await pub?.mute();
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err));
      this.emit(QavEvent.MIC_PERMISSION_DENIED, e);
      // Keep going — the session is still usable through talk()/sendUserMessage().
      this.emit(QavEvent.SERVER_WARNING, `microphone unavailable: ${e.message}`);
    }
  }

  private async waitForAvatar(info: EngineSessionInfo): Promise<void> {
    if (this.videoStream) return;
    const timeout = this.options.avatarJoinTimeoutMs ?? 30_000;
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.removeListener(QavEvent.VIDEO_STREAM_STARTED, onVideo);
        reject(new QavError(`avatar did not publish video within ${timeout}ms`, "avatar_timeout"));
      }, timeout);
      const onVideo = () => {
        clearTimeout(timer);
        this.removeListener(QavEvent.VIDEO_STREAM_STARTED, onVideo);
        resolve();
      };
      this.addListener(QavEvent.VIDEO_STREAM_STARTED, onVideo);
      const onClosed = () => {
        clearTimeout(timer);
        this.removeListener(QavEvent.CONNECTION_CLOSED, onClosed);
        reject(new QavError("connection closed before the avatar joined", "connect_failed"));
      };
      this.addListener(QavEvent.CONNECTION_CLOSED, onClosed);
    }).catch(async (err) => {
      await this.stopStreaming();
      throw err;
    });
  }

  private async consumeTranscription(
    reader: TextStreamReader,
    senderIdentity: string,
    room: Room,
    info: EngineSessionInfo,
  ): Promise<void> {
    const attrs = reader.info.attributes ?? {};
    const id = attrs[ATTR_SEGMENT_ID] ?? reader.info.id;
    // The engine re-publishes user speech under the user's own identity.
    const role: Message["role"] =
      senderIdentity === room.localParticipant.identity
        ? "user"
        : senderIdentity === info.engineIdentity || senderIdentity === info.avatarIdentity
          ? "persona"
          : "user";

    let msg = this.messages.find((m) => m.id === id);
    if (!msg) {
      msg = { id, role, content: "", final: false, createdAt: Date.now() };
      this.messages.push(msg);
    }
    // Adopt a locally-echoed turn rather than showing the same words twice.
    const echoed = role === "user" ? this.messages.find((m) => m.id.startsWith("local-")) : undefined;
    let text = "";
    try {
      for await (const chunk of reader) {
        text += chunk;
        msg.content = text;
        this.emit(QavEvent.MESSAGE_STREAM_EVENT_RECEIVED, { id, role, content: text, final: false });
      }
    } catch {
      // Stream aborted (interruption / disconnect): keep what we have.
    }
    msg.content = text;
    // Stream closed => the engine is done with this segment, whatever the running attribute said.
    msg.final = true;
    // Drop empty turns (e.g. a persona interrupted before its first word).
    if (!msg.content.trim()) {
      this.messages = this.messages.filter((m) => m.id !== id);
      return;
    }
    if (echoed && echoed.content.trim() === text.trim()) {
      this.messages = this.messages.filter((m) => m.id !== echoed.id);
    }
    this.emit(QavEvent.MESSAGE_STREAM_EVENT_RECEIVED, { id, role, content: text, final: true });
    this.emit(QavEvent.MESSAGE_HISTORY_UPDATED, this.getMessageHistory());
  }

  private setPersonaState(raw: string | undefined): void {
    const next = normalizeState(raw);
    if (next === this.personaState) return;
    this.personaState = next;
    this.emit(QavEvent.PERSONA_STATE_CHANGED, next);
  }

  private async rpc(method: string, payload: Record<string, unknown>): Promise<string> {
    const room = this.room;
    const info = this.info;
    if (!room || !info || room.state !== ConnectionState.Connected) {
      throw new QavError("not connected", "not_connected");
    }
    try {
      return await room.localParticipant.performRpc({
        destinationIdentity: info.engineIdentity,
        method,
        payload: JSON.stringify(payload),
        responseTimeout: 10_000,
      });
    } catch (err) {
      throw new QavError(`${method} failed: ${String(err)}`, "rpc_failed", err);
    }
  }

  private async teardown(code: ConnectionClosedCode): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    const room = this.room;
    this.room = null;
    if (this.videoEl) this.videoEl.srcObject = null;
    await room?.disconnect(true).catch(() => undefined);
    this.emit(QavEvent.CONNECTION_CLOSED, code);
  }
}

function normalizeState(raw: string | undefined): PersonaState {
  switch (raw) {
    case "listening":
    case "thinking":
    case "speaking":
      return raw;
    case "idle":
      return "listening";
    default:
      return "initializing";
  }
}

function mapDisconnectReason(reason?: DisconnectReason): ConnectionClosedCode {
  switch (reason) {
    case DisconnectReason.CLIENT_INITIATED:
      return ConnectionClosedCode.NORMAL;
    case DisconnectReason.ROOM_DELETED:
    case DisconnectReason.ROOM_CLOSED:
      return ConnectionClosedCode.SESSION_ENDED;
    case DisconnectReason.SERVER_SHUTDOWN:
    case DisconnectReason.PARTICIPANT_REMOVED:
    case DisconnectReason.DUPLICATE_IDENTITY:
      return ConnectionClosedCode.SERVER_DISCONNECT;
    case DisconnectReason.SIGNAL_CLOSE:
    case DisconnectReason.CONNECTION_TIMEOUT:
    case DisconnectReason.JOIN_FAILURE:
    case DisconnectReason.STATE_MISMATCH:
    case DisconnectReason.MIGRATION:
      return ConnectionClosedCode.NETWORK_ERROR;
    default:
      return ConnectionClosedCode.UNKNOWN;
  }
}
