/**
 * Event names are stable strings so integrations can switch on them by
 * swapping the import. Payloads are documented on {@link QavEventMap}.
 */
export enum QavEvent {
  /** WebRTC room joined. */
  CONNECTION_ESTABLISHED = "connection_established",
  /** Engine + face renderer are in the room and publishing. Payload: sessionId. */
  SESSION_READY = "session_ready",
  /** Payload: ConnectionClosedCode, details. */
  CONNECTION_CLOSED = "connection_closed",

  /** Payload: MediaStream carrying the avatar video track. */
  VIDEO_STREAM_STARTED = "video_stream_started",
  /** First avatar frame rendered into the video element. */
  VIDEO_PLAY_STARTED = "video_play_started",
  /** Payload: MediaStream carrying the avatar audio track. */
  AUDIO_STREAM_STARTED = "audio_stream_started",

  /** Payload: local MediaStream (microphone). */
  INPUT_AUDIO_STREAM_STARTED = "input_audio_stream_started",
  /** Payload: deviceId. */
  INPUT_AUDIO_DEVICE_CHANGED = "input_audio_device_changed",
  MIC_PERMISSION_PENDING = "mic_permission_pending",
  MIC_PERMISSION_GRANTED = "mic_permission_granted",
  /** Payload: Error. */
  MIC_PERMISSION_DENIED = "mic_permission_denied",

  /** Payload: Message[] — full history, emitted whenever a turn completes. */
  MESSAGE_HISTORY_UPDATED = "message_history_updated",
  /** Payload: MessageStreamEvent — partial transcript for the turn in progress. */
  MESSAGE_STREAM_EVENT_RECEIVED = "message_stream_event_received",

  /** Payload: PersonaState — listening | thinking | speaking | initializing. */
  PERSONA_STATE_CHANGED = "persona_state_changed",
  /** Payload: boolean — local user detected speaking. */
  USER_SPEAKING_CHANGED = "user_speaking_changed",

  /** Payload: string. */
  SERVER_WARNING = "server_warning",
}

export enum ConnectionClosedCode {
  NORMAL = "normal",
  SESSION_ENDED = "session_ended",
  SERVER_DISCONNECT = "server_disconnect",
  NETWORK_ERROR = "network_error",
  ENGINE_LEFT = "engine_left",
  UNKNOWN = "unknown",
}

export type PersonaState = "initializing" | "listening" | "thinking" | "speaking";

export interface Message {
  id: string;
  role: "user" | "persona";
  content: string;
  /** Whether the turn is complete. */
  final: boolean;
  createdAt: number;
}

export interface MessageStreamEvent {
  id: string;
  role: "user" | "persona";
  /** Accumulated text so far for this turn. */
  content: string;
  final: boolean;
}

export interface QavEventMap {
  [QavEvent.CONNECTION_ESTABLISHED]: () => void;
  [QavEvent.SESSION_READY]: (sessionId: string) => void;
  [QavEvent.CONNECTION_CLOSED]: (code: ConnectionClosedCode, details?: string) => void;
  [QavEvent.VIDEO_STREAM_STARTED]: (stream: MediaStream) => void;
  [QavEvent.VIDEO_PLAY_STARTED]: () => void;
  [QavEvent.AUDIO_STREAM_STARTED]: (stream: MediaStream) => void;
  [QavEvent.INPUT_AUDIO_STREAM_STARTED]: (stream: MediaStream) => void;
  [QavEvent.INPUT_AUDIO_DEVICE_CHANGED]: (deviceId: string) => void;
  [QavEvent.MIC_PERMISSION_PENDING]: () => void;
  [QavEvent.MIC_PERMISSION_GRANTED]: () => void;
  [QavEvent.MIC_PERMISSION_DENIED]: (error: Error) => void;
  [QavEvent.MESSAGE_HISTORY_UPDATED]: (messages: Message[]) => void;
  [QavEvent.MESSAGE_STREAM_EVENT_RECEIVED]: (event: MessageStreamEvent) => void;
  [QavEvent.PERSONA_STATE_CHANGED]: (state: PersonaState) => void;
  [QavEvent.USER_SPEAKING_CHANGED]: (speaking: boolean) => void;
  [QavEvent.SERVER_WARNING]: (message: string) => void;
}

type Listener = (...args: never[]) => void;

/** Minimal typed emitter; no Node `events` dependency in the browser bundle. */
export class TypedEmitter {
  private listeners = new Map<QavEvent, Set<Listener>>();

  addListener<E extends QavEvent>(event: E, listener: QavEventMap[E]): void {
    let set = this.listeners.get(event);
    if (!set) this.listeners.set(event, (set = new Set()));
    set.add(listener as Listener);
  }

  removeListener<E extends QavEvent>(event: E, listener: QavEventMap[E]): void {
    this.listeners.get(event)?.delete(listener as Listener);
  }

  removeAllListeners(): void {
    this.listeners.clear();
  }

  protected emit<E extends QavEvent>(event: E, ...args: Parameters<QavEventMap[E]>): void {
    const set = this.listeners.get(event);
    if (!set) return;
    for (const l of [...set]) {
      try {
        (l as (...a: unknown[]) => void)(...args);
      } catch (err) {
        console.error(`[qav] listener for ${event} threw`, err);
      }
    }
  }
}
