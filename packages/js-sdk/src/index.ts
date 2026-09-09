import { QavClient } from "./client.js";
import type { QavClientOptions } from "./types.js";

/**
 * Create a client for one persona session.
 *
 * ```ts
 * const { sessionToken } = await fetch("/api/session-token", { method: "POST" }).then(r => r.json());
 * const qav = createClient(sessionToken);
 * qav.addListener(QavEvent.MESSAGE_HISTORY_UPDATED, (m) => render(m));
 * await qav.streamToVideoElement("persona-video");
 * ```
 */
export function createClient(sessionToken: string, options?: QavClientOptions): QavClient {
  return new QavClient(sessionToken, options);
}

export { QavClient } from "./client.js";
export {
  ConnectionClosedCode,
  QavEvent,
  type Message,
  type MessageStreamEvent,
  type PersonaState,
  type QavEventMap,
} from "./events.js";
export { QavError, type EngineSessionInfo, type InputAudioState, type QavClientOptions } from "./types.js";
