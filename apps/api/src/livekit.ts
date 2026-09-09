import { AccessToken, AgentDispatchClient, RoomServiceClient } from "livekit-server-sdk";
import { config, livekitHttpUrl } from "./config.js";

const http = livekitHttpUrl(config.livekit.url);

export const roomService = new RoomServiceClient(http, config.livekit.apiKey, config.livekit.apiSecret);
export const dispatchClient = new AgentDispatchClient(http, config.livekit.apiKey, config.livekit.apiSecret);

/** Browser participant: can publish mic/camera, subscribe to the avatar. */
export async function mintClientToken(roomName: string, identity: string, name: string): Promise<string> {
  const at = new AccessToken(config.livekit.apiKey, config.livekit.apiSecret, {
    identity,
    name,
    ttl: config.sessionTtlSeconds,
  });
  at.addGrant({
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canSubscribe: true,
    canPublishData: true,
  });
  return at.toJwt();
}

export async function createRoom(roomName: string, metadata: string): Promise<void> {
  await roomService.createRoom({
    name: roomName,
    emptyTimeout: 120,
    departureTimeout: 20,
    maxParticipants: 4,
    metadata,
  });
}

export async function deleteRoom(roomName: string): Promise<void> {
  try {
    await roomService.deleteRoom(roomName);
  } catch (err) {
    // Room already gone (empty timeout) is the common case on stop.
    if (!/not found|does not exist/i.test(String(err))) throw err;
  }
}

/** Explicit dispatch: the engine worker registers as `qav-engine` and only joins rooms we tell it to. */
export async function dispatchEngine(roomName: string, metadata: string): Promise<string> {
  const d = await dispatchClient.createDispatch(roomName, config.livekit.engineAgentName, { metadata });
  return d.id;
}
