import { SignJWT, jwtVerify } from "jose";
import { config } from "./config.js";

const secret = new TextEncoder().encode(config.sessionSecret);

export interface SessionTokenClaims {
  /** "session" for browser tokens, "engine" for the worker's callback token. */
  typ: "session" | "engine";
  sid: string;
  /** Where the SDK should call `POST /v1/engine/session`. */
  api: string;
}

/**
 * Session tokens are the browser-facing credential (Anam tier 2): short-lived,
 * bound to one session record, useless for anything but starting that session.
 */
export async function signSessionToken(sessionId: string, ttlSeconds: number): Promise<string> {
  return new SignJWT({ typ: "session", sid: sessionId, api: config.publicUrl } satisfies SessionTokenClaims)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setIssuer("qav")
    .setExpirationTime(`${ttlSeconds}s`)
    .sign(secret);
}

/** Lets the engine worker post transcript/status updates back for its own session only. */
export async function signEngineCallbackToken(sessionId: string): Promise<string> {
  return new SignJWT({ typ: "engine", sid: sessionId, api: config.publicUrl } satisfies SessionTokenClaims)
    .setProtectedHeader({ alg: "HS256" })
    .setIssuedAt()
    .setIssuer("qav")
    .setExpirationTime("12h")
    .sign(secret);
}

export async function verifyToken(token: string, expected: SessionTokenClaims["typ"]): Promise<SessionTokenClaims> {
  const { payload } = await jwtVerify(token, secret, { issuer: "qav" });
  if (payload.typ !== expected || typeof payload.sid !== "string") {
    throw new Error("wrong token type");
  }
  return { typ: expected, sid: payload.sid, api: String(payload.api ?? "") };
}
