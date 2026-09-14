import { createHmac, timingSafeEqual } from "node:crypto";
import { cookies } from "next/headers";

/**
 * Access gate for a publicly-reachable demo.
 *
 * The point is not to be a security product — it is to stop a stray visitor
 * from spending your GPU, LLM and TTS budget. The code never reaches the
 * browser: it is compared server-side, and a success sets a signed httpOnly
 * cookie that the API routes check.
 *
 * A four-character alphanumeric code is ~1.7M combinations, which is only
 * defensible because of the rate limiting below. For anything longer-lived
 * than a demo, lengthen QAV_ACCESS_CODE.
 */

const COOKIE = "qav_access";
const TTL_SECONDS = 60 * 60 * 8; // one working day

/** Attempts allowed per IP before the lockout starts. */
const MAX_ATTEMPTS = 5;
const WINDOW_MS = 10 * 60_000;
const LOCKOUT_MS = 15 * 60_000;

interface Attempts {
  count: number;
  first: number;
  lockedUntil?: number;
}

// In-memory, so it resets on deploy and is per-instance. That is fine on a
// single always-on host (Railway, Fly, a VPS). On a serverless platform that
// fans out across instances this weakens: put the code behind a longer secret
// or move this map to Redis/Vercel KV if the demo becomes long-lived.
const attempts = new Map<string, Attempts>();

function secret(): string {
  const s = process.env.QAV_ACCESS_SECRET || process.env.QAV_SESSION_SECRET;
  if (!s) throw new Error("QAV_ACCESS_SECRET is not set");
  return s;
}

/** Undefined when no code is configured — the gate is then disabled entirely. */
export function accessCode(): string | undefined {
  const c = process.env.QAV_ACCESS_CODE?.trim();
  return c ? c : undefined;
}

export function gateEnabled(): boolean {
  return accessCode() !== undefined;
}

function sign(expiresAt: number): string {
  return createHmac("sha256", secret()).update(`${expiresAt}`).digest("hex");
}

function equal(a: string, b: string): boolean {
  const ab = Buffer.from(a);
  const bb = Buffer.from(b);
  // timingSafeEqual throws on a length mismatch, which would itself leak length
  if (ab.length !== bb.length) return false;
  return timingSafeEqual(ab, bb);
}

export function checkRate(ip: string): { ok: true } | { ok: false; retryAfterSeconds: number } {
  const now = Date.now();
  const a = attempts.get(ip);
  if (!a) return { ok: true };
  if (a.lockedUntil && now < a.lockedUntil) {
    return { ok: false, retryAfterSeconds: Math.ceil((a.lockedUntil - now) / 1000) };
  }
  if (now - a.first > WINDOW_MS) {
    attempts.delete(ip);
    return { ok: true };
  }
  return { ok: true };
}

export function recordFailure(ip: string): void {
  const now = Date.now();
  const a = attempts.get(ip);
  if (!a || now - a.first > WINDOW_MS) {
    attempts.set(ip, { count: 1, first: now });
    return;
  }
  a.count += 1;
  if (a.count >= MAX_ATTEMPTS) a.lockedUntil = now + LOCKOUT_MS;
}

export function clearFailures(ip: string): void {
  attempts.delete(ip);
}

/** Compare a submitted code against the configured one, in constant time. */
export function codeMatches(submitted: string): boolean {
  const expected = accessCode();
  if (!expected) return true;
  return equal(submitted.trim().toLowerCase(), expected.toLowerCase());
}

export async function grantAccess(): Promise<void> {
  const expiresAt = Date.now() + TTL_SECONDS * 1000;
  const jar = await cookies();
  jar.set(COOKIE, `${expiresAt}.${sign(expiresAt)}`, {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: TTL_SECONDS,
  });
}

/** True when the caller holds a valid, unexpired cookie (or the gate is off). */
export async function hasAccess(): Promise<boolean> {
  if (!gateEnabled()) return true;
  const raw = (await cookies()).get(COOKIE)?.value;
  if (!raw) return false;
  const [expiresAt, mac] = raw.split(".");
  if (!expiresAt || !mac) return false;
  if (Number(expiresAt) < Date.now()) return false;
  return equal(mac, sign(Number(expiresAt)));
}

/** Best-effort client IP behind Vercel / Railway / a reverse proxy. */
export function clientIp(req: Request): string {
  const h = req.headers;
  return (
    h.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    h.get("x-real-ip") ||
    h.get("cf-connecting-ip") ||
    "unknown"
  );
}
