import assert from "node:assert/strict";
import { test } from "node:test";

process.env.QAV_API_KEY = "test-key";
process.env.QAV_SESSION_SECRET = "test-secret";

const { createApp } = await import("../src/app.js");
const { Store } = await import("../src/store.js");
const { verifyToken } = await import("../src/tokens.js");

const app = createApp(new Store(""));
const auth = { authorization: "Bearer test-key", "content-type": "application/json" };

test("rejects requests without the API key", async () => {
  const res = await app.request("/v1/personas");
  assert.equal(res.status, 401);
});

test("lists seeded catalogs", async () => {
  for (const path of ["/v1/avatars", "/v1/voices", "/v1/llms", "/v1/personas"]) {
    const res = await app.request(path, { headers: auth });
    assert.equal(res.status, 200, path);
    const body = (await res.json()) as { data: unknown[] };
    assert.ok(body.data.length > 0, `${path} should be seeded`);
  }
});

test("mints a session token bound to an ephemeral persona", async () => {
  const res = await app.request("/v1/auth/session-token", {
    method: "POST",
    headers: auth,
    body: JSON.stringify({
      personaConfig: {
        name: "Test",
        avatarId: "qav-nova",
        voiceId: "voice-mock",
        llmId: "mock",
        systemPrompt: "hi",
      },
    }),
  });
  assert.equal(res.status, 200);
  const body = (await res.json()) as { sessionToken: string; sessionId: string };
  const claims = await verifyToken(body.sessionToken, "session");
  assert.equal(claims.sid, body.sessionId);

  // A session token must not unlock the admin surface.
  const admin = await app.request("/v1/personas", { headers: { authorization: `Bearer ${body.sessionToken}` } });
  assert.equal(admin.status, 401);
});

test("rejects unknown avatar / voice references", async () => {
  const res = await app.request("/v1/auth/session-token", {
    method: "POST",
    headers: auth,
    body: JSON.stringify({ personaConfig: { name: "x", avatarId: "nope", voiceId: "voice-mock" } }),
  });
  assert.equal(res.status, 400);
});

test("engine session requires a session token", async () => {
  const res = await app.request("/v1/engine/session", { method: "POST" });
  assert.equal(res.status, 401);
  const body = (await res.json()) as { error?: string };
  assert.match(body.error ?? "", /session token/, "must be gated by the session-token check, not the admin one");
});

test("a session token unlocks the engine route (admin middleware must not intercept it)", async () => {
  const mint = await app.request("/v1/auth/session-token", {
    method: "POST",
    headers: auth,
    body: JSON.stringify({ personaConfig: { name: "t", avatarId: "qav-nova", voiceId: "voice-mock", llmId: "mock" } }),
  });
  const { sessionToken } = (await mint.json()) as { sessionToken: string };
  const res = await app.request("/v1/engine/session", {
    method: "POST",
    headers: { authorization: `Bearer ${sessionToken}`, "content-type": "application/json" },
    body: "{}",
  });
  // No LiveKit in unit tests → the room/dispatch step fails with 502; anything but 401 proves auth passed.
  assert.notEqual(res.status, 401);
  assert.ok([200, 502].includes(res.status), `unexpected status ${res.status}`);

  const stop = await app.request("/v1/engine/session/stop", {
    method: "POST",
    headers: { authorization: `Bearer ${sessionToken}` },
  });
  assert.equal(stop.status, 200);
});
