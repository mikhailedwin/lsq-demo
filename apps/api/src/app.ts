import { randomBytes, randomUUID } from "node:crypto";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { HTTPException } from "hono/http-exception";
import { logger } from "hono/logger";
import { z } from "zod";
import { config } from "./config.js";
import { createRoom, deleteRoom, dispatchEngine, mintClientToken } from "./livekit.js";
import { Store } from "./store.js";
import { signEngineCallbackToken, signSessionToken, verifyToken } from "./tokens.js";
import {
  AvatarSchema,
  CreateSessionTokenSchema,
  LlmSchema,
  PersonaConfigSchema,
  VoiceSchema,
  type EngineJobMetadata,
  type EngineSessionResponse,
  type PersonaConfig,
  type Session,
} from "./types.js";

type Env = { Variables: { sessionId: string } };

function bearer(c: { req: { header: (n: string) => string | undefined } }): string | null {
  const h = c.req.header("authorization") ?? "";
  const m = /^Bearer\s+(.+)$/i.exec(h);
  return m ? m[1].trim() : null;
}

function nowIso(): string {
  return new Date().toISOString();
}

function publicSession(s: Session) {
  // Never echo LiveKit credentials from a BYO environment back out.
  const { environment, ...rest } = s;
  return { ...rest, environment: environment ? { livekitUrl: environment.livekitUrl } : undefined };
}

export function createApp(store: Store) {
  const app = new Hono<Env>();
  app.use("*", logger());
  app.use(
    "*",
    cors({
      origin: (o) => o || "*",
      allowHeaders: ["Authorization", "Content-Type"],
      allowMethods: ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    }),
  );

  app.onError((err, c) => {
    if (err instanceof HTTPException) {
      return c.json({ error: err.message || "error" }, err.status);
    }
    if (err instanceof z.ZodError) {
      return c.json({ error: "validation_error", issues: err.issues }, 400);
    }
    console.error(err);
    return c.json({ error: "internal_error", message: String(err) }, 500);
  });

  app.get("/health", (c) => c.json({ ok: true, service: "qav-api", livekit: config.livekit.url }));

  // ---------------------------------------------------------------------------
  // Tier 1: API key (server-to-server)
  // ---------------------------------------------------------------------------
  const admin = new Hono<Env>();
  admin.use("*", async (c, next) => {
    // `app.route("/v1", admin)` would otherwise apply this to /v1/engine/* and
    // /v1/internal/* as well; those routers carry their own token checks.
    if (c.req.path.startsWith("/v1/engine") || c.req.path.startsWith("/v1/internal")) {
      return next();
    }
    if (bearer(c) !== config.apiKey) {
      throw new HTTPException(401, { message: "invalid API key" });
    }
    await next();
  });

  // --- catalogs ---------------------------------------------------------------
  admin.get("/avatars", (c) => c.json({ data: [...store.avatars.values()] }));
  admin.get("/avatars/:id", (c) => {
    const a = store.avatars.get(c.req.param("id"));
    if (!a) throw new HTTPException(404, { message: "avatar not found" });
    return c.json(a);
  });
  admin.post("/avatars", async (c) => {
    const body = AvatarSchema.omit({ createdAt: true }).partial({ id: true }).parse(await c.req.json());
    const avatar = { ...body, id: body.id ?? `qav-${randomBytes(4).toString("hex")}`, createdAt: nowIso() };
    store.avatars.set(avatar.id, AvatarSchema.parse(avatar));
    store.persist();
    return c.json(avatar, 201);
  });

  admin.get("/voices", (c) => c.json({ data: [...store.voices.values()] }));
  admin.post("/voices", async (c) => {
    const body = VoiceSchema.omit({ createdAt: true }).partial({ id: true }).parse(await c.req.json());
    const voice = { ...body, id: body.id ?? `voice-${randomBytes(4).toString("hex")}`, createdAt: nowIso() };
    store.voices.set(voice.id, VoiceSchema.parse(voice));
    store.persist();
    return c.json(voice, 201);
  });

  admin.get("/llms", (c) => c.json({ data: [...store.llms.values()] }));
  admin.post("/llms", async (c) => {
    const body = LlmSchema.omit({ createdAt: true }).partial({ id: true }).parse(await c.req.json());
    const llm = { ...body, id: body.id ?? `llm-${randomBytes(4).toString("hex")}`, createdAt: nowIso() };
    store.llms.set(llm.id, LlmSchema.parse(llm));
    store.persist();
    return c.json(llm, 201);
  });

  // --- personas (stateful configs) ------------------------------------------
  admin.get("/personas", (c) => c.json({ data: [...store.personas.values()] }));
  admin.get("/personas/:id", (c) => {
    const p = store.personas.get(c.req.param("id"));
    if (!p) throw new HTTPException(404, { message: "persona not found" });
    return c.json(p);
  });
  admin.post("/personas", async (c) => {
    const body = PersonaConfigSchema.omit({ type: true }).parse(await c.req.json());
    validateReferences(store, body);
    const ts = nowIso();
    const persona = { ...body, id: `persona-${randomBytes(6).toString("hex")}`, createdAt: ts, updatedAt: ts };
    store.personas.set(persona.id, persona);
    store.persist();
    return c.json(persona, 201);
  });
  admin.put("/personas/:id", async (c) => {
    const existing = store.personas.get(c.req.param("id"));
    if (!existing) throw new HTTPException(404, { message: "persona not found" });
    const body = PersonaConfigSchema.omit({ type: true }).partial().parse(await c.req.json());
    const merged = { ...existing, ...body, updatedAt: nowIso() };
    validateReferences(store, merged);
    store.personas.set(existing.id, merged);
    store.persist();
    return c.json(merged);
  });
  admin.delete("/personas/:id", (c) => {
    if (!store.personas.delete(c.req.param("id"))) {
      throw new HTTPException(404, { message: "persona not found" });
    }
    store.persist();
    return c.body(null, 204);
  });

  // --- session tokens ---------------------------------------------------------
  admin.post("/auth/session-token", async (c) => {
    const body = CreateSessionTokenSchema.parse(await c.req.json().catch(() => ({})));

    let personaConfig: PersonaConfig;
    if (body.personaId) {
      const p = store.personas.get(body.personaId);
      if (!p) throw new HTTPException(404, { message: "persona not found" });
      personaConfig = { type: "stateful", ...p, ...(body.personaConfig ?? {}) };
    } else if (body.personaConfig) {
      personaConfig = { type: "ephemeral", ...body.personaConfig };
    } else {
      throw new HTTPException(400, { message: "personaConfig or personaId is required" });
    }
    validateReferences(store, personaConfig);

    const ttl = body.expiresIn ?? config.sessionTtlSeconds;
    const session: Session = {
      id: randomUUID(),
      status: "pending",
      personaConfig,
      sessionOptions: body.sessionOptions ?? {},
      environment: body.environment,
      createdAt: nowIso(),
      transcript: [],
    };
    store.sessions.set(session.id, session);
    store.persist();

    const sessionToken = await signSessionToken(session.id, ttl);
    return c.json({ sessionToken, sessionId: session.id, expiresIn: ttl });
  });

  // --- sessions ---------------------------------------------------------------
  admin.get("/sessions", (c) => {
    const data = [...store.sessions.values()].slice(-100).reverse().map(publicSession);
    return c.json({ data });
  });
  admin.get("/sessions/:id", (c) => {
    const s = store.sessions.get(c.req.param("id"));
    if (!s) throw new HTTPException(404, { message: "session not found" });
    return c.json(publicSession(s));
  });
  admin.get("/sessions/:id/transcript", (c) => {
    const s = store.sessions.get(c.req.param("id"));
    if (!s) throw new HTTPException(404, { message: "session not found" });
    return c.json({ sessionId: s.id, messages: s.transcript });
  });
  admin.post("/sessions/:id/stop", async (c) => {
    const s = store.sessions.get(c.req.param("id"));
    if (!s) throw new HTTPException(404, { message: "session not found" });
    await stopSession(store, s);
    return c.json(publicSession(s));
  });

  // ---------------------------------------------------------------------------
  // Tier 2: session token (browser SDK)
  // ---------------------------------------------------------------------------
  const engine = new Hono<Env>();
  engine.use("*", async (c, next) => {
    const tok = bearer(c);
    if (!tok) throw new HTTPException(401, { message: "session token required" });
    try {
      const claims = await verifyToken(tok, "session");
      c.set("sessionId", claims.sid);
    } catch {
      throw new HTTPException(401, { message: "invalid or expired session token" });
    }
    await next();
  });

  /**
   * Start the live session (`POST /v1/engine/session`):
   * creates the LiveKit room, dispatches the engine (which brings the face
   * renderer with it), and returns the browser's LiveKit credentials.
   */
  engine.post("/session", async (c) => {
    const s = store.sessions.get(c.get("sessionId"));
    if (!s) throw new HTTPException(404, { message: "session not found (API restarted?)" });
    if (s.status !== "pending") {
      throw new HTTPException(409, { message: `session already ${s.status}` });
    }
    const body = (await c.req.json().catch(() => ({}))) as { clientMetadata?: unknown; clientLabel?: string };

    s.status = "starting";
    s.roomName = s.environment ? `byo-${s.id}` : `qav-${s.id}`;
    s.clientIdentity = `user-${randomBytes(4).toString("hex")}`;
    s.startedAt = nowIso();

    const jobMeta: EngineJobMetadata = {
      sessionId: s.id,
      personaConfig: s.personaConfig,
      sessionOptions: s.sessionOptions,
      avatar: store.avatars.get(s.personaConfig.avatarId) ?? null,
      voice: store.voices.get(s.personaConfig.voiceId) ?? null,
      llm: s.personaConfig.llmId ? (store.llms.get(s.personaConfig.llmId) ?? null) : null,
      callbackUrl: config.publicUrl,
      callbackToken: await signEngineCallbackToken(s.id),
    };

    try {
      if (s.environment) {
        // BYO LiveKit: caller runs its own voice agent; we only supply the face.
        // The engine joins the caller's room using their token (face-only mode).
        s.dispatchId = await dispatchEngine(s.roomName, JSON.stringify({ ...jobMeta, environment: s.environment }));
      } else {
        await createRoom(s.roomName, JSON.stringify({ sessionId: s.id, clientLabel: body.clientLabel ?? null }));
        s.dispatchId = await dispatchEngine(s.roomName, JSON.stringify(jobMeta));
      }
    } catch (err) {
      s.status = "failed";
      s.error = String(err);
      s.endedAt = nowIso();
      store.persist();
      throw new HTTPException(502, { message: `could not start engine: ${String(err)}` });
    }

    const livekitToken = s.environment
      ? s.environment.livekitToken
      : await mintClientToken(s.roomName, s.clientIdentity, body.clientLabel ?? "user");

    s.status = "active";
    store.persist();

    const res: EngineSessionResponse = {
      sessionId: s.id,
      livekitUrl: s.environment?.livekitUrl ?? config.livekit.url,
      livekitToken,
      roomName: s.roomName,
      engineIdentity: config.identities.engine,
      avatarIdentity: config.identities.avatar,
      persona: { name: s.personaConfig.name, avatarId: s.personaConfig.avatarId },
    };
    return c.json(res);
  });

  engine.post("/session/stop", async (c) => {
    const s = store.sessions.get(c.get("sessionId"));
    if (!s) throw new HTTPException(404, { message: "session not found" });
    await stopSession(store, s);
    return c.json({ sessionId: s.id, status: s.status });
  });

  app.route("/v1/engine", engine);

  // ---------------------------------------------------------------------------
  // Engine → API callbacks (worker's per-session token)
  // ---------------------------------------------------------------------------
  const callbacks = new Hono<Env>();
  callbacks.use("*", async (c, next) => {
    const tok = bearer(c);
    if (!tok) throw new HTTPException(401, { message: "engine token required" });
    try {
      const claims = await verifyToken(tok, "engine");
      c.set("sessionId", claims.sid);
    } catch {
      throw new HTTPException(401, { message: "invalid engine token" });
    }
    await next();
  });

  const EngineEventSchema = z.discriminatedUnion("type", [
    z.object({ type: z.literal("status"), status: z.enum(["active", "ended", "failed"]), error: z.string().optional() }),
    z.object({
      type: z.literal("transcript"),
      role: z.enum(["user", "persona"]),
      content: z.string(),
    }),
  ]);

  callbacks.post("/events", async (c) => {
    const s = store.sessions.get(c.get("sessionId"));
    if (!s) throw new HTTPException(404, { message: "session not found" });
    const ev = EngineEventSchema.parse(await c.req.json());
    if (ev.type === "transcript") {
      s.transcript.push({ role: ev.role, content: ev.content, at: nowIso() });
    } else {
      s.status = ev.status;
      if (ev.error) s.error = ev.error;
      if (ev.status !== "active") s.endedAt ??= nowIso();
    }
    store.persist();
    return c.json({ ok: true });
  });

  // Token-scoped routers first so their handlers win for their own prefixes.
  app.route("/v1/engine", engine);
  app.route("/v1/internal", callbacks);
  app.route("/v1", admin);

  return app;
}

function validateReferences(store: Store, cfg: Pick<PersonaConfig, "avatarId" | "voiceId" | "llmId">): void {
  if (!store.avatars.has(cfg.avatarId)) {
    throw new HTTPException(400, { message: `unknown avatarId "${cfg.avatarId}"` });
  }
  if (!store.voices.has(cfg.voiceId)) {
    throw new HTTPException(400, { message: `unknown voiceId "${cfg.voiceId}"` });
  }
  if (cfg.llmId && !store.llms.has(cfg.llmId)) {
    throw new HTTPException(400, { message: `unknown llmId "${cfg.llmId}"` });
  }
}

async function stopSession(store: Store, s: Session): Promise<void> {
  if (s.status === "ended" || s.status === "failed") return;
  s.status = "ended";
  s.endedAt = nowIso();
  store.persist();
  if (s.roomName && !s.environment) {
    // Deleting the room disconnects the browser, the engine and the renderer in one go.
    await deleteRoom(s.roomName).catch((err) => console.warn("[sessions] deleteRoom failed:", err));
  }
}
