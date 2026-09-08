import { z } from "zod";

/** Palette + geometry hints consumed by the procedural face renderer. */
export const AvatarStyleSchema = z.object({
  skin: z.string(),
  hair: z.string(),
  eyes: z.string(),
  lips: z.string(),
  shirt: z.string(),
  background: z.string(),
  hairStyle: z.enum(["short", "long", "bun", "bald"]).default("short"),
});
export type AvatarStyle = z.infer<typeof AvatarStyleSchema>;

export const AvatarSchema = z.object({
  id: z.string(),
  name: z.string(),
  /** Which renderer family produces this face. `qav-face-1` = built-in procedural. */
  model: z.string().default("qav-face-1"),
  style: AvatarStyleSchema,
  createdAt: z.string(),
});
export type Avatar = z.infer<typeof AvatarSchema>;

export const VoiceSchema = z.object({
  id: z.string(),
  name: z.string(),
  provider: z.enum(["elevenlabs", "cartesia", "openai", "mock"]),
  providerVoiceId: z.string(),
  language: z.string().default("en"),
  createdAt: z.string(),
});
export type Voice = z.infer<typeof VoiceSchema>;

export const LlmSchema = z.object({
  id: z.string(),
  name: z.string(),
  provider: z.enum(["anthropic", "openai", "mock"]),
  model: z.string(),
  createdAt: z.string(),
});
export type Llm = z.infer<typeof LlmSchema>;

/** Mirrors Anam's `personaConfig` shape so integrations port 1:1. */
export const PersonaConfigSchema = z.object({
  type: z.enum(["ephemeral", "stateful"]).optional(),
  name: z.string().min(1).max(120),
  avatarId: z.string(),
  avatarModel: z.string().optional(),
  voiceId: z.string(),
  llmId: z.string().optional(),
  systemPrompt: z.string().max(20_000).optional(),
  /** Free-form JSON handed to the engine (tools config, user context, ...). */
  metadata: z.record(z.unknown()).optional(),
});
export type PersonaConfig = z.infer<typeof PersonaConfigSchema>;

export const PersonaSchema = PersonaConfigSchema.omit({ type: true }).extend({
  id: z.string(),
  createdAt: z.string(),
  updatedAt: z.string(),
});
export type Persona = z.infer<typeof PersonaSchema>;

export const SessionOptionsSchema = z.object({
  videoWidth: z.number().int().min(128).max(1920).optional(),
  videoHeight: z.number().int().min(128).max(1920).optional(),
  /** Seconds of user silence before the session is torn down (0 = never). */
  idleTimeoutSeconds: z.number().int().min(0).max(3600).optional(),
  /** Persona greets the user as soon as it joins. */
  greetOnJoin: z.boolean().optional(),
});
export type SessionOptions = z.infer<typeof SessionOptionsSchema>;

export const CreateSessionTokenSchema = z.object({
  personaConfig: PersonaConfigSchema.optional(),
  personaId: z.string().optional(),
  sessionOptions: SessionOptionsSchema.optional(),
  /** Seconds. Defaults to QAV_SESSION_TTL_SECONDS. */
  expiresIn: z.number().int().min(60).max(86_400).optional(),
  /** Bring-your-own LiveKit (like Anam's `environment`): renderer joins your room instead. */
  environment: z
    .object({
      livekitUrl: z.string().url(),
      livekitToken: z.string(),
    })
    .optional(),
});
export type CreateSessionTokenBody = z.infer<typeof CreateSessionTokenSchema>;

export type SessionStatus = "pending" | "starting" | "active" | "ended" | "failed";

export interface Session {
  id: string;
  status: SessionStatus;
  personaConfig: PersonaConfig;
  sessionOptions: SessionOptions;
  environment?: { livekitUrl: string; livekitToken: string };
  roomName?: string;
  dispatchId?: string;
  clientIdentity?: string;
  createdAt: string;
  startedAt?: string;
  endedAt?: string;
  transcript: TranscriptMessage[];
  error?: string;
}

export interface TranscriptMessage {
  role: "user" | "persona";
  content: string;
  at: string;
}

/** What `POST /v1/engine/session` hands the browser SDK. */
export interface EngineSessionResponse {
  sessionId: string;
  livekitUrl: string;
  livekitToken: string;
  roomName: string;
  engineIdentity: string;
  avatarIdentity: string;
  persona: { name: string; avatarId: string };
}

/** What the API hands the engine worker as job metadata. */
export interface EngineJobMetadata {
  sessionId: string;
  personaConfig: PersonaConfig;
  sessionOptions: SessionOptions;
  avatar: Avatar | null;
  voice: Voice | null;
  llm: Llm | null;
  callbackUrl: string;
  callbackToken: string;
}
