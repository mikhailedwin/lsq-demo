import type { Avatar, Llm, Persona, Voice } from "./types.js";

const now = new Date(0).toISOString();

export const SEED_AVATARS: Avatar[] = [
  {
    id: "qav-nova",
    name: "Nova",
    model: "qav-face-1",
    style: {
      skin: "#f1c9a5",
      hair: "#3b2a1f",
      eyes: "#3f6d8e",
      lips: "#c46a6a",
      shirt: "#2f4858",
      background: "#e9eef5",
      hairStyle: "long",
    },
    createdAt: now,
  },
  {
    id: "qav-atlas",
    name: "Atlas",
    model: "qav-face-1",
    style: {
      skin: "#c98f63",
      hair: "#1f1a17",
      eyes: "#4a3b2a",
      lips: "#9c5a52",
      shirt: "#1f2937",
      background: "#f3f0ea",
      hairStyle: "short",
    },
    createdAt: now,
  },
  {
    id: "qav-sage",
    name: "Sage",
    model: "qav-face-1",
    style: {
      skin: "#e8b89a",
      hair: "#8a6f4e",
      eyes: "#4f7c5a",
      lips: "#b86b7a",
      shirt: "#3b5d50",
      background: "#eef4ee",
      hairStyle: "bun",
    },
    createdAt: now,
  },
];

export const SEED_VOICES: Voice[] = [
  {
    id: "voice-rachel",
    name: "Rachel (ElevenLabs)",
    provider: "elevenlabs",
    providerVoiceId: "21m00Tcm4TlvDq8ikWAM",
    language: "en",
    createdAt: now,
  },
  {
    id: "voice-adam",
    name: "Adam (ElevenLabs)",
    provider: "elevenlabs",
    providerVoiceId: "pNInz6obpgDQGcFmaJgB",
    language: "en",
    createdAt: now,
  },
  {
    id: "voice-nova-openai",
    name: "Nova (OpenAI)",
    provider: "openai",
    providerVoiceId: "nova",
    language: "en",
    createdAt: now,
  },
  {
    id: "voice-cartesia-default",
    name: "Cartesia default",
    provider: "cartesia",
    providerVoiceId: "c2ac25f9-ecc4-4f56-9095-651354df60c0",
    language: "en",
    createdAt: now,
  },
  {
    id: "voice-mock",
    name: "Mock tone (offline)",
    provider: "mock",
    providerVoiceId: "tone",
    language: "en",
    createdAt: now,
  },
];

export const SEED_LLMS: Llm[] = [
  { id: "claude-opus-5", name: "Claude Opus 5", provider: "anthropic", model: "claude-opus-5", createdAt: now },
  { id: "claude-sonnet-5", name: "Claude Sonnet 5", provider: "anthropic", model: "claude-sonnet-5", createdAt: now },
  { id: "gpt-4.1-mini", name: "GPT-4.1 mini", provider: "openai", model: "gpt-4.1-mini", createdAt: now },
  { id: "mock", name: "Mock (echo, offline)", provider: "mock", model: "mock", createdAt: now },
];

export const SEED_PERSONAS: Persona[] = [
  {
    id: "persona-nova-support",
    name: "Nova",
    avatarId: "qav-nova",
    avatarModel: "qav-face-1",
    voiceId: "voice-rachel",
    llmId: "claude-opus-5",
    systemPrompt:
      "You are Nova, a friendly customer-success representative. Keep answers short and conversational — you are speaking out loud, so avoid lists and markdown.",
    createdAt: now,
    updatedAt: now,
  },
];
