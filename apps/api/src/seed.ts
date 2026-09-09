import type { Avatar, Llm, Persona, Voice } from "./types.js";

const now = new Date(0).toISOString();

/**
 * Avatar catalog. `procedural` avatars render on CPU inside the engine (a
 * stylised placeholder). `musetalk` / `liveavatar` avatars need the qav-face
 * GPU worker; their assets are the demo clips/images shipped with those
 * repositories (MIT / Apache-2.0), prepared by `qav-face demo-avatars`.
 */
export const SEED_AVATARS: Avatar[] = [
  // --- photoreal, MuseTalk 1.5 (lip-sync on a real clip) --------------------
  {
    id: "mt-yongen",
    name: "Yongen (photoreal)",
    renderer: "musetalk",
    model: "musetalk-v1.5",
    assets: { avatarDir: "yongen" },
    attribution: "Sample clip data/video/yongen.mp4 from TMElyralab/MuseTalk (MIT)",
    createdAt: now,
  },
  {
    id: "mt-sun",
    name: "Sun (photoreal)",
    renderer: "musetalk",
    model: "musetalk-v1.5",
    assets: { avatarDir: "sun" },
    attribution: "Sample clip data/video/sun.mp4 from TMElyralab/MuseTalk (MIT)",
    createdAt: now,
  },
  // --- generative, Live Avatar (Wan2.2-S2V-14B, 80 GB GPU) --------------------
  {
    id: "la-anchor",
    name: "Anchor (Live Avatar)",
    renderer: "liveavatar",
    model: "liveavatar-v1.1",
    assets: { avatarDir: "anchor" },
    attribution: "examples/anchor.jpg from Alibaba-Quark/LiveAvatar (Apache-2.0)",
    createdAt: now,
  },
  {
    id: "la-fashion-blogger",
    name: "Fashion blogger (Live Avatar)",
    renderer: "liveavatar",
    model: "liveavatar-v1.1",
    assets: { avatarDir: "fashion_blogger" },
    attribution: "examples/fashion_blogger.jpg from Alibaba-Quark/LiveAvatar (Apache-2.0)",
    createdAt: now,
  },
  {
    id: "la-kitchen-grandmother",
    name: "Kitchen grandmother (Live Avatar)",
    renderer: "liveavatar",
    model: "liveavatar-v1.1",
    assets: { avatarDir: "kitchen_grandmother" },
    attribution: "examples/kitchen_grandmother.jpg from Alibaba-Quark/LiveAvatar (Apache-2.0)",
    createdAt: now,
  },
  // --- CPU placeholder (no GPU) -----------------------------------------------
  {
    id: "qav-nova",
    name: "Nova (placeholder, CPU)",
    renderer: "procedural",
    model: "qav-face-1",
    style: { skin: "#f1c9a5", hair: "#3b2a1f", eyes: "#3f6d8e", lips: "#c46a6a", shirt: "#2f4858", background: "#e9eef5", hairStyle: "long" },
    createdAt: now,
  },
  {
    id: "qav-atlas",
    name: "Atlas (placeholder, CPU)",
    renderer: "procedural",
    model: "qav-face-1",
    style: { skin: "#c98f63", hair: "#1f1a17", eyes: "#4a3b2a", lips: "#9c5a52", shirt: "#1f2937", background: "#f3f0ea", hairStyle: "short" },
    createdAt: now,
  },
  {
    id: "qav-sage",
    name: "Sage (placeholder, CPU)",
    renderer: "procedural",
    model: "qav-face-1",
    style: { skin: "#e8b89a", hair: "#8a6f4e", eyes: "#4f7c5a", lips: "#b86b7a", shirt: "#3b5d50", background: "#eef4ee", hairStyle: "bun" },
    createdAt: now,
  },
];

export const SEED_VOICES: Voice[] = [
  { id: "voice-rachel", name: "Rachel (ElevenLabs)", provider: "elevenlabs", providerVoiceId: "21m00Tcm4TlvDq8ikWAM", language: "en", createdAt: now },
  { id: "voice-adam", name: "Adam (ElevenLabs)", provider: "elevenlabs", providerVoiceId: "pNInz6obpgDQGcFmaJgB", language: "en", createdAt: now },
  { id: "voice-nova-openai", name: "Nova (OpenAI)", provider: "openai", providerVoiceId: "nova", language: "en", createdAt: now },
  { id: "voice-cartesia-default", name: "Cartesia default", provider: "cartesia", providerVoiceId: "c2ac25f9-ecc4-4f56-9095-651354df60c0", language: "en", createdAt: now },
  { id: "voice-mock", name: "Mock tone (offline)", provider: "mock", providerVoiceId: "tone", language: "en", createdAt: now },
];

export const SEED_LLMS: Llm[] = [
  { id: "claude-opus-5", name: "Claude Opus 5", provider: "anthropic", model: "claude-opus-5", createdAt: now },
  { id: "claude-sonnet-5", name: "Claude Sonnet 5", provider: "anthropic", model: "claude-sonnet-5", createdAt: now },
  { id: "gpt-4.1-mini", name: "GPT-4.1 mini", provider: "openai", model: "gpt-4.1-mini", createdAt: now },
  { id: "mock", name: "Mock (echo, offline)", provider: "mock", model: "mock", createdAt: now },
];

export const SEED_PERSONAS: Persona[] = [
  {
    id: "persona-yongen-support",
    name: "Yongen",
    avatarId: "mt-yongen",
    avatarModel: "musetalk-v1.5",
    voiceId: "voice-rachel",
    llmId: "claude-opus-5",
    systemPrompt:
      "You are Yongen, a friendly customer-success representative. Keep answers short and conversational — you are speaking out loud, so avoid lists and markdown.",
    createdAt: now,
    updatedAt: now,
  },
];
