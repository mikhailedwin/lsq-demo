function required(name: string, fallback?: string): string {
  const v = process.env[name] ?? fallback;
  if (v === undefined || v === "") {
    throw new Error(`${name} environment variable is required`);
  }
  return v;
}

export const config = {
  port: Number(process.env.QAV_API_PORT ?? 8787),
  /** Public URL of this API; embedded in session tokens so the SDK knows where to call. */
  publicUrl: process.env.QAV_API_URL ?? `http://localhost:${process.env.QAV_API_PORT ?? 8787}`,
  apiKey: required("QAV_API_KEY", "qav_dev_key_change_me"),
  sessionSecret: required("QAV_SESSION_SECRET", "qav_dev_session_secret_change_me"),
  sessionTtlSeconds: Number(process.env.QAV_SESSION_TTL_SECONDS ?? 3600),
  dataFile: process.env.QAV_DATA_FILE ?? "",
  livekit: {
    url: required("LIVEKIT_URL", "ws://localhost:7880"),
    apiKey: required("LIVEKIT_API_KEY", "devkey"),
    apiSecret: required("LIVEKIT_API_SECRET", "secret"),
    /** Name the engine worker registers with (WorkerOptions.agent_name). */
    engineAgentName: process.env.QAV_ENGINE_AGENT_NAME ?? "qav-engine",
  },
  identities: {
    engine: "qav-engine",
    avatar: "qav-avatar",
  },
} as const;

/** LiveKit's REST APIs speak HTTP; convert ws(s):// to http(s)://. */
export function livekitHttpUrl(url: string): string {
  return url.replace(/^ws:/, "http:").replace(/^wss:/, "https:");
}
