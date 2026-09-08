"use client";

import { createClient, QavEvent, type Message, type PersonaState, type QavClient } from "@qav/js-sdk";
import { useCallback, useEffect, useRef, useState } from "react";
import { SettingsSheet, type Catalog } from "./SettingsSheet";
import { GearIcon, MicIcon, MicOffIcon, SendIcon, WaveIcon } from "./icons";

type Phase = "idle" | "starting" | "live" | "stopping";
type Mode = "ask" | "say";

const STATE_LABEL: Record<string, string> = {
  initializing: "Connecting",
  listening: "Listening",
  thinking: "Thinking",
  speaking: "Speaking",
};

export function QavPersona() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const clientRef = useRef<QavClient | null>(null);

  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [name, setName] = useState("Nova");
  const [avatarId, setAvatarId] = useState("mt-yongen");
  const [voiceId, setVoiceId] = useState("voice-rachel");
  const [llmId, setLlmId] = useState("claude-opus-5");
  const [systemPrompt, setSystemPrompt] = useState(
    "You are Nova, a friendly product specialist. Keep replies short and spoken-word natural.",
  );

  const [phase, setPhase] = useState<Phase>("idle");
  const [status, setStatus] = useState("Ready");
  const [error, setError] = useState<string | null>(null);
  const [personaState, setPersonaState] = useState<PersonaState>("initializing");
  const [muted, setMuted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [partial, setPartial] = useState<{ role: Message["role"]; content: string } | null>(null);
  const [talkText, setTalkText] = useState("");
  const [mode, setMode] = useState<Mode>("ask");
  const [videoPlaying, setVideoPlaying] = useState(false);
  const [aspect, setAspect] = useState(1);
  const [sheet, setSheet] = useState(false);
  const [sheetTab, setSheetTab] = useState<"persona" | "transcript">("persona");

  useEffect(() => {
    fetch("/api/catalog")
      .then(async (r) => {
        const body = (await r.json()) as Catalog & { error?: string };
        if (!r.ok || body.error) throw new Error(body.error ?? `HTTP ${r.status}`);
        setCatalog(body);
        // Prefer whatever the deployment can actually voice.
        if (body.voices.length && !body.voices.some((v) => v.id === voiceId)) setVoiceId(body.voices[0].id);
        if (body.llms.length && !body.llms.some((l) => l.id === llmId)) setLlmId(body.llms[0].id);
      })
      .catch((e) => setCatalogError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stop = useCallback(async () => {
    const c = clientRef.current;
    clientRef.current = null;
    setPhase("stopping");
    await c?.stopStreaming().catch(() => undefined);
    setPhase("idle");
    setStatus("Call ended");
    setVideoPlaying(false);
    setPersonaState("initializing");
  }, []);

  const start = useCallback(async () => {
    setError(null);
    setMessages([]);
    setPartial(null);
    setPhase("starting");
    setStatus("Creating session…");
    try {
      const res = await fetch("/api/session-token", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name, avatarId, voiceId, llmId, systemPrompt }),
      });
      const body = (await res.json()) as { sessionToken?: string; error?: string };
      if (!res.ok || !body.sessionToken) throw new Error(body.error ?? `HTTP ${res.status}`);

      const client = createClient(body.sessionToken, { clientLabel: "qav-web-demo" });
      clientRef.current = client;

      client.addListener(QavEvent.CONNECTION_ESTABLISHED, () => setStatus("Waiting for the avatar…"));
      client.addListener(QavEvent.VIDEO_PLAY_STARTED, () => {
        setVideoPlaying(true);
        setStatus("");
      });
      client.addListener(QavEvent.SESSION_READY, () => setStatus(""));
      client.addListener(QavEvent.PERSONA_STATE_CHANGED, setPersonaState);
      client.addListener(QavEvent.MESSAGE_HISTORY_UPDATED, (m) => {
        setMessages(m);
        setPartial(null);
      });
      client.addListener(QavEvent.MESSAGE_STREAM_EVENT_RECEIVED, (ev) => {
        if (!ev.final) setPartial({ role: ev.role, content: ev.content });
      });
      client.addListener(QavEvent.MIC_PERMISSION_DENIED, () => setStatus("Mic blocked — you can still type"));
      client.addListener(QavEvent.SERVER_WARNING, (msg) => console.warn("[qav]", msg));
      client.addListener(QavEvent.CONNECTION_CLOSED, (code) => {
        if (clientRef.current === client) {
          clientRef.current = null;
          setPhase("idle");
          setVideoPlaying(false);
          setStatus(`Disconnected (${code})`);
        }
      });

      setStatus("Connecting…");
      await client.streamToVideoElement(videoRef.current!);
      setPhase("live");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("");
      setPhase("idle");
      clientRef.current = null;
    }
  }, [name, avatarId, voiceId, llmId, systemPrompt]);

  useEffect(() => () => void clientRef.current?.stopStreaming(), []);

  const toggleMute = async () => {
    const c = clientRef.current;
    if (!c) return;
    const s = muted ? await c.unmuteInputAudio() : await c.muteInputAudio();
    setMuted(s.isMuted);
  };

  const send = async () => {
    const c = clientRef.current;
    const text = talkText.trim();
    if (!c || !text) return;
    setTalkText("");
    try {
      if (mode === "say") await c.talk(text);
      else await c.sendUserMessage(text);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const live = phase === "live";
  const busy = phase === "starting" || phase === "stopping";
  // the last thing said, shown as a caption over the video
  const caption = partial?.content ?? [...messages].reverse().find((m) => m.role === "persona")?.content ?? "";

  return (
    <div className="app">
      <button
        className={`icon-btn floating ${sheet ? "is-hidden" : ""}`}
        onClick={() => setSheet(true)}
        aria-label="Session settings"
      >
        <GearIcon />
      </button>

      <div className="stage-col">
        {/* The card is the white area: logos live in it, above the call. */}
        <section className="stage">
          <header className="stage-brand">
            <img src="/logos/lux-sanans.png" alt="Lux Sanans" width={760} height={117} />
            <img src="/logos/quantanite.png" alt="Quantanite" width={512} height={107} />
          </header>

          {/* Before the call this is a short hero; once video arrives it takes the
              stream's own aspect ratio, whatever the renderer is configured for. */}
          <div className="viewport" style={{ aspectRatio: videoPlaying ? String(aspect) : "16 / 10" }}>
            <video
              ref={videoRef}
              id="qav-video"
              autoPlay
              playsInline
              onLoadedMetadata={(e) => {
                const v = e.currentTarget;
                if (v.videoWidth && v.videoHeight) setAspect(v.videoWidth / v.videoHeight);
              }}
            />

            {!videoPlaying && (
              <div className="placeholder">
                {phase === "idle" && !error ? (
                  <>
                    <span className="orb" aria-hidden="true" />
                    <h1>Meet {name}</h1>
                    <p>A real-time avatar you can talk to, running on your own infrastructure.</p>
                  </>
                ) : (
                  <>
                    {busy && <span className="spinner" />}
                    <p className={error ? "is-error" : ""}>{error ?? status}</p>
                  </>
                )}
              </div>
            )}

            {live && videoPlaying && (
              <>
                <div className={`state-pill ${personaState}`}>
                  <span className="dot" />
                  {STATE_LABEL[personaState] ?? personaState}
                </div>
                {caption && <div className="caption">{caption}</div>}
              </>
            )}
          </div>
        </section>

        {!live ? (
          <div className="dock">
            <button className="btn primary" onClick={start} disabled={busy}>
              {phase === "starting" ? "Starting…" : "Start call"}
            </button>
          </div>
        ) : (
          <>
            <div className="composer">
              <div className="mode" role="group" aria-label="Send mode">
                <button className={mode === "ask" ? "is-active" : ""} onClick={() => setMode("ask")}>
                  Ask
                </button>
                <button className={mode === "say" ? "is-active" : ""} onClick={() => setMode("say")}>
                  Say
                </button>
              </div>
              <input
                value={talkText}
                placeholder={mode === "ask" ? `Message ${name}…` : "Words to speak verbatim…"}
                onChange={(e) => setTalkText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void send();
                }}
              />
              <button className="icon-btn send" onClick={send} disabled={!talkText.trim()} aria-label="Send">
                <SendIcon />
              </button>
            </div>

            <div className="dock">
              <button className="btn ghost" onClick={toggleMute}>
                {muted ? <MicOffIcon /> : <MicIcon />}
                {muted ? "Unmute" : "Mute"}
              </button>
              <button className="btn ghost" onClick={() => clientRef.current?.interruptPersona()}>
                <WaveIcon />
                Interrupt
              </button>
              <button className="btn danger" onClick={stop} disabled={busy}>
                End call
              </button>
            </div>
          </>
        )}

        {error && live && <p className="inline-error">{error}</p>}
      </div>

      <SettingsSheet
        open={sheet}
        onClose={() => setSheet(false)}
        tab={sheetTab}
        onTab={setSheetTab}
        locked={live || busy}
        catalog={catalog}
        catalogError={catalogError}
        name={name}
        onName={setName}
        avatarId={avatarId}
        onAvatar={setAvatarId}
        voiceId={voiceId}
        onVoice={setVoiceId}
        llmId={llmId}
        onLlm={setLlmId}
        systemPrompt={systemPrompt}
        onSystemPrompt={setSystemPrompt}
        messages={messages}
        partial={partial}
      />
    </div>
  );
}
