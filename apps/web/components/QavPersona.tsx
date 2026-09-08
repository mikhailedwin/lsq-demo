"use client";

import { createClient, QavEvent, type Message, type PersonaState, type QavClient } from "@qav/js-sdk";
import { useCallback, useEffect, useRef, useState } from "react";

interface Catalog {
  avatars: { id: string; name: string }[];
  voices: { id: string; name: string; provider: string }[];
  llms: { id: string; name: string }[];
  personas: { id: string; name: string }[];
}

type Phase = "idle" | "starting" | "live" | "stopping";

export function QavPersona() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const clientRef = useRef<QavClient | null>(null);

  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [name, setName] = useState("Nova");
  const [avatarId, setAvatarId] = useState("qav-nova");
  const [voiceId, setVoiceId] = useState("voice-rachel");
  const [llmId, setLlmId] = useState("claude-opus-5");
  const [systemPrompt, setSystemPrompt] = useState(
    "You are Nova, a friendly product specialist. Keep replies short and spoken-word natural.",
  );

  const [phase, setPhase] = useState<Phase>("idle");
  const [status, setStatus] = useState<string>("Ready");
  const [error, setError] = useState<string | null>(null);
  const [personaState, setPersonaState] = useState<PersonaState>("initializing");
  const [muted, setMuted] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [partial, setPartial] = useState<{ role: Message["role"]; content: string } | null>(null);
  const [talkText, setTalkText] = useState("");
  const [videoPlaying, setVideoPlaying] = useState(false);

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
    setStatus("Session ended");
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

      client.addListener(QavEvent.CONNECTION_ESTABLISHED, () => setStatus("Connected — waiting for the avatar…"));
      client.addListener(QavEvent.VIDEO_PLAY_STARTED, () => {
        setVideoPlaying(true);
        setStatus("Live — say hello");
      });
      client.addListener(QavEvent.SESSION_READY, () => setStatus("Live — say hello"));
      client.addListener(QavEvent.PERSONA_STATE_CHANGED, setPersonaState);
      client.addListener(QavEvent.MESSAGE_HISTORY_UPDATED, (m) => {
        setMessages(m);
        setPartial(null);
      });
      client.addListener(QavEvent.MESSAGE_STREAM_EVENT_RECEIVED, (ev) => {
        if (!ev.final) setPartial({ role: ev.role, content: ev.content });
      });
      client.addListener(QavEvent.MIC_PERMISSION_DENIED, () =>
        setStatus("Mic blocked — you can still type to the persona"),
      );
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
      setStatus("Failed to start");
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

  const send = async (mode: "talk" | "ask") => {
    const c = clientRef.current;
    const text = talkText.trim();
    if (!c || !text) return;
    setTalkText("");
    try {
      if (mode === "talk") await c.talk(text);
      else await c.sendUserMessage(text);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const live = phase === "live";

  return (
    <div className="grid">
      <section className="stage">
        <div className="video-wrap">
          <video ref={videoRef} id="qav-video" autoPlay playsInline />
          {!videoPlaying && (
            <div className="overlay">{phase === "idle" ? "Press Start to meet your avatar" : status}</div>
          )}
          {live && (
            <div className="badge">
              <span className={`dot ${personaState}`} />
              {personaState}
            </div>
          )}
        </div>

        <div className="controls">
          {!live ? (
            <button onClick={start} disabled={phase !== "idle"}>
              {phase === "starting" ? "Starting…" : "Start"}
            </button>
          ) : (
            <button className="danger" onClick={stop}>
              End session
            </button>
          )}
          <button className="secondary" onClick={toggleMute} disabled={!live}>
            {muted ? "Unmute mic" : "Mute mic"}
          </button>
          <button className="secondary" onClick={() => clientRef.current?.interruptPersona()} disabled={!live}>
            Interrupt
          </button>
          <span className={`status ${error ? "err" : ""}`}>{error ?? status}</span>
        </div>

        <div className="row">
          <input
            placeholder={live ? "Type something…" : "Start a session first"}
            value={talkText}
            disabled={!live}
            onChange={(e) => setTalkText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") void send("ask");
            }}
          />
          <button className="secondary" disabled={!live || !talkText.trim()} onClick={() => send("ask")}>
            Ask
          </button>
          <button className="secondary" disabled={!live || !talkText.trim()} onClick={() => send("talk")}>
            Make it say this
          </button>
        </div>
      </section>

      <aside className="side">
        <div className="card">
          <h2>Persona</h2>
          {catalogError && <p className="empty">Catalog unavailable: {catalogError}</p>}
          <div className="field">
            <label>Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} disabled={live} />
          </div>
          <div className="field">
            <label>Avatar</label>
            <select value={avatarId} onChange={(e) => setAvatarId(e.target.value)} disabled={live}>
              {(catalog?.avatars ?? [{ id: "qav-nova", name: "Nova" }]).map((a) => (
                <option key={a.id} value={a.id}>
                  {a.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Voice</label>
            <select value={voiceId} onChange={(e) => setVoiceId(e.target.value)} disabled={live}>
              {(catalog?.voices ?? [{ id: "voice-rachel", name: "Rachel", provider: "elevenlabs" }]).map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Brain</label>
            <select value={llmId} onChange={(e) => setLlmId(e.target.value)} disabled={live}>
              {(catalog?.llms ?? [{ id: "claude-opus-5", name: "Claude Opus 5" }]).map((l) => (
                <option key={l.id} value={l.id}>
                  {l.name}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>System prompt</label>
            <textarea value={systemPrompt} onChange={(e) => setSystemPrompt(e.target.value)} disabled={live} />
          </div>
        </div>

        <div className="card">
          <h2>Transcript</h2>
          <div className="transcript">
            {messages.length === 0 && !partial && <p className="empty">Nothing yet — start talking.</p>}
            {messages.map((m) => (
              <div key={m.id} className={`msg ${m.role}`}>
                {m.content}
              </div>
            ))}
            {partial && <div className={`msg ${partial.role} partial`}>{partial.content}</div>}
          </div>
        </div>
      </aside>
    </div>
  );
}
