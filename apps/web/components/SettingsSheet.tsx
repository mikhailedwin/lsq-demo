"use client";

import { useEffect } from "react";
import { CloseIcon } from "./icons";
import type { Message } from "@qav/js-sdk";

export interface Catalog {
  avatars: { id: string; name: string; renderer?: string; attribution?: string }[];
  voices: { id: string; name: string; provider: string }[];
  llms: { id: string; name: string }[];
  personas: { id: string; name: string }[];
}

interface Props {
  open: boolean;
  onClose: () => void;
  tab: "persona" | "transcript";
  onTab: (t: "persona" | "transcript") => void;
  locked: boolean;
  catalog: Catalog | null;
  catalogError: string | null;
  name: string;
  onName: (v: string) => void;
  avatarId: string;
  onAvatar: (v: string) => void;
  voiceId: string;
  onVoice: (v: string) => void;
  llmId: string;
  onLlm: (v: string) => void;
  systemPrompt: string;
  onSystemPrompt: (v: string) => void;
  messages: Message[];
  partial: { role: Message["role"]; content: string } | null;
}

/** Slide-over sheet holding everything that isn't the call itself. */
export function SettingsSheet(p: Props) {
  useEffect(() => {
    if (!p.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") p.onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [p.open, p.onClose, p]);

  const avatars = p.catalog?.avatars ?? [];
  const attribution = avatars.find((a) => a.id === p.avatarId)?.attribution;

  return (
    <>
      <div className={`scrim ${p.open ? "is-open" : ""}`} onClick={p.onClose} aria-hidden={!p.open} />
      <aside
        className={`sheet ${p.open ? "is-open" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label="Session settings"
        aria-hidden={!p.open}
      >
        <header className="sheet-head">
          <div className="segmented" role="tablist">
            <button
              role="tab"
              aria-selected={p.tab === "persona"}
              className={p.tab === "persona" ? "is-active" : ""}
              onClick={() => p.onTab("persona")}
            >
              Persona
            </button>
            <button
              role="tab"
              aria-selected={p.tab === "transcript"}
              className={p.tab === "transcript" ? "is-active" : ""}
              onClick={() => p.onTab("transcript")}
            >
              Transcript
            </button>
          </div>
          <button className="icon-btn glass" onClick={p.onClose} aria-label="Close settings">
            <CloseIcon />
          </button>
        </header>

        <div className="sheet-body">
          {p.tab === "persona" ? (
            <>
              {p.catalogError && <p className="notice">Catalog unavailable — {p.catalogError}</p>}
              {p.locked && <p className="notice subtle">End the call to change these.</p>}

              <label className="field">
                <span>Name</span>
                <input value={p.name} onChange={(e) => p.onName(e.target.value)} disabled={p.locked} />
              </label>

              <label className="field">
                <span>Avatar</span>
                <select value={p.avatarId} onChange={(e) => p.onAvatar(e.target.value)} disabled={p.locked}>
                  {(avatars.length ? avatars : [{ id: p.avatarId, name: p.avatarId, renderer: undefined }]).map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.name}
                      {a.renderer && a.renderer !== "procedural" ? "  ·  GPU" : ""}
                    </option>
                  ))}
                </select>
                {attribution && <small>{attribution}</small>}
              </label>

              <label className="field">
                <span>Voice</span>
                <select value={p.voiceId} onChange={(e) => p.onVoice(e.target.value)} disabled={p.locked}>
                  {(p.catalog?.voices ?? [{ id: p.voiceId, name: p.voiceId, provider: "" }]).map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Model</span>
                <select value={p.llmId} onChange={(e) => p.onLlm(e.target.value)} disabled={p.locked}>
                  {(p.catalog?.llms ?? [{ id: p.llmId, name: p.llmId }]).map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="field">
                <span>Instructions</span>
                <textarea
                  rows={5}
                  value={p.systemPrompt}
                  onChange={(e) => p.onSystemPrompt(e.target.value)}
                  disabled={p.locked}
                />
              </label>
            </>
          ) : (
            <div className="transcript">
              {p.messages.length === 0 && !p.partial && <p className="notice subtle">Nothing said yet.</p>}
              {p.messages.map((m) => (
                <div key={m.id} className={`bubble ${m.role}`}>
                  {m.content}
                </div>
              ))}
              {p.partial && <div className={`bubble ${p.partial.role} is-partial`}>{p.partial.content}</div>}
            </div>
          )}
        </div>
      </aside>
    </>
  );
}
