"use client";

import type { Message } from "@qav/js-sdk";
import { useEffect, useRef } from "react";
import { CloseIcon } from "./icons";

interface Props {
  open: boolean;
  onClose: () => void;
  personaName: string;
  messages: Message[];
  partial: { role: Message["role"]; content: string } | null;
}

/**
 * The conversation, alongside the call.
 *
 * Open by default so what is said is always readable, and collapsible so the
 * face can have the screen to itself. It is a glass rail rather than a solid
 * panel: the call stays the subject and this defers to it.
 */
export function ChatRail({ open, onClose, personaName, messages, partial }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  // Follow the conversation, but don't yank the view if someone has scrolled
  // back to read something.
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;
    const nearBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 120;
    if (nearBottom) endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, partial]);

  return (
    <aside className={`rail ${open ? "is-open" : ""}`} aria-label="Conversation" aria-hidden={!open}>
      <header className="rail-head">
        <h2>Conversation</h2>
        <button className="icon-btn rail-close" onClick={onClose} aria-label="Hide conversation">
          <CloseIcon />
        </button>
      </header>

      <div className="rail-body" ref={bodyRef}>
        {messages.length === 0 && !partial ? (
          <p className="rail-empty">Say hello, or type below — everything {personaName} hears and says shows up here.</p>
        ) : (
          <>
            {messages.map((m) => (
              <div key={m.id} className={`bubble ${m.role}`}>
                {m.content}
              </div>
            ))}
            {partial && <div className={`bubble ${partial.role} is-partial`}>{partial.content}</div>}
          </>
        )}
        <div ref={endRef} />
      </div>
    </aside>
  );
}
