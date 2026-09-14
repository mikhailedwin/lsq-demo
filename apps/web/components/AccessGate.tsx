"use client";

import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

const LENGTH = 4;

/**
 * The code is checked server-side and exchanged for a signed cookie; nothing
 * here ever holds it. One box per character, because a short code reads better
 * that way and it makes the expected length obvious.
 */
export function AccessGate() {
  const router = useRouter();
  const [chars, setChars] = useState<string[]>(Array(LENGTH).fill(""));
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const boxes = useRef<(HTMLInputElement | null)[]>([]);

  const submit = async (code: string) => {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/access", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ code }),
      });
      const body = (await res.json().catch(() => ({}))) as { error?: string };
      if (!res.ok) {
        setError(body.error ?? "That code doesn't work.");
        setChars(Array(LENGTH).fill(""));
        // re-enable first, then focus: focusing a still-disabled input is a
        // no-op and would silently swallow the next keystrokes
        setBusy(false);
        requestAnimationFrame(() => boxes.current[0]?.focus());
        return;
      }
      router.refresh(); // the server component re-renders with the cookie set
    } catch {
      setError("Couldn't reach the server. Check your connection.");
    } finally {
      setBusy(false);
    }
  };

  const setAt = (i: number, v: string) => {
    const next = [...chars];
    next[i] = v;
    setChars(next);
    // clear a previous failure as soon as they start again, so the old message
    // never sits under a code they're still typing
    if (error) setError(null);
    if (v && i < LENGTH - 1) boxes.current[i + 1]?.focus();
    const code = next.join("");
    if (code.length === LENGTH && !next.includes("")) void submit(code);
  };

  return (
    <div className="gate">
      <div className="gate-card glass">
        <header className="gate-brand">
          <img src="/logos/lux-sanans.png" alt="Lux Sanans" width={760} height={117} />
          <img src="/logos/quantanite.png" alt="Quantanite" width={512} height={107} />
        </header>

        <h1>Enter access code</h1>
        <p>This demo is limited to invited guests.</p>

        <div className="gate-boxes">
          {chars.map((c, i) => (
            <input
              key={i}
              ref={(el) => {
                boxes.current[i] = el;
              }}
              value={c}
              disabled={busy}
              inputMode="text"
              autoCapitalize="characters"
              autoComplete="off"
              spellCheck={false}
              maxLength={1}
              aria-label={`Character ${i + 1} of ${LENGTH}`}
              autoFocus={i === 0}
              onChange={(e) => {
                const v = e.target.value.replace(/[^a-zA-Z0-9]/g, "").slice(-1);
                setAt(i, v);
              }}
              onKeyDown={(e) => {
                if (e.key === "Backspace" && !chars[i] && i > 0) boxes.current[i - 1]?.focus();
                if (e.key === "ArrowLeft" && i > 0) boxes.current[i - 1]?.focus();
                if (e.key === "ArrowRight" && i < LENGTH - 1) boxes.current[i + 1]?.focus();
              }}
              onPaste={(e) => {
                // let someone paste the whole code into any box
                e.preventDefault();
                const text = e.clipboardData.getData("text").replace(/[^a-zA-Z0-9]/g, "").slice(0, LENGTH);
                if (!text) return;
                const next = Array(LENGTH).fill("");
                [...text].forEach((ch, k) => (next[k] = ch));
                setChars(next);
                if (text.length === LENGTH) void submit(text);
                else boxes.current[text.length]?.focus();
              }}
            />
          ))}
        </div>

        {error && <p className="gate-error">{error}</p>}
        {busy && <p className="gate-busy">Checking…</p>}
      </div>
    </div>
  );
}
