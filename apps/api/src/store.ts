import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { SEED_AVATARS, SEED_LLMS, SEED_PERSONAS, SEED_VOICES } from "./seed.js";
import type { Avatar, Llm, Persona, Session, Voice } from "./types.js";

interface Snapshot {
  avatars: Avatar[];
  voices: Voice[];
  llms: Llm[];
  personas: Persona[];
  sessions: Session[];
}

/**
 * Tiny in-memory store with optional JSON persistence. Swap for Postgres by
 * keeping the same method surface — routes never touch the maps directly.
 */
export class Store {
  readonly avatars = new Map<string, Avatar>();
  readonly voices = new Map<string, Voice>();
  readonly llms = new Map<string, Llm>();
  readonly personas = new Map<string, Persona>();
  readonly sessions = new Map<string, Session>();

  private saveTimer: NodeJS.Timeout | null = null;

  constructor(private readonly dataFile: string) {
    for (const a of SEED_AVATARS) this.avatars.set(a.id, a);
    for (const v of SEED_VOICES) this.voices.set(v.id, v);
    for (const l of SEED_LLMS) this.llms.set(l.id, l);
    for (const p of SEED_PERSONAS) this.personas.set(p.id, p);
    this.load();
  }

  private load(): void {
    if (!this.dataFile) return;
    try {
      const snap = JSON.parse(readFileSync(this.dataFile, "utf8")) as Partial<Snapshot>;
      for (const a of snap.avatars ?? []) this.avatars.set(a.id, a);
      for (const v of snap.voices ?? []) this.voices.set(v.id, v);
      for (const l of snap.llms ?? []) this.llms.set(l.id, l);
      for (const p of snap.personas ?? []) this.personas.set(p.id, p);
      for (const s of snap.sessions ?? []) {
        // Anything mid-flight when we died is gone; don't resurrect it as live.
        if (s.status === "active" || s.status === "starting" || s.status === "pending") {
          s.status = "ended";
          s.endedAt ??= new Date().toISOString();
        }
        this.sessions.set(s.id, s);
      }
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== "ENOENT") {
        console.warn(`[store] could not load ${this.dataFile}:`, err);
      }
    }
  }

  /** Debounced write so a burst of transcript updates costs one disk write. */
  persist(): void {
    if (!this.dataFile) return;
    if (this.saveTimer) return;
    this.saveTimer = setTimeout(() => {
      this.saveTimer = null;
      const snap: Snapshot = {
        avatars: [...this.avatars.values()],
        voices: [...this.voices.values()],
        llms: [...this.llms.values()],
        personas: [...this.personas.values()],
        sessions: [...this.sessions.values()].slice(-500),
      };
      try {
        mkdirSync(dirname(this.dataFile), { recursive: true });
        writeFileSync(this.dataFile, JSON.stringify(snap, null, 2));
      } catch (err) {
        console.warn(`[store] could not write ${this.dataFile}:`, err);
      }
    }, 250);
  }
}
