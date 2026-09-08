import { NextResponse } from "next/server";

/**
 * Server-side token exchange — the only place the QAV API key lives.
 * The browser gets back a session token that can start exactly one session.
 */
export async function POST(req: Request) {
  const apiUrl = process.env.QAV_API_URL ?? process.env.NEXT_PUBLIC_QAV_API_URL ?? "http://localhost:8787";
  const apiKey = process.env.QAV_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "QAV_API_KEY is not set on the web server" }, { status: 500 });
  }

  const body = (await req.json().catch(() => ({}))) as {
    personaId?: string;
    avatarId?: string;
    voiceId?: string;
    llmId?: string;
    name?: string;
    systemPrompt?: string;
  };

  // Either reference a saved persona or build an ephemeral one from the form.
  const payload = body.personaId
    ? { personaId: body.personaId, sessionOptions: { greetOnJoin: true } }
    : {
        personaConfig: {
          name: body.name || "Nova",
          avatarId: body.avatarId || "mt-yongen",
          voiceId: body.voiceId || "voice-rachel",
          llmId: body.llmId || "claude-opus-5",
          systemPrompt:
            body.systemPrompt ||
            "You are Nova, a friendly product specialist. Keep replies short and spoken-word natural.",
        },
        sessionOptions: { greetOnJoin: true },
      };

  const res = await fetch(`${apiUrl}/v1/auth/session-token`, {
    method: "POST",
    headers: { "content-type": "application/json", authorization: `Bearer ${apiKey}` },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    return NextResponse.json({ error: `QAV API ${res.status}: ${text}` }, { status: res.status });
  }
  const data = (await res.json()) as { sessionToken: string; sessionId: string };
  return NextResponse.json(data);
}
