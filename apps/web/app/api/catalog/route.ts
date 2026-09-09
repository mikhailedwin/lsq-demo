import { NextResponse } from "next/server";

/** Avatars / voices / LLMs for the persona picker (proxied so the API key stays server-side). */
export async function GET() {
  const apiUrl = process.env.QAV_API_URL ?? process.env.NEXT_PUBLIC_QAV_API_URL ?? "http://localhost:8787";
  const apiKey = process.env.QAV_API_KEY;
  if (!apiKey) return NextResponse.json({ error: "QAV_API_KEY is not set" }, { status: 500 });

  const headers = { authorization: `Bearer ${apiKey}` };
  try {
    const [avatars, voices, llms, personas] = await Promise.all(
      ["avatars", "voices", "llms", "personas"].map(async (k) => {
        const r = await fetch(`${apiUrl}/v1/${k}`, { headers, cache: "no-store" });
        if (!r.ok) throw new Error(`${k}: ${r.status}`);
        return ((await r.json()) as { data: unknown[] }).data;
      }),
    );
    return NextResponse.json({ avatars, voices, llms, personas });
  } catch (err) {
    return NextResponse.json({ error: String(err) }, { status: 502 });
  }
}
