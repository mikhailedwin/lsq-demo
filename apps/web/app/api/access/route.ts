import { NextResponse } from "next/server";
import { checkRate, clearFailures, clientIp, codeMatches, gateEnabled, grantAccess, recordFailure } from "@/lib/access";

/** Exchange the access code for a signed cookie. The code is never sent to the browser. */
export async function POST(req: Request) {
  if (!gateEnabled()) return NextResponse.json({ ok: true });

  const ip = clientIp(req);
  const rate = checkRate(ip);
  if (!rate.ok) {
    return NextResponse.json(
      { error: `Too many attempts. Try again in ${Math.ceil(rate.retryAfterSeconds / 60)} minutes.` },
      { status: 429, headers: { "retry-after": String(rate.retryAfterSeconds) } },
    );
  }

  const { code } = (await req.json().catch(() => ({}))) as { code?: string };
  if (!code) return NextResponse.json({ error: "Enter the access code." }, { status: 400 });

  if (!codeMatches(code)) {
    recordFailure(ip);
    // Deliberately vague, and the same shape every time: no hint about which
    // part was wrong, and no way to tell a wrong code from a locked-out IP.
    return NextResponse.json({ error: "That code doesn't work." }, { status: 401 });
  }

  clearFailures(ip);
  await grantAccess();
  return NextResponse.json({ ok: true });
}
