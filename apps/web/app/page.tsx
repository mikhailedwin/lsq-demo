import { AccessGate } from "@/components/AccessGate";
import { QavPersona } from "@/components/QavPersona";
import { hasAccess } from "@/lib/access";

// The gate is checked on the server, so the app's markup is never sent to a
// visitor who hasn't unlocked it.
export const dynamic = "force-dynamic";

export default async function Page() {
  const unlocked = await hasAccess();
  return <main>{unlocked ? <QavPersona /> : <AccessGate />}</main>;
}
