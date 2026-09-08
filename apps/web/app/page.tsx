import { BrandBar } from "@/components/BrandBar";
import { QavPersona } from "@/components/QavPersona";

export default function Page() {
  return (
    <main className="shell">
      <BrandBar />
      <div className="brand">
        <h1>QAV</h1>
        <span>real-time avatars over WebRTC · self-hosted</span>
      </div>
      <QavPersona />
    </main>
  );
}
