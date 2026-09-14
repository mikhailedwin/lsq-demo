import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  // What the client sees in the browser tab and in a shared link preview, so
  // it carries the same positioning as the hero rather than the plumbing.
  title: "Project LSQ — the new face of customer experience",
  description: "A live, on-brand AI persona your customers can talk to, 24/7.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
