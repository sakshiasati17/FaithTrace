import type { Metadata } from "next";
import "./globals.css";
import { ClientLayout } from "./ClientLayout";

export const metadata: Metadata = {
  title: "FaithTrace — RAG Diagnostics Platform",
  description: "Temporal + Multimodal RAG Diagnostics Platform for enterprise knowledge systems",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-zinc-100 antialiased">
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  );
}
