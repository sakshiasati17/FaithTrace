import Link from "next/link";
import { Home, SearchX } from "lucide-react";

export default function NotFound() {
  return (
    <div className="min-h-screen bg-[#09090b] flex flex-col items-center justify-center text-center px-6">
      <SearchX className="w-12 h-12 text-zinc-700 mb-4" />
      <h1 className="text-4xl font-bold text-white tabular-nums mb-2">404</h1>
      <p className="text-zinc-400 mb-1 text-sm">Page not found</p>
      <p className="text-zinc-600 text-xs mb-8 max-w-xs">
        The page you&apos;re looking for doesn&apos;t exist or has been moved.
      </p>
      <Link
        href="/"
        className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white rounded-lg transition-all"
        style={{ background: "linear-gradient(135deg,#7c3aed,#6d28d9)" }}
      >
        <Home className="w-3.5 h-3.5" />
        Back to Overview
      </Link>
    </div>
  );
}
