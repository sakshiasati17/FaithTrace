"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Database,
  FlaskConical,
  BarChart3,
  Stethoscope,
  Home,
  ChevronRight,
  Zap,
  Lightbulb,
} from "lucide-react";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: Home },
  { href: "/corpus", label: "Corpus", icon: Database },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/leaderboard", label: "Leaderboard", icon: BarChart3 },
  { href: "/diagnostics", label: "Diagnostics", icon: Stethoscope },
  { href: "/recommendations", label: "Recommendations", icon: Lightbulb },
];

function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-60 bg-[#09090b] flex flex-col z-40" style={{ borderRight: "1px solid rgba(255,255,255,0.06)" }}>
      {/* Top gradient accent bar */}
      <div className="absolute top-0 left-0 right-0 h-px" style={{ background: "linear-gradient(90deg, transparent, rgba(124,58,237,0.6), rgba(168,85,247,0.3), transparent)" }} />

      {/* Logo */}
      <div className="px-5 py-5" style={{ borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 shadow-glow-sm" style={{ background: "linear-gradient(135deg, #7c3aed, #a855f7)" }}>
            <Zap className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white tracking-tight">FaithTrace</p>
            <p className="text-[10px] text-zinc-600 leading-none mt-0.5 font-medium tracking-widest uppercase">RAG Diagnostics</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200 group relative",
                active
                  ? "text-white"
                  : "text-zinc-500 hover:text-zinc-200"
              )}
              style={active ? { background: "rgba(124,58,237,0.12)" } : undefined}
            >
              {/* Active left indicator */}
              {active && (
                <span
                  className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-r-full"
                  style={{ background: "linear-gradient(180deg, #7c3aed, #a855f7)" }}
                />
              )}
              <Icon
                className={clsx(
                  "w-4 h-4 flex-shrink-0 transition-colors",
                  active ? "text-violet-400" : "text-zinc-600 group-hover:text-zinc-400"
                )}
              />
              <span className="flex-1">{label}</span>
              {active && <ChevronRight className="w-3 h-3 text-violet-600" />}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4" style={{ borderTop: "1px solid rgba(255,255,255,0.06)" }}>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
          <p className="text-[10px] text-zinc-600 font-mono tracking-wide">v0.1.0 · Research Build</p>
        </div>
      </div>
    </aside>
  );
}

export function ClientLayout({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 30_000,
            retry: 1,
          },
        },
      })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <Sidebar />
      <main className="ml-60 min-h-screen">
        {children}
      </main>
    </QueryClientProvider>
  );
}
