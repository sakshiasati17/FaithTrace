"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Database, FlaskConical, BarChart3, Stethoscope,
  Home, Lightbulb, ChevronRight, Activity, Zap, Cpu,
} from "lucide-react";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/",              label: "Overview",        icon: Home },
  { href: "/corpus",        label: "Corpus",          icon: Database },
  { href: "/experiments",   label: "Experiments",     icon: FlaskConical },
  { href: "/leaderboard",   label: "Leaderboard",     icon: BarChart3 },
  { href: "/diagnostics",   label: "Diagnostics",     icon: Stethoscope },
  { href: "/recommendations", label: "Recommendations", icon: Lightbulb },
  { href: "/optimizer",     label: "Optimizer",       icon: Zap },
  { href: "/inference",     label: "Inference",       icon: Cpu },
];

function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 flex flex-col z-40 bg-[#09090b] border-r border-zinc-800/80">

      {/* Logo */}
      <div className="px-4 py-5 border-b border-zinc-800/80">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "linear-gradient(135deg,#7c3aed,#6d28d9)" }}>
            <Activity className="w-3.5 h-3.5 text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-white tracking-tight leading-none">FaithTrace</p>
            <p className="text-[9px] text-zinc-500 mt-0.5 tracking-widest uppercase font-medium">RAG Diagnostics</p>
          </div>
        </div>
      </div>

      {/* Nav section */}
      <div className="px-2 pt-3 pb-2">
        <p className="px-2 mb-1.5 text-[9px] font-semibold text-zinc-600 uppercase tracking-widest">Platform</p>
        <nav className="space-y-0.5">
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={clsx(
                  "flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-xs font-medium transition-all duration-150 group relative",
                  active
                    ? "text-white bg-zinc-800 border border-zinc-700/80"
                    : "text-zinc-500 hover:text-zinc-200 hover:bg-zinc-800/50"
                )}
              >
                {active && (
                  <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 rounded-r-full bg-violet-500" />
                )}
                <Icon className={clsx("w-3.5 h-3.5 flex-shrink-0", active ? "text-violet-400" : "text-zinc-600 group-hover:text-zinc-400")} />
                <span className="flex-1">{label}</span>
                {active && <ChevronRight className="w-3 h-3 text-zinc-600" />}
              </Link>
            );
          })}
        </nav>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Footer */}
      <div className="px-4 py-4 border-t border-zinc-800/80">
        <div className="flex items-center gap-2 mb-2">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
          <p className="text-[10px] text-zinc-600 font-medium">All systems operational</p>
        </div>
        <p className="text-[9px] text-zinc-700 font-mono tracking-wide">v1.0.0 · Research Build</p>
      </div>
    </aside>
  );
}

export function ClientLayout({ children }: { children: React.ReactNode }) {
  const [queryClient] = useState(
    () => new QueryClient({
      defaultOptions: { queries: { staleTime: 30_000, retry: 1 } },
    })
  );

  return (
    <QueryClientProvider client={queryClient}>
      <Sidebar />
      <main className="ml-56 min-h-screen bg-[#09090b]">
        {children}
      </main>
    </QueryClientProvider>
  );
}
