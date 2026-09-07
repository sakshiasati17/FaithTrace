"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Database, FlaskConical, BarChart3, Stethoscope,
  Home, Lightbulb, ChevronRight,
  MessageCircle,
} from "lucide-react";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/",                label: "Overview",        icon: Home,         section: "platform" },
  { href: "/corpus",          label: "Corpus",          icon: Database,     section: "platform" },
  { href: "/experiments",     label: "Experiments",     icon: FlaskConical, section: "platform" },
  { href: "/leaderboard",     label: "Leaderboard",     icon: BarChart3,    section: "platform" },
  { href: "/diagnostics",     label: "Diagnostics",     icon: Stethoscope,  section: "analysis" },
  { href: "/recommendations", label: "Recommendations", icon: Lightbulb,    section: "analysis" },
];

const SECTIONS: Record<string, string> = {
  platform: "Platform",
  analysis: "Analysis",
};

function Logo() {
  return (
    <div className="flex items-center gap-3">
      <div className="relative w-8 h-8 rounded-xl flex items-center justify-center overflow-hidden group-hover:scale-105 transition-transform"
        style={{ background: "linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #6d28d9 100%)" }}>
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="relative z-10">
          <path d="M8 1L14.5 5V11L8 15L1.5 11V5L8 1Z" stroke="white" strokeWidth="1.2" fill="none" />
          <path d="M8 5L11 7V11L8 13L5 11V7L8 5Z" fill="white" fillOpacity="0.9" />
        </svg>
        <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity" />
      </div>
      <div>
        <p className="text-[13px] font-bold text-white tracking-tight leading-none">
          Faith<span className="gradient-text">Trace</span>
        </p>
        <p className="text-[8px] text-zinc-500 mt-1 tracking-[0.2em] uppercase font-medium">RAG Diagnostics</p>
      </div>
    </div>
  );
}

function Sidebar() {
  const pathname = usePathname();

  const grouped = Object.entries(SECTIONS).map(([key, title]) => ({
    title,
    items: NAV_ITEMS.filter((n) => n.section === key),
  }));

  return (
    <aside className="fixed left-0 top-0 h-screen w-56 flex flex-col z-40 bg-[#0a0a0c]/90 backdrop-blur-xl border-r border-zinc-800/50">

      {/* Logo */}
      <div className="px-4 py-5 border-b border-zinc-800/50 group cursor-default">
        <Logo />
      </div>

      {/* Nav sections */}
      <div className="px-2 pt-3 pb-2 flex-1 overflow-y-auto">
        {grouped.map(({ title, items }) => (
          <div key={title} className="mb-3">
            <p className="px-3 mb-1.5 text-[9px] font-semibold text-zinc-600 uppercase tracking-[0.15em]">{title}</p>
            <nav className="space-y-0.5">
              {items.map(({ href, label, icon: Icon }) => {
                const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
                return (
                  <Link
                    key={href}
                    href={href}
                    className={clsx(
                      "flex items-center gap-2.5 px-3 py-[7px] rounded-lg text-[12px] font-medium transition-all duration-200 group/item relative",
                      active
                        ? "text-white bg-white/[0.06] nav-glow"
                        : "text-zinc-500 hover:text-zinc-200 hover:bg-white/[0.03]"
                    )}
                  >
                    <Icon className={clsx(
                      "w-[14px] h-[14px] flex-shrink-0 transition-colors duration-200",
                      active ? "text-violet-400" : "text-zinc-600 group-hover/item:text-zinc-400"
                    )} />
                    <span className="flex-1">{label}</span>
                    {active && <ChevronRight className="w-3 h-3 text-zinc-700" />}
                  </Link>
                );
              })}
            </nav>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-zinc-800/50">
        <div className="flex items-center gap-2 mb-2">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-40" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
          <p className="text-[10px] text-zinc-500 font-medium">All systems online</p>
        </div>
        <p className="text-[9px] text-zinc-700 font-mono tracking-wide">v1.0.0</p>
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
      <main className="ml-56 min-h-screen bg-[#09090b] page-enter">
        {children}
      </main>
    </QueryClientProvider>
  );
}
