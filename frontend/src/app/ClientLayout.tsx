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
  Activity,
} from "lucide-react";
import { clsx } from "clsx";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: Home },
  { href: "/corpus", label: "Corpus", icon: Database },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/leaderboard", label: "Leaderboard", icon: BarChart3 },
  { href: "/diagnostics", label: "Diagnostics", icon: Stethoscope },
];

function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 h-screen w-60 bg-zinc-950 border-r border-zinc-800 flex flex-col z-40">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-zinc-800">
        <div className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-violet-600 flex items-center justify-center">
            <Activity className="w-4 h-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-semibold text-white tracking-tight">FaithTrace</p>
            <p className="text-[10px] text-zinc-500 leading-none mt-0.5">RAG Diagnostics</p>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2 rounded-md text-sm font-medium transition-all duration-150",
                active
                  ? "bg-violet-600/20 text-violet-300"
                  : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/60"
              )}
            >
              <Icon className={clsx("w-4 h-4", active ? "text-violet-400" : "text-zinc-500")} />
              {label}
              {active && <ChevronRight className="w-3 h-3 ml-auto text-violet-500" />}
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-zinc-800">
        <p className="text-[10px] text-zinc-600">v0.1.0 — Research Build</p>
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
