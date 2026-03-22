import { clsx } from "clsx";

type Status = "pending" | "running" | "done" | "failed" | string;

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-zinc-800 text-zinc-400 border-zinc-700",
  running: "bg-blue-500/10 text-blue-400 border-blue-500/30",
  done: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
  failed: "bg-red-500/10 text-red-400 border-red-500/30",
};

const STATUS_DOT: Record<string, string> = {
  pending: "bg-zinc-500",
  running: "bg-blue-400 animate-pulse",
  done: "bg-emerald-400",
  failed: "bg-red-400",
};

interface StatusBadgeProps {
  status: Status;
  className?: string;
}

export function StatusBadge({ status, className }: StatusBadgeProps) {
  const style = STATUS_STYLES[status] ?? "bg-zinc-800 text-zinc-400 border-zinc-700";
  const dot = STATUS_DOT[status] ?? "bg-zinc-500";

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border",
        style,
        className
      )}
    >
      <span className={clsx("w-1.5 h-1.5 rounded-full", dot)} />
      {status}
    </span>
  );
}
