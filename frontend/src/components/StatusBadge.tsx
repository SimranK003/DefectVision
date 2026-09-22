import clsx from "clsx";

const STYLES: Record<string, string> = {
  completed: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/20",
  processing: "bg-amber-500/10 text-amber-400 ring-amber-500/20",
  pending: "bg-slate-500/10 text-slate-400 ring-slate-500/20",
  failed: "bg-rose-500/10 text-rose-400 ring-rose-500/20",
  ok: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/20",
  degraded: "bg-rose-500/10 text-rose-400 ring-rose-500/20",
};

export function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium capitalize ring-1 ring-inset",
        STYLES[status] ?? "bg-slate-500/10 text-slate-400 ring-slate-500/20",
      )}
    >
      {status}
    </span>
  );
}
