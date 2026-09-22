export function StatCard({
  label,
  value,
  sublabel,
  accent = "cyan",
}: {
  label: string;
  value: string;
  sublabel?: string;
  accent?: "cyan" | "rose" | "emerald" | "amber";
}) {
  const accentClasses: Record<string, string> = {
    cyan: "text-cyan-400",
    rose: "text-rose-400",
    emerald: "text-emerald-400",
    amber: "text-amber-400",
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
      <p className="text-xs font-medium uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-2 text-3xl font-semibold ${accentClasses[accent]}`}>{value}</p>
      {sublabel && <p className="mt-1 text-xs text-slate-500">{sublabel}</p>}
    </div>
  );
}
