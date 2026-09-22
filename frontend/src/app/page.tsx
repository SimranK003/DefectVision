"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, ApiError } from "@/lib/api";
import type { AnalyticsSummary, ModelInfo } from "@/lib/types";
import { StatCard } from "@/components/StatCard";
import { StatusBadge } from "@/components/StatusBadge";

export default function DashboardPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const data = await api.analyticsSummary();
        if (!cancelled) setSummary(data);
      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Failed to reach the API");
      } finally {
        if (!cancelled) setLoading(false);
      }

      try {
        const m = await api.model();
        if (!cancelled) setModel(m);
      } catch (err) {
        if (!cancelled) setModelError(err instanceof ApiError ? err.message : "Failed to load model info");
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-8">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white">Dashboard</h1>
          <p className="mt-1 text-sm text-slate-400">
            Live overview of inspection activity across all uploaded images.
          </p>
        </div>
        <Link
          href="/inspect"
          className="rounded-lg bg-cyan-500 px-4 py-2 text-sm font-medium text-slate-950 hover:bg-cyan-400"
        >
          New Inspection
        </Link>
      </header>

      {error && (
        <div className="rounded-lg border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
          Couldn&apos;t reach the DefectVision API: {error}
        </div>
      )}

      {loading && !error && <p className="text-sm text-slate-500">Loading dashboard…</p>}

      {summary && (
        <>
          <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Total Inspections" value={summary.total_predictions.toString()} />
            <StatCard
              label="Defect Rate"
              value={`${(summary.defect_rate * 100).toFixed(1)}%`}
              sublabel={`${summary.images_with_defects} of ${summary.completed_predictions} completed`}
              accent={summary.defect_rate > 0.3 ? "rose" : "amber"}
            />
            <StatCard
              label="Failed Inspections"
              value={summary.failed_predictions.toString()}
              accent={summary.failed_predictions > 0 ? "rose" : "emerald"}
            />
            <StatCard
              label="Batch Jobs"
              value={summary.batch_stats.total_batches.toString()}
              sublabel={`${summary.batch_stats.processing} in progress`}
            />
          </section>

          <section className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5 lg:col-span-2">
              <div className="mb-4 flex items-center justify-between">
                <h2 className="text-sm font-semibold text-white">Recent Predictions</h2>
                <Link href="/history" className="text-xs font-medium text-cyan-400 hover:text-cyan-300">
                  View all →
                </Link>
              </div>
              {summary.recent_predictions.length === 0 ? (
                <EmptyState message="No inspections yet. Upload an image to get started." />
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                        <th className="pb-2 pr-4">File</th>
                        <th className="pb-2 pr-4">Status</th>
                        <th className="pb-2 pr-4">Top Defect</th>
                        <th className="pb-2 pr-4">Confidence</th>
                        <th className="pb-2">When</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {summary.recent_predictions.map((p) => (
                        <tr key={p.id}>
                          <td className="py-2.5 pr-4 text-slate-300">{p.original_filename}</td>
                          <td className="py-2.5 pr-4">
                            <StatusBadge status={p.status} />
                          </td>
                          <td className="py-2.5 pr-4 text-slate-300">{p.top_class ?? "—"}</td>
                          <td className="py-2.5 pr-4 text-slate-300">
                            {p.top_confidence ? `${(p.top_confidence * 100).toFixed(0)}%` : "—"}
                          </td>
                          <td className="py-2.5 text-slate-500">
                            {new Date(p.created_at).toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
              <h2 className="mb-4 text-sm font-semibold text-white">Defects by Category</h2>
              {Object.keys(summary.defects_by_category).length === 0 ? (
                <EmptyState message="No defects recorded yet." />
              ) : (
                <ul className="space-y-3">
                  {Object.entries(summary.defects_by_category)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, count]) => {
                      const max = Math.max(...Object.values(summary.defects_by_category));
                      return (
                        <li key={name}>
                          <div className="mb-1 flex items-center justify-between text-xs">
                            <span className="capitalize text-slate-300">{name.replace(/_/g, " ")}</span>
                            <span className="text-slate-500">{count}</span>
                          </div>
                          <div className="h-1.5 w-full rounded-full bg-slate-800">
                            <div
                              className="h-1.5 rounded-full bg-cyan-500"
                              style={{ width: `${(count / max) * 100}%` }}
                            />
                          </div>
                        </li>
                      );
                    })}
                </ul>
              )}
            </div>
          </section>
        </>
      )}

      <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
        <h2 className="mb-3 text-sm font-semibold text-white">Production Model</h2>
        {modelError ? (
          <p className="text-sm text-amber-400">{modelError} — train and promote a model to enable inference.</p>
        ) : model ? (
          <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm text-slate-300">
            <span>
              Version <span className="font-medium text-white">{model.version}</span>
            </span>
            <span>
              mAP@50{" "}
              <span className="font-medium text-white">
                {((model.test_metrics ?? model.val_metrics).mAP50 * 100).toFixed(1)}%
              </span>
            </span>
            <span>
              Hardware <span className="font-medium text-white">{model.hardware}</span>
            </span>
            <Link href="/model" className="font-medium text-cyan-400 hover:text-cyan-300">
              Full details →
            </Link>
          </div>
        ) : (
          <p className="text-sm text-slate-500">Loading…</p>
        )}
      </section>
    </div>
  );
}

function EmptyState({ message }: { message: string }) {
  return <p className="py-6 text-center text-sm text-slate-500">{message}</p>;
}
