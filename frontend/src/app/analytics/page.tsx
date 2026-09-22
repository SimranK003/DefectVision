"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, ApiError } from "@/lib/api";
import type { AnalyticsSummary } from "@/lib/types";
import { StatCard } from "@/components/StatCard";

const CATEGORY_COLORS = ["#22d3ee", "#f59e0b", "#f43f5e", "#a78bfa", "#34d399", "#60a5fa"];

export default function AnalyticsPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .analyticsSummary()
      .then(setSummary)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load analytics"));
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
        {error}
      </div>
    );
  }
  if (!summary) return <p className="text-sm text-slate-500">Loading…</p>;

  const categoryData = Object.entries(summary.defects_by_category).map(([name, value]) => ({
    name: name.replace(/_/g, " "),
    value,
  }));

  const hasCategoryData = categoryData.length > 0;
  const hasConfidenceData = summary.confidence_distribution.some((b) => b.count > 0);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold text-white">Analytics</h1>
        <p className="mt-1 text-sm text-slate-400">Aggregate statistics computed from stored predictions.</p>
      </header>

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Completed Inspections" value={summary.completed_predictions.toString()} />
        <StatCard
          label="Defect Rate"
          value={`${(summary.defect_rate * 100).toFixed(1)}%`}
          accent={summary.defect_rate > 0.3 ? "rose" : "amber"}
        />
        <StatCard label="Images w/ Defects" value={summary.images_with_defects.toString()} />
        <StatCard label="Total Batch Images" value={summary.batch_stats.total_images_via_batch.toString()} />
      </section>

      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="mb-4 text-sm font-semibold text-white">Defects by Category</h2>
          {!hasCategoryData ? (
            <EmptyChart message="No defects recorded yet." />
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={categoryData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  isAnimationActive={false}
                  label
                >
                  {categoryData.map((_, i) => (
                    <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="mb-4 text-sm font-semibold text-white">Confidence Distribution</h2>
          {!hasConfidenceData ? (
            <EmptyChart message="No detections recorded yet." />
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={summary.confidence_distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="range_label" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} allowDecimals={false} />
                <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8 }} />
                <Bar dataKey="count" fill="#22d3ee" radius={[4, 4, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
        <h2 className="mb-4 text-sm font-semibold text-white">Batch Job Statistics</h2>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <MiniStat label="Pending" value={summary.batch_stats.pending} />
          <MiniStat label="Processing" value={summary.batch_stats.processing} />
          <MiniStat label="Completed" value={summary.batch_stats.completed} />
          <MiniStat label="Failed" value={summary.batch_stats.failed} />
        </div>
      </section>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg bg-slate-950 p-4 text-center">
      <p className="text-2xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-xs text-slate-500">{label}</p>
    </div>
  );
}

function EmptyChart({ message }: { message: string }) {
  return <div className="flex h-64 items-center justify-center text-sm text-slate-500">{message}</div>;
}
