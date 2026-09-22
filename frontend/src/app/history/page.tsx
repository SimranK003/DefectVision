"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { api, ApiError, resolveImageUrl } from "@/lib/api";
import type { Prediction, PredictionListItem, PredictionStatus } from "@/lib/types";
import { StatusBadge } from "@/components/StatusBadge";
import { DetectionTable } from "@/components/DetectionTable";

const PAGE_SIZE = 15;

export default function PredictionHistoryPage() {
  return (
    <Suspense fallback={null}>
      <PredictionHistoryContent />
    </Suspense>
  );
}

function PredictionHistoryContent() {
  const searchParams = useSearchParams();
  const [items, setItems] = useState<PredictionListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [statusFilter, setStatusFilter] = useState<PredictionStatus | "all">("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Prediction | null>(null);

  // Deep-link support: /history?id=<prediction_id> opens that prediction's
  // detail modal directly - lets a row be shared/bookmarked, not just clicked.
  useEffect(() => {
    const id = searchParams.get("id");
    if (id) openDetail(id);
  }, [searchParams]);

  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: show a spinner while re-fetching on page/filter change
    setLoading(true);
    api
      .listPredictions({
        limit: PAGE_SIZE,
        offset,
        status: statusFilter === "all" ? undefined : statusFilter,
      })
      .then((res) => {
        if (cancelled) return;
        setItems(res.items);
        setTotal(res.total);
        setError(null);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.message : "Failed to load predictions");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [offset, statusFilter]);

  async function openDetail(id: string) {
    try {
      const detail = await api.getPrediction(id);
      setSelected(detail);
    } catch {
      // detail view is a nice-to-have; leave the row clickable but silently no-op on failure
    }
  }

  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-6">
      <header className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white">Prediction History</h1>
          <p className="mt-1 text-sm text-slate-400">Every inspection ever submitted, newest first.</p>
        </div>
        <select
          value={statusFilter}
          onChange={(e) => {
            setStatusFilter(e.target.value as PredictionStatus | "all");
            setOffset(0);
          }}
          className="rounded-md border border-slate-700 bg-slate-900 px-3 py-1.5 text-sm text-slate-300"
        >
          <option value="all">All statuses</option>
          <option value="completed">Completed</option>
          <option value="processing">Processing</option>
          <option value="pending">Pending</option>
          <option value="failed">Failed</option>
        </select>
      </header>

      {error && (
        <div className="rounded-lg border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
        {loading ? (
          <p className="py-6 text-center text-sm text-slate-500">Loading…</p>
        ) : items.length === 0 ? (
          <p className="py-6 text-center text-sm text-slate-500">No predictions match this filter.</p>
        ) : (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                <th className="pb-2 pr-4">File</th>
                <th className="pb-2 pr-4">Status</th>
                <th className="pb-2 pr-4">Top Defect</th>
                <th className="pb-2 pr-4">Confidence</th>
                <th className="pb-2 pr-4">Detections</th>
                <th className="pb-2 pr-4">Inference</th>
                <th className="pb-2 pr-4">Model</th>
                <th className="pb-2">When</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {items.map((p) => (
                <tr
                  key={p.id}
                  onClick={() => openDetail(p.id)}
                  className="cursor-pointer hover:bg-slate-800/40"
                >
                  <td className="py-2.5 pr-4 text-slate-300">{p.original_filename}</td>
                  <td className="py-2.5 pr-4">
                    <StatusBadge status={p.status} />
                  </td>
                  <td className="py-2.5 pr-4 capitalize text-slate-300">
                    {p.top_class?.replace(/_/g, " ") ?? "—"}
                  </td>
                  <td className="py-2.5 pr-4 text-slate-300">
                    {p.top_confidence ? `${(p.top_confidence * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className="py-2.5 pr-4 text-slate-300">{p.num_detections}</td>
                  <td className="py-2.5 pr-4 text-slate-300">
                    {p.inference_time_ms ? `${p.inference_time_ms.toFixed(0)} ms` : "—"}
                  </td>
                  <td className="py-2.5 pr-4 text-slate-500">{p.model_version ?? "—"}</td>
                  <td className="py-2.5 text-slate-500">{new Date(p.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        <div className="mt-4 flex items-center justify-between text-sm text-slate-500">
          <span>
            Page {page} of {totalPages} · {total} total
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              disabled={offset === 0}
              className="rounded-md border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-40"
            >
              Previous
            </button>
            <button
              onClick={() => setOffset(offset + PAGE_SIZE)}
              disabled={offset + PAGE_SIZE >= total}
              className="rounded-md border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-40"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {selected && (
        <div
          className="fixed inset-0 flex items-center justify-center bg-black/60 p-6"
          onClick={() => setSelected(null)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-xl border border-slate-800 bg-slate-900 p-6"
          >
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-white">{selected.original_filename}</h3>
              <button onClick={() => setSelected(null)} className="text-slate-500 hover:text-slate-300">
                ✕
              </button>
            </div>
            {selected.annotated_image_url && (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={resolveImageUrl(selected.annotated_image_url) ?? undefined}
                alt="Annotated result"
                className="mb-4 w-full rounded-lg border border-slate-800"
              />
            )}
            <DetectionTable detections={selected.detections} />
          </div>
        </div>
      )}
    </div>
  );
}
