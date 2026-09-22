"use client";

import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { BatchStatusResponse, PredictionListItem } from "@/lib/types";
import { StatusBadge } from "@/components/StatusBadge";

const POLL_INTERVAL_MS = 1500;

export default function BatchProcessingPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [batch, setBatch] = useState<BatchStatusResponse | null>(null);
  const [items, setItems] = useState<PredictionListItem[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => () => {
    if (pollRef.current) clearInterval(pollRef.current);
  }, []);

  async function submit() {
    if (files.length === 0) return;
    setSubmitting(true);
    setError(null);
    setItems([]);
    try {
      const submitted = await api.submitBatch(files);
      const status = await api.getBatchStatus(submitted.batch_id);
      setBatch(status);
      startPolling(submitted.batch_id);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Batch submission failed");
    } finally {
      setSubmitting(false);
    }
  }

  function startPolling(batchId: string) {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const status = await api.getBatchStatus(batchId);
        setBatch(status);
        if (status.status === "completed" || status.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          const results = await api.listPredictions({ batchId, limit: 200 });
          setItems(results.items);
        }
      } catch {
        if (pollRef.current) clearInterval(pollRef.current);
      }
    }, POLL_INTERVAL_MS);
  }

  function csvField(value: string | number): string {
    const s = String(value);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  }

  function downloadCsv() {
    const header = "filename,status,top_class,top_confidence,inference_time_ms,num_detections\n";
    const rows = items.map((i) =>
      [
        i.original_filename,
        i.status,
        i.top_class ?? "",
        i.top_confidence ?? "",
        i.inference_time_ms ?? "",
        i.num_detections,
      ]
        .map(csvField)
        .join(","),
    );
    const csv = header + rows.join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `defectvision_batch_${batch?.batch_id ?? "results"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const attempted = batch ? batch.completed_images + batch.failed_images : 0;
  const progressPct = batch && batch.total_images > 0 ? (attempted / batch.total_images) * 100 : 0;

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold text-white">Batch Processing</h1>
        <p className="mt-1 text-sm text-slate-400">
          Submit many images at once. Processing happens asynchronously in the background.
        </p>
      </header>

      <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
        <div
          onClick={() => inputRef.current?.click()}
          className="flex h-32 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-slate-700 text-center hover:border-cyan-600"
        >
          <p className="text-sm font-medium text-slate-300">
            {files.length > 0 ? `${files.length} file(s) selected` : "Click to select multiple images"}
          </p>
          <p className="mt-1 text-xs text-slate-500">Up to 200 images per batch</p>
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            multiple
            className="hidden"
            onChange={(e) => setFiles(Array.from(e.target.files ?? []))}
          />
        </div>

        <button
          onClick={submit}
          disabled={files.length === 0 || submitting}
          className="mt-4 w-full rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-medium text-slate-950 transition-colors hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
        >
          {submitting ? "Submitting…" : `Submit Batch${files.length ? ` (${files.length} images)` : ""}`}
        </button>

        {error && (
          <div className="mt-4 rounded-lg border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
            {error}
          </div>
        )}
      </div>

      {batch && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-sm font-semibold text-white">Batch {batch.batch_id.slice(0, 8)}</h2>
            <StatusBadge status={batch.status} />
          </div>

          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-2 rounded-full bg-cyan-500 transition-all"
              style={{ width: `${progressPct}%` }}
            />
          </div>
          <div className="mt-2 flex justify-between text-xs text-slate-500">
            <span>
              {attempted} / {batch.total_images} processed
            </span>
            <span>
              {batch.completed_images} succeeded · {batch.failed_images} failed
            </span>
          </div>

          {items.length > 0 && (
            <div className="mt-6">
              <div className="mb-3 flex items-center justify-between">
                <h3 className="text-sm font-semibold text-white">Results</h3>
                <button
                  onClick={downloadCsv}
                  className="rounded-md border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-800"
                >
                  Download CSV
                </button>
              </div>
              <div className="max-h-96 overflow-y-auto">
                <table className="w-full text-left text-sm">
                  <thead className="sticky top-0 bg-slate-900">
                    <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                      <th className="pb-2 pr-4">File</th>
                      <th className="pb-2 pr-4">Status</th>
                      <th className="pb-2 pr-4">Top Defect</th>
                      <th className="pb-2">Confidence</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {items.map((item) => (
                      <tr key={item.id}>
                        <td className="py-2 pr-4 text-slate-300">{item.original_filename}</td>
                        <td className="py-2 pr-4">
                          <StatusBadge status={item.status} />
                        </td>
                        <td className="py-2 pr-4 text-slate-300">{item.top_class ?? "—"}</td>
                        <td className="py-2 text-slate-300">
                          {item.top_confidence ? `${(item.top_confidence * 100).toFixed(0)}%` : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
