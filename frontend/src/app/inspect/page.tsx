"use client";

import { useRef, useState } from "react";
import { api, ApiError, resolveImageUrl } from "@/lib/api";
import type { Prediction } from "@/lib/types";
import { DetectionTable } from "@/components/DetectionTable";
import { StatusBadge } from "@/components/StatusBadge";

export default function InspectionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFile(selected: File | null) {
    setResult(null);
    setError(null);
    setFile(selected);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(selected ? URL.createObjectURL(selected) : null);
  }

  function onDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) handleFile(dropped);
  }

  async function runInspection() {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    try {
      const prediction = await api.predictSingle(file);
      setResult(prediction);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Inspection failed - is the API running?");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold text-white">Inspection</h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload a single component image to run defect detection.
        </p>
      </header>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="space-y-4">
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className="flex h-64 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-900/40 text-center transition-colors hover:border-cyan-600"
          >
            {previewUrl ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={previewUrl} alt="Selected preview" className="h-full w-full rounded-xl object-contain p-2" />
            ) : (
              <>
                <p className="text-sm font-medium text-slate-300">Drop an image here or click to browse</p>
                <p className="mt-1 text-xs text-slate-500">JPEG or PNG, up to 10&nbsp;MB</p>
              </>
            )}
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />
          </div>

          <button
            onClick={runInspection}
            disabled={!file || submitting}
            className="w-full rounded-lg bg-cyan-500 px-4 py-2.5 text-sm font-medium text-slate-950 transition-colors hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
          >
            {submitting ? "Running inference…" : "Run Inspection"}
          </button>

          {error && (
            <div className="rounded-lg border border-rose-800 bg-rose-950/40 px-4 py-3 text-sm text-rose-300">
              {error}
            </div>
          )}
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="mb-4 text-sm font-semibold text-white">Result</h2>
          {!result ? (
            <p className="text-sm text-slate-500">Run an inspection to see annotated results here.</p>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between text-sm">
                <StatusBadge status={result.status} />
                <span className="text-slate-500">
                  {result.inference_time_ms?.toFixed(1)} ms · model {result.model_version}
                </span>
              </div>

              {result.annotated_image_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={resolveImageUrl(result.annotated_image_url) ?? undefined}
                  alt="Annotated inspection result"
                  className="w-full rounded-lg border border-slate-800"
                />
              )}

              <DetectionTable detections={result.detections} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
