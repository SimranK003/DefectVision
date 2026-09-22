"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { ModelInfo } from "@/lib/types";
import { StatCard } from "@/components/StatCard";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

interface PerClassMetric {
  class_name: string;
  precision: number;
  recall: number;
  mAP50: number;
  mAP50_95: number;
}

export default function ModelInformationPage() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .model()
      .then(setModel)
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load model info"));
  }, []);

  if (error) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold text-white">Model Information</h1>
        <div className="rounded-lg border border-amber-800 bg-amber-950/30 px-4 py-3 text-sm text-amber-300">
          {error}. Train a model with <code>ml/scripts/train.py</code>, evaluate it with{" "}
          <code>ml/scripts/evaluate.py</code>, then promote it with <code>ml/scripts/promote_model.py</code>.
        </div>
      </div>
    );
  }

  if (!model) {
    return <p className="text-sm text-slate-500">Loading…</p>;
  }

  const metrics = model.test_metrics ?? model.val_metrics;
  const metricsLabel = model.test_metrics ? "Held-out test set" : "Validation set (no test-set evaluation recorded yet)";
  const perClass = (metrics.per_class as unknown as PerClassMetric[] | undefined) ?? null;
  const evalDir = `${API_BASE_URL}/ml-static/reports/evaluation`;

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-semibold text-white">Model Information</h1>
        <p className="mt-1 text-sm text-slate-400">
          Version {model.version} · trained {new Date(model.trained_at).toLocaleString()}
        </p>
      </header>

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="mAP@50" value={`${(metrics.mAP50 * 100).toFixed(1)}%`} sublabel={metricsLabel} />
        <StatCard label="mAP@50-95" value={`${(metrics.mAP50_95 * 100).toFixed(1)}%`} sublabel={metricsLabel} />
        <StatCard label="Precision" value={`${(metrics.precision * 100).toFixed(1)}%`} sublabel={metricsLabel} />
        <StatCard label="Recall" value={`${(metrics.recall * 100).toFixed(1)}%`} sublabel={metricsLabel} />
      </section>

      <section className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="mb-3 text-sm font-semibold text-white">Training Configuration</h2>
          <dl className="space-y-2 text-sm">
            <Row label="Dataset version" value={model.dataset_version} />
            <Row label="Classes" value={model.classes.join(", ") || "—"} />
            <Row label="Hardware" value={model.hardware} />
            <Row label="Epochs run" value={String(model.config.epochs_run ?? "—")} />
            <Row label="Config file" value={String(model.config.train_config_path ?? "—")} />
          </dl>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
          <h2 className="mb-3 text-sm font-semibold text-white">Per-Class Performance</h2>
          {!perClass ? (
            <p className="text-sm text-slate-500">
              Run <code>ml/scripts/evaluate.py</code> to populate per-class metrics.
            </p>
          ) : (
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                  <th className="pb-2 pr-2">Class</th>
                  <th className="pb-2 pr-2">P</th>
                  <th className="pb-2 pr-2">R</th>
                  <th className="pb-2 pr-2">mAP50</th>
                  <th className="pb-2">mAP50-95</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60">
                {perClass.map((row) => (
                  <tr key={row.class_name}>
                    <td className="py-2 pr-2 capitalize text-slate-300">{row.class_name.replace(/_/g, " ")}</td>
                    <td className="py-2 pr-2 text-slate-300">{(row.precision * 100).toFixed(0)}%</td>
                    <td className="py-2 pr-2 text-slate-300">{(row.recall * 100).toFixed(0)}%</td>
                    <td className="py-2 pr-2 text-slate-300">{(row.mAP50 * 100).toFixed(0)}%</td>
                    <td className="py-2 text-slate-300">{(row.mAP50_95 * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      <section className="rounded-xl border border-slate-800 bg-slate-900/50 p-5">
        <h2 className="mb-4 text-sm font-semibold text-white">Evaluation Artifacts</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <EvalImage src={`${evalDir}/confusion_matrix.png`} label="Confusion Matrix" />
          <EvalImage src={`${evalDir}/PR_curve.png`} label="Precision-Recall Curve" />
          <EvalImage src={`${evalDir}/sample_predictions.jpg`} label="Sample Predictions" />
        </div>
      </section>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between border-b border-slate-800/60 pb-2">
      <dt className="text-slate-500">{label}</dt>
      <dd className="max-w-[60%] truncate text-right text-slate-300">{value}</dd>
    </div>
  );
}

function EvalImage({ src, label }: { src: string; label: string }) {
  const [failed, setFailed] = useState(false);
  return (
    <div>
      <p className="mb-1.5 text-xs font-medium text-slate-500">{label}</p>
      {failed ? (
        <div className="flex h-40 items-center justify-center rounded-lg border border-slate-800 bg-slate-950 text-xs text-slate-600">
          Not generated yet
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={src}
          alt={label}
          className="h-40 w-full rounded-lg border border-slate-800 object-contain bg-white"
          onError={() => setFailed(true)}
        />
      )}
    </div>
  );
}
