import type { Detection } from "@/lib/types";

export function DetectionTable({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return (
      <div className="rounded-lg border border-emerald-800 bg-emerald-950/30 px-4 py-3 text-sm text-emerald-300">
        No defects detected above the confidence threshold.
      </div>
    );
  }

  return (
    <table className="w-full text-left text-sm">
      <thead>
        <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
          <th className="pb-2 pr-4">Defect Class</th>
          <th className="pb-2 pr-4">Confidence</th>
          <th className="pb-2">Bounding Box (normalized)</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-800/60">
        {detections
          .slice()
          .sort((a, b) => b.confidence - a.confidence)
          .map((d) => (
            <tr key={d.id}>
              <td className="py-2.5 pr-4 font-medium capitalize text-slate-200">
                {d.class_name.replace(/_/g, " ")}
              </td>
              <td className="py-2.5 pr-4">
                <span className="inline-flex items-center gap-2">
                  <span className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-800">
                    <span
                      className="block h-full rounded-full bg-cyan-500"
                      style={{ width: `${d.confidence * 100}%` }}
                    />
                  </span>
                  <span className="text-slate-300">{(d.confidence * 100).toFixed(1)}%</span>
                </span>
              </td>
              <td className="py-2.5 font-mono text-xs text-slate-500">
                ({d.x_min.toFixed(2)}, {d.y_min.toFixed(2)}) → ({d.x_max.toFixed(2)}, {d.y_max.toFixed(2)})
              </td>
            </tr>
          ))}
      </tbody>
    </table>
  );
}
