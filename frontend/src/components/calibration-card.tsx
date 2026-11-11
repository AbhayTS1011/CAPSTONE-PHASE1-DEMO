import { AlertTriangle, CheckCircle2, ChevronRight, Circle, Loader2, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Calibration, SessionState, Stage } from "@/lib/api";

// Backend stage ids -> user-facing names (in pipeline order)
const STAGES: [string, string][] = [
  ["inspect", "Inspecting video"],
  ["analysing_geometry", "Analysing camera geometry"],
  ["vanishing_points", "Detecting vanishing points"],
  ["camera_orientation", "Estimating camera orientation"],
  ["road_plane", "Estimating road plane"],
  ["validating", "Validating homography"],
  ["building_projection", "Building IPM projection"],
  ["calibration_completed", "Calibration completed"],
];

const SOURCE_STYLE: Record<string, string> = {
  measured: "bg-emerald-500/15 text-emerald-300 ring-emerald-500/30",
  estimated: "bg-sky-500/15 text-sky-300 ring-sky-500/30",
  assumed: "bg-amber-500/15 text-amber-300 ring-amber-500/30",
};

export function statusHeadline(cal: Calibration) {
  if (cal.status === "failed")
    return "Automatic road-plane estimation has low confidence. A reliable BEV transformation could not be established for this video.";
  if (cal.status === "low_confidence")
    return "Automatic calibration completed with low confidence — relative scale only. Treat the BEV geometry with caution.";
  return "Automatic calibration completed — relative scale only. Absolute road distances could not be established.";
}

function fmtVal(v: unknown) {
  if (Array.isArray(v)) return v.join(", ");
  return String(v);
}

export function CalibrationCard({ state, stages, cal, inspected }: { state: SessionState | null; stages: Stage[]; cal: Calibration | null; inspected: boolean }) {
  const seen = new Set(stages.map((s) => s.stage));
  if (inspected) seen.add("inspect");
  const running = state === "calibrating" || state === "inspecting";
  const firstOpen = STAGES.findIndex(([id]) => !seen.has(id));
  const pass = cal?.checks.filter((c) => c.status === "pass").length ?? 0;
  const warn = cal?.checks.filter((c) => c.status === "warn").length ?? 0;
  const fail = cal?.checks.filter((c) => c.status === "fail").length ?? 0;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold tracking-wide">Automatic calibration</h3>
        <span className="text-[11px] text-muted-foreground">no manual input · traditional CV only</span>
      </div>

      {!inspected && !cal && (
        <p className="mt-3 text-sm text-muted-foreground">
          Once a video is uploaded, the program estimates the camera orientation and road plane from the footage itself
          (line segments, vanishing points, optical flow). Nothing has to be entered.
        </p>
      )}

      {(inspected || cal) && (
        <ol className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs" aria-label="Calibration stages">
          {STAGES.map(([id, label], i) => {
            const done = seen.has(id) && !(running && i === firstOpen);
            const active = running && i === firstOpen;
            return (
              <li key={id} className={cn("flex items-center gap-1.5", done ? "text-foreground" : "text-muted-foreground")}>
                {done ? <CheckCircle2 className="size-3.5 text-emerald-400" /> : active ? <Loader2 className="size-3.5 animate-spin text-sky-300" /> : <Circle className="size-3.5" />}
                {label}
              </li>
            );
          })}
        </ol>
      )}

      {cal && (
        <div className="mt-4 space-y-3">
          <div
            className={cn(
              "rounded-md border p-3 text-xs leading-5",
              cal.status === "ok" && "border-sky-500/30 bg-sky-500/10 text-sky-100",
              cal.status === "low_confidence" && "border-amber-500/40 bg-amber-500/10 text-amber-100",
              cal.status === "failed" && "border-red-500/40 bg-red-500/10 text-red-100",
            )}
            role="status"
          >
            <p className="flex items-start gap-2 font-medium">
              {cal.status === "failed" ? <XCircle className="mt-0.5 size-4 shrink-0" /> : cal.status === "low_confidence" ? <AlertTriangle className="mt-0.5 size-4 shrink-0" /> : <CheckCircle2 className="mt-0.5 size-4 shrink-0" />}
              {statusHeadline(cal)}
            </p>
            {cal.reason && <p className="mt-1 pl-6 opacity-90">{cal.reason}</p>}
            {cal.scale_note && <p className="mt-1 pl-6 opacity-80">{cal.scale_note}</p>}
          </div>

          {cal.parameters.length > 0 && (
            <table className="w-full text-xs tabular">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-wider text-muted-foreground">
                  <th className="pb-1 font-medium">Parameter</th><th className="pb-1 font-medium">Value</th><th className="pb-1 font-medium">Source</th>
                </tr>
              </thead>
              <tbody>
                {cal.parameters.map((p) => (
                  <tr key={p.name} className="border-t border-border/60 align-top" title={p.method}>
                    <td className="py-1 pr-2 text-muted-foreground">{p.name}</td>
                    <td className="py-1 pr-2 font-medium">{fmtVal(p.value)} {p.unit}{p.extra ? <span className="text-muted-foreground"> · {p.extra}</span> : null}</td>
                    <td className="py-1"><span className={cn("rounded px-1.5 py-0.5 text-[10px] font-medium uppercase ring-1 ring-inset", SOURCE_STYLE[p.source])}>{p.source}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {cal.checks.length > 0 && (
            <details className="group rounded-md border border-border/70 bg-background/40 text-xs">
              <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2">
                <ChevronRight className="size-3.5 transition-transform group-open:rotate-90" />
                Quality checks (measured): <span className="text-emerald-400">{pass} pass</span>
                {warn > 0 && <span className="text-amber-400">{warn} warn</span>}
                {fail > 0 && <span className="text-red-400">{fail} fail</span>}
              </summary>
              <ul className="space-y-1 px-3 pb-3">
                {cal.checks.map((c) => (
                  <li key={c.name} className="flex gap-2">
                    <span className={cn("w-10 shrink-0 font-medium uppercase", c.status === "pass" ? "text-emerald-400" : c.status === "warn" ? "text-amber-400" : "text-red-400")}>{c.status}</span>
                    <span className="text-muted-foreground"><span className="text-foreground">{c.name.replaceAll("_", " ")}</span> = {c.value ?? "n/a"} — {c.detail}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}

          {cal.warnings.length > 0 && (
            <ul className="space-y-1 text-xs text-amber-200/90">
              {cal.warnings.map((w) => (
                <li key={w} className="flex gap-2"><AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-amber-400" />{w}</li>
              ))}
            </ul>
          )}

          {cal.preview && (
            <details className="rounded-md border border-border/70 bg-background/40 text-xs">
            <summary className="cursor-pointer px-3 py-2 text-muted-foreground">Calibration preview on a sample frame</summary>
            <div className="grid grid-cols-[1.75fr_1fr] gap-2 px-3 pb-3">
              <figure>
                <img className="w-full rounded border border-border" src={`data:image/jpeg;base64,${cal.preview.original}`} alt="Calibration preview: horizon and BEV region drawn on a sample frame" />
                <figcaption className="mt-1 text-[10px] text-muted-foreground">Frame {cal.preview.frame_index + 1}: estimated horizon (orange) and BEV region (cyan)</figcaption>
              </figure>
              <figure>
                <img className="w-full rounded border border-border" src={`data:image/jpeg;base64,${cal.preview.bev}`} alt="Calibration preview: IPM of the same frame" />
                <figcaption className="mt-1 text-[10px] text-muted-foreground">IPM of that frame</figcaption>
              </figure>
            </div>
            </details>
          )}

          {cal.diagnostics && (
            <details className="rounded-md border border-border/70 bg-background/40 text-xs">
              <summary className="cursor-pointer px-3 py-2 text-muted-foreground">Technical diagnostics (read-only)</summary>
              <pre className="max-h-72 overflow-auto px-3 pb-3 font-mono text-[10.5px] leading-4 text-muted-foreground">
                {JSON.stringify({ method: cal.method, roi_units_h: cal.roi_units_h, horizon_row_px: cal.horizon_row_px, hood_top_row_px: cal.hood_top_row_px, bev_valid_fraction: cal.bev_valid_fraction, elapsed_s: cal.elapsed_s, ...cal.diagnostics }, null, 1)}
              </pre>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
