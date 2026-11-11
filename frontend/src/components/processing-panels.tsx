import { cn } from "@/lib/utils";
import { fmtTime, type Calibration, type FrameUpdate } from "@/lib/api";

const LEGEND: { key: string; label: string; color: string }[] = [
  { key: "road_boundary", label: "Road boundary", color: "#2ca02c" },
  { key: "lane_divider", label: "Lane divider", color: "#ff7f0e" },
  { key: "pedestrian_crossing", label: "Pedestrian crossing", color: "#1f77b4" },
];
const CLASS_OF: Record<string, string> = { road_boundary: "road_boundary", road_edge_candidate: "road_boundary", pedestrian_crossing: "pedestrian_crossing" };

function PanelShell({ title, frame, children, footer, className }: { title: string; frame: FrameUpdate | null; children: React.ReactNode; footer?: React.ReactNode; className?: string }) {
  return (
    <section className={cn("flex min-w-0 flex-col rounded-xl border border-border bg-card/90 backdrop-blur-sm", className)} aria-label={title}>
      <header className="flex items-center justify-between gap-2 border-b border-border px-3 py-2">
        <h2 className="truncate font-mono text-[11px] font-semibold tracking-[0.06em] text-foreground" title={title}>{title}</h2>
        <span className="shrink-0 font-mono text-[11px] text-muted-foreground tabular" data-testid="panel-frame">
          {frame ? `#${frame.frame_index + 1} · ${fmtTime(frame.timestamp_seconds)}` : "—"}
        </span>
      </header>
      <div className="relative flex flex-1 items-start justify-center bg-black/60 p-2">{children}</div>
      {footer && <footer className="border-t border-border px-3 py-2 text-[11px] text-muted-foreground">{footer}</footer>}
    </section>
  );
}

function Placeholder({ text, tall }: { text: string; tall?: boolean }) {
  return (
    <div className={cn("flex w-full items-center justify-center rounded border border-dashed border-border/70 px-6 text-center text-xs text-muted-foreground", tall ? "aspect-[2/3]" : "aspect-video")}>
      {text}
    </div>
  );
}

export function ProcessingPanels({ frame, cal, processingStatus }: { frame: FrameUpdate | null; cal: Calibration | null; processingStatus: string }) {
  const bevBlocked = cal?.status === "failed";
  const counts: Record<string, number> = {};
  frame?.vector_candidates.forEach((v) => { const k = CLASS_OF[v.type] ?? "lane_divider"; counts[k] = (counts[k] ?? 0) + 1; });
  const scaleNote = cal && !bevBlocked ? `RELATIVE SCALE · grid in camera heights (h) · ROI ${cal.roi_units_h?.x_max.toFixed(1)}h ahead, ±${cal.roi_units_h?.y_max.toFixed(1)}h` : "";

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-[1.7fr_1fr_1fr]" data-testid="panels">
      <PanelShell
        title="01 — ORIGINAL DASHCAM FEED"
        frame={frame}
        footer={
          <div className="flex flex-wrap justify-between gap-x-4 gap-y-1 tabular">
            <span>Frame {frame ? `${frame.frame_index + 1} / ${frame.total_frames}` : "—"}</span>
            <span>Video time {frame ? fmtTime(frame.timestamp_seconds) : "—"}</span>
            <span>Status: {processingStatus}</span>
          </div>
        }
      >
        {frame ? (
          <img className="w-full rounded" src={`data:image/jpeg;base64,${frame.original_frame}`} alt={`Original frame ${frame.frame_index + 1}`} />
        ) : (
          <Placeholder text="The uploaded video is decoded frame by frame as a simulated live camera feed." />
        )}
      </PanelShell>

      <PanelShell
        title="02 — INVERSE PERSPECTIVE MAPPING"
        frame={frame}
        footer={
          bevBlocked ? <span className="text-red-300">No validated road-plane transformation: the BEV is not generated rather than shown misleadingly.</span>
          : <div className="space-y-0.5"><div className="font-mono text-[10.5px] text-sky-300">{scaleNote}</div><div>Hatched = ground the camera cannot see (never filled in). Flat-road assumption: raised objects smear. Red box = approximate ego vehicle.</div></div>
        }
      >
        {bevBlocked ? <Placeholder tall text="BEV unavailable for this video (automatic road-plane estimation failed)." />
          : frame?.bev_frame ? <img className="w-full rounded" src={`data:image/jpeg;base64,${frame.bev_frame}`} alt={`IPM of frame ${frame.frame_index + 1}`} />
          : <Placeholder tall text="IPM of the current frame, using the automatically estimated calibration." />}
      </PanelShell>

      <PanelShell
        title="03 — VECTORIZED ROAD REPRESENTATION"
        frame={frame}
        footer={
          <div className="space-y-1">
            <ul className="grid grid-cols-1 gap-0.5">
              {LEGEND.map((l) => (
                <li key={l.key} className="flex items-center gap-2">
                  <span className="inline-block h-0 w-5 border-t-[3px]" style={{ borderColor: l.color }} />
                  <span className="flex-1">{l.label}</span>
                  <span className="font-mono tabular text-foreground">{frame ? counts[l.key] ?? 0 : ""}</span>
                </li>
              ))}
            </ul>
            <p className="text-[10.5px]">Extracted from the video with traditional CV (top-hat paint detection, road-surface scan, sliding-window curve fitting). Evidence is fused over ~5 frames using optical-flow ego-motion. No map data; not an HD map.</p>
          </div>
        }
      >
        {bevBlocked ? <Placeholder tall text="No vector output: there is no trustworthy BEV for this video." />
          : frame?.vectorized_bev_frame ? (
            <div className="relative w-full">
              <img className="w-full rounded" src={`data:image/jpeg;base64,${frame.vectorized_bev_frame}`} alt={`Vectorized BEV of frame ${frame.frame_index + 1}`} />
              {frame.vector_candidates.length === 0 && (
                <p className="sr-only" role="status">No reliable vector candidates detected in this frame.</p>
              )}
            </div>
          )
          : <Placeholder tall text="Polylines extracted from the current frame's BEV with traditional CV." />}
      </PanelShell>
    </div>
  );
}
