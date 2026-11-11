import { useCallback, useEffect, useRef, useState } from "react";
import { AlertTriangle, Pause, Play, RotateCcw, Square, Wifi, WifiOff } from "lucide-react";
import FileUpload05, { type LocalMeta, type UploadPhase } from "@/components/ui/file-upload-1";
import { LiquidMetalButton, type LiquidMetalState } from "@/components/ui/liquid-metal-button";
import { SparklesCore } from "@/components/ui/sparkles";
import { Button } from "@/components/ui/button";
import { CalibrationCard } from "@/components/calibration-card";
import { ProcessingPanels } from "@/components/processing-panels";
import { cn } from "@/lib/utils";
import {
  deleteVideo, fmtTime, getHealth, uploadVideo, wsUrl,
  type ApiError, type Calibration, type FrameUpdate, type Inspection, type RunSummary, type ServerMessage, type SessionState, type Stage,
} from "@/lib/api";

const prefersReducedMotion = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

export default function App() {
  const [maxMb, setMaxMb] = useState(2048);
  const [backendDown, setBackendDown] = useState(false);
  // upload
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<UploadPhase>("idle");
  const [progress, setProgress] = useState(0);
  const [localMeta, setLocalMeta] = useState<LocalMeta | null>(null);
  const [inspection, setInspection] = useState<Inspection | null>(null);
  const [uploadError, setUploadError] = useState<ApiError | null>(null);
  // session
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [state, setState] = useState<SessionState | null>(null);
  const [stages, setStages] = useState<Stage[]>([]);
  const [cal, setCal] = useState<Calibration | null>(null);
  const [frame, setFrame] = useState<FrameUpdate | null>(null);
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [sessionError, setSessionError] = useState<ApiError | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [wsStatus, setWsStatus] = useState<"idle" | "open" | "reconnecting" | "lost">("idle");
  const [starting, setStarting] = useState(false);
  const [pace, setPace] = useState(true);
  const [record, setRecord] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const sessionRef = useRef<string | null>(null);
  const abortRef = useRef<(() => void) | null>(null);
  const retryRef = useRef(0);

  useEffect(() => {
    getHealth().then((h) => { setMaxMb(h.max_upload_mb); setBackendDown(false); }).catch(() => setBackendDown(true));
  }, []);

  const handleMessage = useCallback((m: ServerMessage) => {
    switch (m.type) {
      case "session_state":
        setState(m.state); setStages(m.calibration_stages); setCal(m.calibration); setSummary(m.summary);
        if (m.inspection) setInspection(m.inspection);
        if (m.error) setSessionError(m.error);
        setPace(m.pace);
        break;
      case "status":
        setState(m.state);
        if (m.state !== "ready") setStarting(false);
        if (m.summary) setSummary(m.summary);
        if (m.error) setSessionError(m.error);
        if (m.state === "processing") setSummary(null);
        break;
      case "calibration_stage":
        setStages((s) => [...s, { stage: m.stage, detail: m.detail, t: m.t }]);
        break;
      case "calibration_result":
        setCal(m.calibration);
        break;
      case "frame_update":
        setFrame(m);                         // one message carries all three panels of ONE frame
        break;
      case "notice":
        setNotice(m.message);
        break;
      case "error":
        setSessionError(m.error);
        break;
    }
  }, []);

  const connect = useCallback((sid: string) => {
    const ws = new WebSocket(wsUrl(sid));
    wsRef.current = ws;
    ws.onopen = () => { retryRef.current = 0; setWsStatus("open"); };
    ws.onmessage = (ev) => handleMessage(JSON.parse(ev.data));
    ws.onclose = () => {
      if (sessionRef.current !== sid || wsRef.current !== ws) return;   // closed on purpose
      if (retryRef.current < 6) {
        retryRef.current += 1;
        setWsStatus("reconnecting");
        setTimeout(() => sessionRef.current === sid && connect(sid), 500 * 2 ** retryRef.current);
      } else {
        setWsStatus("lost");
      }
    };
  }, [handleMessage]);

  const resetSession = useCallback(async () => {
    const sid = sessionRef.current;
    sessionRef.current = null;
    const ws = wsRef.current; wsRef.current = null; ws?.close();
    if (sid) await deleteVideo(sid);
    setSessionId(null); setState(null); setStages([]); setCal(null); setFrame(null); setSummary(null);
    setSessionError(null); setNotice(null); setWsStatus("idle"); setStarting(false);
  }, []);

  const onSelect = useCallback(async (f: File) => {
    abortRef.current?.();
    setUploadError(null); setInspection(null); setLocalMeta(null); setProgress(0); setPhase("reading"); setFile(f);
    await resetSession();
    if (!f.name.toLowerCase().endsWith(".mp4")) {
      setFile(f); setPhase("error");
      setUploadError({ code: "unsupported_extension", message: `"${f.name}" is not an .mp4 file.`, hint: "Only MP4 dashcam videos are accepted." });
      return;
    }
    if (f.size > maxMb * 1024 * 1024) {
      setFile(f); setPhase("error");
      setUploadError({ code: "too_large", message: `The file is ${(f.size / 1e6).toFixed(0)} MB; the limit is ${maxMb} MB.`, hint: "Trim the video or raise PSJ_MAX_UPLOAD_MB for the backend." });
      return;
    }
    setFile(f); setPhase("reading");
    // quick local preview of duration/resolution (the backend's inspection is authoritative)
    const url = URL.createObjectURL(f);
    await new Promise<void>((resolve) => {
      const v = document.createElement("video");
      v.preload = "metadata";
      v.onloadedmetadata = () => { setLocalMeta({ duration: v.duration, width: v.videoWidth, height: v.videoHeight }); resolve(); };
      v.onerror = () => resolve();             // browser may not decode the codec; the backend decides
      v.src = url;
      setTimeout(resolve, 3000);
    });
    URL.revokeObjectURL(url);
    setPhase("uploading");
    const up = uploadVideo(f, (p) => { setProgress(p); if (p >= 1) setPhase("validating"); });
    abortRef.current = up.abort;
    try {
      const res = await up.promise;
      abortRef.current = null;
      setInspection(res.inspection); setPhase("done"); setState("calibrating");
      sessionRef.current = res.session_id; setSessionId(res.session_id);
      connect(res.session_id);
    } catch (e) {
      abortRef.current = null;
      const err = e as ApiError;
      if (err.code === "aborted") { setPhase("idle"); return; }
      setPhase("error"); setUploadError(err);
    }
  }, [connect, maxMb, resetSession]);

  const onRemove = useCallback(async () => {
    abortRef.current?.(); abortRef.current = null;
    await resetSession();
    setFile(null); setPhase("idle"); setProgress(0); setLocalMeta(null); setInspection(null); setUploadError(null);
  }, [resetSession]);

  const send = (cmd: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd));
    else setNotice("Not connected to the backend; the command was not sent.");
  };

  const start = () => {
    if (state === "paused") { send({ cmd: "resume" }); return; }
    setFrame(null); setSummary(null); setStarting(true);
    send({ cmd: "start", pace, record });
  };

  // ---------- derived status ----------
  const processing = state === "processing";
  const calibrated = !!cal && state !== "calibrating";
  const headerStatus =
    backendDown ? "Backend not reachable"
    : sessionError || state === "error" ? "Error"
    : !file || phase === "idle" ? "Waiting for video"
    : phase === "error" ? "Error"
    : phase === "reading" || phase === "uploading" ? "Video uploaded"
    : phase === "validating" ? "Analysing video"
    : state === "calibrating" || state === "inspecting" || (!cal && state !== null) ? "Automatically calibrating"
    : processing ? "Processing"
    : state === "paused" ? "Paused"
    : state === "completed" ? "Completed"
    : state === "stopped" ? "Stopped"
    : cal?.status === "failed" ? "Calibration warning"
    : state === "ready" ? "Ready to process"
    : "Calibration completed";
  const statusTone =
    headerStatus === "Error" || headerStatus === "Backend not reachable" ? "red"
    : headerStatus === "Calibration warning" ? "amber"
    : processing ? "green" : headerStatus === "Completed" || headerStatus === "Ready to process" ? "sky" : "muted";

  const btnState: LiquidMetalState =
    sessionError || state === "error" ? "error"
    : starting ? "loading"
    : processing ? "processing"
    : state === "paused" ? "paused"
    : state === "completed" || state === "stopped" ? "completed"
    : state === "ready" ? "ready"
    : state === "calibrating" ? "loading"
    : "disabled";
  const btnLabel =
    btnState === "processing" ? "PROCESSING" : btnState === "paused" ? "RESUME PROCESSING"
    : btnState === "completed" ? "START PROCESSING" : btnState === "error" ? "ERROR"
    : btnState === "loading" ? (state === "calibrating" ? "CALIBRATING" : "STARTING") : "START PROCESSING";

  const m = frame?.metrics;
  const notKeepingUp = m && m.keeping_up_with_source === false;

  return (
    <div className="relative min-h-full">
      {/* Sparkles background: behind everything, non-interactive, unmounted while frames are processed */}
      {!processing && !prefersReducedMotion && (
        <div className="pointer-events-none fixed inset-0 -z-10 opacity-60" aria-hidden="true">
          <SparklesCore id="bg-sparkles" background="transparent" minSize={0.4} maxSize={1.1} particleDensity={40}
                        speed={0.8} particleColor="#9fb3d1" fpsLimit={30} className="h-full w-full" />
        </div>
      )}
      <div className="pointer-events-none fixed inset-0 -z-20 bg-[radial-gradient(ellipse_at_top,rgba(56,92,140,0.18),transparent_60%)]" aria-hidden="true" />

      <div className="mx-auto max-w-[1680px] px-4 py-5 sm:px-6">
        {/* ---------------- header ---------------- */}
        <header className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-[0.14em] sm:text-2xl">EDGE-DEPLOYED VECTOR-BASED ROAD MAPPING</h1>
            <p className="mt-1 font-mono text-xs text-muted-foreground sm:text-sm">Monocular Dashcam → Automatic Calibration → IPM → Vectorized BEV</p>
            <p className="mt-1 text-xs text-muted-foreground">
              The uploaded MP4 is processed frame by frame as a <span className="text-foreground">simulated live camera feed</span>. Everything runs locally; no neural networks, no map data.
            </p>
          </div>
          <div className="flex flex-col items-end gap-1.5">
            <span data-testid="status-pill" className={cn(
              "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium ring-1 ring-inset",
              statusTone === "red" && "bg-red-500/15 text-red-200 ring-red-500/40",
              statusTone === "amber" && "bg-amber-500/15 text-amber-200 ring-amber-500/40",
              statusTone === "green" && "bg-emerald-500/15 text-emerald-200 ring-emerald-500/40",
              statusTone === "sky" && "bg-sky-500/15 text-sky-200 ring-sky-500/40",
              statusTone === "muted" && "bg-secondary text-muted-foreground ring-border",
            )}>
              <span className={cn("size-1.5 rounded-full bg-current", processing && "animate-pulse")} />
              {headerStatus}
            </span>
            {calibrated && cal?.status !== "failed" && (
              <span className="rounded-full bg-secondary px-2.5 py-0.5 text-[11px] text-sky-200 ring-1 ring-inset ring-border">Relative-scale mode</span>
            )}
            {sessionId && (
              <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
                {wsStatus === "open" ? <Wifi className="size-3 text-emerald-400" /> : <WifiOff className="size-3 text-amber-400" />}
                {wsStatus === "open" ? "stream connected" : wsStatus === "reconnecting" ? "reconnecting…" : wsStatus === "lost" ? "stream lost" : "connecting…"}
              </span>
            )}
          </div>
        </header>

        {backendDown && (
          <Banner tone="red">The local backend is not reachable at http://127.0.0.1:8000. Start it with <code className="font-mono">start_backend.bat</code> and reload this page.</Banner>
        )}
        {wsStatus === "lost" && <Banner tone="red">The streaming connection to the backend was lost. Processing was paused on the server; reload the page or re-upload the video.</Banner>}
        {sessionError && <Banner tone="red">{sessionError.message}{sessionError.hint ? ` — ${sessionError.hint}` : ""}</Banner>}

        {/* ---------------- upload | calibration | controls ---------------- */}
        <div className="mt-4 grid grid-cols-1 gap-3 lg:grid-cols-[minmax(320px,1fr)_minmax(380px,1.5fr)_minmax(280px,0.8fr)]">
          <Card className="lg:max-h-[330px] lg:overflow-y-auto">
            <FileUpload05 file={file} phase={phase} progress={progress} localMeta={localMeta} inspection={inspection} error={uploadError}
                          maxSizeMb={maxMb} disabled={backendDown} onSelect={onSelect} onRemove={onRemove} />
            {inspection && (
              <p className="mt-3 text-[11px] leading-4 text-muted-foreground">{inspection.metadata_note}</p>
            )}
          </Card>
          <Card className="lg:max-h-[330px] lg:overflow-y-auto">
            <CalibrationCard state={state} stages={stages} cal={cal} inspected={!!inspection} />
          </Card>
          <Card className="flex flex-col gap-3 lg:max-h-[330px] lg:overflow-y-auto">
            <h3 className="text-sm font-semibold tracking-wide">Processing</h3>
            <div className="flex justify-center py-1">
              <LiquidMetalButton label={btnLabel} state={btnState} width={250} onClick={start} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <Button variant="secondary" size="sm" disabled={!processing} onClick={() => send({ cmd: "pause" })}><Pause className="mr-1.5 size-3.5" />Pause</Button>
              <Button variant="secondary" size="sm" disabled={state !== "paused"} onClick={() => send({ cmd: "resume" })}><Play className="mr-1.5 size-3.5" />Resume</Button>
              <Button variant="secondary" size="sm" disabled={!processing && state !== "paused"} onClick={() => send({ cmd: "stop" })}><Square className="mr-1.5 size-3.5" />Stop</Button>
              <Button variant="secondary" size="sm" disabled={!sessionId || !cal || state === "calibrating"} onClick={() => { send({ cmd: "reset" }); setFrame(null); setSummary(null); }}><RotateCcw className="mr-1.5 size-3.5" />Reset</Button>
            </div>
            <label className="flex cursor-pointer items-start gap-2 text-xs text-muted-foreground">
              <input type="checkbox" className="mt-0.5 accent-sky-400" checked={pace}
                     onChange={(e) => { setPace(e.target.checked); if (sessionId) send({ cmd: "set_pace", pace: e.target.checked }); }} />
              <span><span className="text-foreground">Pace to source FPS</span> (simulate a live camera). Off = process as fast as the computer allows; frames are never skipped either way.</span>
            </label>
            <label className="flex cursor-pointer items-start gap-2 text-xs text-muted-foreground">
              <input type="checkbox" className="mt-0.5 accent-sky-400" checked={record} disabled={processing || state === "paused"}
                     onChange={(e) => setRecord(e.target.checked)} />
              <span><span className="text-foreground">Record demo video</span> (1920×1080 MP4 in the same layout as the PSJ_01 final-demo video; slows processing a little).</span>
            </label>
            {notice && <p className="text-[11px] text-muted-foreground">{notice}</p>}
            {summary && (
              <div className="rounded-md border border-border bg-background/50 p-3 text-xs tabular" data-testid="run-summary">
                <p className="font-medium text-foreground">Run summary (measured)</p>
                <p className="mt-1 text-muted-foreground">{summary.frames_processed} / {summary.total_frames} frames in {summary.wall_time_s.toFixed(1)} s · mean {summary.mean_processing_ms?.toFixed(1)} ms/frame (p95 {summary.p95_processing_ms?.toFixed(1)}) · compute {summary.mean_processing_fps?.toFixed(1)} fps · delivered {summary.effective_fps?.toFixed(1)} fps · {summary.frames_with_vector_candidates} frames with vector candidates</p>
                {summary.demo_video && (
                  <a href={summary.demo_video} download className="mt-2 inline-flex items-center rounded-md bg-sky-500/20 px-3 py-1.5 text-xs font-medium text-sky-100 ring-1 ring-inset ring-sky-400/40 hover:bg-sky-500/30" data-testid="demo-download">
                    Download demo video ({summary.demo_video_frames} frames)
                  </a>
                )}
              </div>
            )}
          </Card>
        </div>

        {/* ---------------- live metrics ---------------- */}
        <div className="mt-3 rounded-xl border border-border bg-card/90 p-3 backdrop-blur-sm" data-testid="metrics">
          <div className="grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-5 xl:grid-cols-10">
            <Metric k="Current frame" v={frame ? String(frame.frame_index + 1) : "—"} />
            <Metric k="Total frames" v={inspection ? String(inspection.frame_count) : "—"} />
            <Metric k="Timestamp" v={frame ? fmtTime(frame.timestamp_seconds) : "—"} />
            <Metric k="Source FPS" v={inspection ? inspection.fps.toFixed(2) : "—"} />
            <Metric k="Processing FPS" v={m ? m.processing_fps.toFixed(1) : "—"} hint="1000 / rolling mean of per-frame compute time" />
            <Metric k="Playback FPS" v={m?.effective_fps != null ? m.effective_fps.toFixed(1) : "—"} hint="frames actually delivered per wall-clock second" />
            <Metric k="Processing time" v={m ? `${m.processing_time_ms.toFixed(0)} ms` : "—"} hint={m ? `IPM ${m.ipm_ms ?? "-"} · vector ${m.vectorize_ms ?? "-"} · render ${m.render_ms ?? "-"} · encode ${m.encode_ms ?? "-"} ms; decode ${m.decode_ms} ms` : undefined} />
            <Metric k="Progress" v={m ? `${(m.progress * 100).toFixed(1)} %` : "—"} />
            <Metric k="Calibration mode" v={cal ? (cal.status === "failed" ? "none (no BEV)" : "relative scale") : "—"} />
            <Metric k="Calibration status" v={cal ? cal.status.replace("_", " ") : state === "calibrating" ? "running" : "—"} />
          </div>
          <div className="mt-3 flex items-center gap-3">
            <div className="h-1.5 flex-1 overflow-hidden rounded bg-secondary" role="progressbar" aria-valuenow={m ? Math.round(m.progress * 100) : 0} aria-valuemin={0} aria-valuemax={100}>
              <div className="h-full bg-sky-400 transition-[width] duration-100" style={{ width: `${(m?.progress ?? (state === "completed" ? 1 : 0)) * 100}%` }} />
            </div>
            <span className="text-[11px] text-muted-foreground">Session: <span className="text-foreground">{state ?? "no video"}</span></span>
          </div>
          {notKeepingUp && (
            <p className="mt-2 flex items-center gap-1.5 text-xs text-amber-300"><AlertTriangle className="size-3.5" />
              Not keeping up with real time: processing at {m!.processing_fps.toFixed(1)} fps vs a {m!.source_fps.toFixed(1)} fps source. Every frame is still processed, so playback runs slower than the video.
            </p>
          )}
        </div>

        {/* ---------------- three synchronized panels ---------------- */}
        <div className="mt-3">
          <ProcessingPanels frame={frame} cal={cal} processingStatus={state ?? "waiting"} />
        </div>

        <footer className="mt-5 pb-4 text-[11px] leading-5 text-muted-foreground">
          Frame-level local vectorized BEV from one monocular camera, assuming a flat road. Scale is relative (camera heights) because no physical reference is available;
          parameters marked <span className="text-amber-300">assumed</span> are not measured from the video. Detection by thresholding, morphology and Hough transforms cannot be
          relied on for every unmarked or unstructured road. Prototype for PSJ_01 — not an HD map.
        </footer>
      </div>
    </div>
  );
}

function Card({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("rounded-xl border border-border bg-card/90 p-4 backdrop-blur-sm", className)}>{children}</div>;
}

function Metric({ k, v, hint }: { k: string; v: string; hint?: string }) {
  return (
    <div className="min-w-0" title={hint}>
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{k}</div>
      <div className="truncate font-mono text-sm font-medium tabular text-foreground">{v}</div>
    </div>
  );
}

function Banner({ tone, children }: { tone: "red" | "amber"; children: React.ReactNode }) {
  return (
    <div role="alert" className={cn("mt-4 rounded-md border p-3 text-sm", tone === "red" ? "border-red-500/40 bg-red-500/10 text-red-100" : "border-amber-500/40 bg-amber-500/10 text-amber-100")}>
      {children}
    </div>
  );
}
