// Types and transport for the local FastAPI backend (same origin; Vite proxies /api and /ws in development).

export interface ApiError { code: string; message: string; hint?: string }

export interface Inspection {
  filename?: string;
  resolution: [number, number];
  fps: number;
  frame_count: number;
  duration_s: number;
  codec_fourcc: string;
  container_brand?: string;
  sample_entries: string[];
  metadata_tags: Record<string, string>;
  camera_related_tags: Record<string, string>;
  camera_intrinsics_in_metadata: boolean;
  metadata_note: string;
  file_size_bytes: number;
}

export interface CalParam { name: string; value: unknown; unit: string; source: "estimated" | "assumed" | "measured"; method: string; extra?: string }
export interface CalCheck { name: string; value: number | null; status: "pass" | "warn" | "fail"; detail: string }

export interface Calibration {
  status: "ok" | "low_confidence" | "failed";
  scale_mode: "relative" | "metric" | "none";
  scale_note?: string;
  reason?: string;
  method?: string;
  parameters: CalParam[];
  checks: CalCheck[];
  warnings: string[];
  roi_units_h?: { x_min: number; x_max: number; y_max: number };
  horizon_row_px?: number;
  hood_top_row_px?: number | null;
  bev_valid_fraction?: number;
  diagnostics?: Record<string, unknown>;
  preview?: { frame_index: number; original: string; bev: string } | null;
  elapsed_s: number;
}

export interface VectorCandidate {
  type: "road_boundary" | "lane_divider" | "lane_divider_yellow" | "pedestrian_crossing" | "lane_marking" | "lane_marking_yellow" | "transverse_marking" | "oblique_marking" | "road_edge_candidate";
  points_h: [number, number][];
  length_h: number;
  parts_merged: number;
  cv_score: number;
  contrast_ratio: number;
  stripes?: number;
}

export interface FrameMetrics {
  processing_time_ms: number;
  decode_ms: number;
  ipm_ms?: number;
  vectorize_ms?: number;
  render_ms?: number;
  encode_ms?: number;
  processing_fps: number;
  effective_fps: number | null;
  source_fps: number;
  keeping_up_with_source: boolean | null;
  paced_to_source: boolean;
  progress: number;
  memory_rss_mb: number;
  queue_depth: number;
}

export interface FrameUpdate {
  type: "frame_update";
  session_id: string;
  run_id: number;
  frame_index: number;
  timestamp_seconds: number;
  total_frames: number;
  original_frame: string;
  bev_frame: string | null;
  vectorized_bev_frame: string | null;
  vector_candidates: VectorCandidate[];
  vector_info: Record<string, unknown>;
  metrics: FrameMetrics;
}

export interface RunSummary {
  frames_processed: number; total_frames: number; wall_time_s: number; mean_processing_ms: number | null;
  p95_processing_ms: number | null; mean_processing_fps: number | null; effective_fps: number | null;
  frames_with_vector_candidates: number; paced_to_source: boolean;
  demo_video?: string | null; demo_video_frames?: number;
}

export type SessionState = "uploaded" | "inspecting" | "calibrating" | "ready" | "processing" | "paused" | "completed" | "stopped" | "error";

export type ServerMessage =
  | { type: "session_state"; state: SessionState; inspection: Inspection | null; calibration_stages: Stage[]; calibration: Calibration | null; error: ApiError | null; summary: RunSummary | null; pace: boolean }
  | { type: "status"; state: SessionState; error?: ApiError; summary?: RunSummary }
  | ({ type: "calibration_stage" } & Stage)
  | { type: "calibration_result"; calibration: Calibration }
  | FrameUpdate
  | { type: "notice"; message: string }
  | { type: "error"; error: ApiError };

export interface Stage { stage: string; detail: string; t: number }

export async function getHealth(): Promise<{ max_upload_mb: number }> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`backend responded ${r.status}`);
  return r.json();
}

/** Upload with real progress events (XMLHttpRequest exposes upload progress; fetch does not). */
export function uploadVideo(file: File, onProgress: (fraction: number) => void): { promise: Promise<{ session_id: string; inspection: Inspection }>; abort: () => void } {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<{ session_id: string; inspection: Inspection }>((resolve, reject) => {
    xhr.open("POST", "/api/videos");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.onload = () => {
      let body: any = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
      if (xhr.status === 200 && body) resolve(body);
      else reject(body?.error ?? { code: `http_${xhr.status}`, message: `Upload failed (HTTP ${xhr.status}).` });
    };
    xhr.onerror = () => reject({ code: "network", message: "Could not reach the local backend.", hint: "Is start_backend.bat running on http://127.0.0.1:8000?" });
    xhr.onabort = () => reject({ code: "aborted", message: "Upload cancelled." });
    const fd = new FormData();
    fd.append("file", file);
    xhr.send(fd);
  });
  return { promise, abort: () => xhr.abort() };
}

export async function deleteVideo(sessionId: string) {
  try { await fetch(`/api/videos/${sessionId}`, { method: "DELETE" }); } catch { /* backend gone: nothing to clean */ }
}

export function wsUrl(sessionId: string) {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/ws/${sessionId}`;
}

export const fmtBytes = (b: number) =>
  b >= 1e9 ? `${(b / 1e9).toFixed(2)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} kB`;
export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, "0")}`;
};
