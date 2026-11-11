import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { CheckCircle2, FileVideo, Loader2, Upload, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { fmtBytes, fmtTime, type ApiError, type Inspection } from "@/lib/api";

/**
 * Supplied file-upload component (originally a spreadsheet uploader), adapted into a functional MP4
 * drag-and-drop uploader: .mp4 only, drag & drop or browse, size limit, local preview metadata, real
 * upload progress, server-side validation result (fps, frame count, codec), remove/replace.
 */
export type UploadPhase = "idle" | "reading" | "uploading" | "validating" | "done" | "error";

export interface LocalMeta { duration?: number; width?: number; height?: number }

interface FileUploadProps {
  file: File | null;
  phase: UploadPhase;
  progress: number;
  localMeta: LocalMeta | null;
  inspection: Inspection | null;
  error: ApiError | null;
  maxSizeMb: number;
  disabled?: boolean;
  onSelect: (file: File) => void;
  onRemove: () => void;
}

export default function FileUpload05({
  file, phase, progress, localMeta, inspection, error, maxSizeMb, disabled, onSelect, onRemove,
}: FileUploadProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pick = (files: FileList | null) => {
    if (!files || files.length === 0 || disabled) return;
    onSelect(files[0]);
    if (inputRef.current) inputRef.current.value = "";
  };

  const status =
    phase === "reading" ? "Reading file…"
    : phase === "uploading" ? `Uploading ${Math.round(progress * 100)}%`
    : phase === "validating" ? "Validating on the local backend…"
    : phase === "done" ? "Validated"
    : phase === "error" ? "Rejected"
    : "";

  const res = inspection?.resolution ?? (localMeta?.width ? [localMeta.width, localMeta.height] : null);
  const duration = inspection?.duration_s ?? localMeta?.duration;

  return (
    <div className="w-full">
      <form onSubmit={(e) => e.preventDefault()}>
        <h3 className="text-sm font-semibold tracking-wide text-foreground">Video input</h3>
        <div
          onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); pick(e.dataTransfer.files); }}
          className={cn(
            "mt-3 flex justify-center rounded-md border border-dashed border-input px-6 transition-colors",
            file ? "py-3" : "py-8",
            dragging && "border-sky-400/70 bg-sky-400/5",
            disabled && "opacity-50",
          )}
          data-testid="dropzone"
        >
          <div className={cn("flex flex-col items-center text-center", file ? "gap-1" : "gap-3")}>
            {!file && <Upload className="h-7 w-7 text-muted-foreground" aria-hidden={true} />}
            <div className="text-sm leading-6 text-foreground">
              <p className="font-medium">Drag and drop your dashcam MP4 here</p>
              <Label
                htmlFor="video-upload"
                className={cn(
                  "relative cursor-pointer rounded-sm font-medium text-sky-300 hover:underline hover:underline-offset-4",
                  disabled && "pointer-events-none",
                )}
              >
                <span>or browse your computer</span>
                <input
                  ref={inputRef}
                  id="video-upload"
                  name="video-upload"
                  type="file"
                  accept="video/mp4,.mp4"
                  className="sr-only"
                  disabled={disabled}
                  onChange={(e) => pick(e.target.files)}
                />
              </Label>
            </div>
          </div>
        </div>
        <p className="mt-2 text-xs leading-5 text-muted-foreground">
          Max. size: {maxSizeMb >= 1024 ? `${(maxSizeMb / 1024).toFixed(0)} GB` : `${maxSizeMb} MB`} · Accepted: MP4 (H.264 / HEVC decodable by FFmpeg) · processed locally, never uploaded elsewhere
        </p>

        {file && (
          <div className="relative mt-4 rounded-lg bg-muted p-3" data-testid="file-card">
            <div className="absolute right-1 top-1">
              <Button
                type="button"
                variant="ghost"
                size="sm"
                className="rounded-sm p-2 text-muted-foreground hover:text-foreground"
                aria-label="Remove video"
                onClick={onRemove}
                disabled={phase === "reading"}
              >
                <X className="size-4 shrink-0" aria-hidden={true} />
              </Button>
            </div>
            <div className="flex items-center space-x-2.5 pr-8">
              <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-sm bg-background shadow-sm ring-1 ring-inset ring-input">
                <FileVideo className="size-5 text-foreground" aria-hidden={true} />
              </span>
              <div className="w-full min-w-0">
                <p className="truncate text-xs font-medium text-foreground" title={file.name}>{file.name}</p>
                <p className="mt-0.5 flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  <span>{fmtBytes(file.size)}</span>
                  <span className={cn("flex items-center gap-1", phase === "error" && "text-red-400", phase === "done" && "text-emerald-400")}>
                    {(phase === "uploading" || phase === "validating" || phase === "reading") && <Loader2 className="size-3 animate-spin" />}
                    {phase === "done" && <CheckCircle2 className="size-3" />}
                    {status}
                  </span>
                </p>
              </div>
            </div>
            {(phase === "uploading" || phase === "validating") && (
              <div className="mt-3 h-1 w-full overflow-hidden rounded bg-background" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-valuemin={0} aria-valuemax={100}>
                <div className={cn("h-full bg-sky-400 transition-[width]", phase === "validating" && "animate-pulse")} style={{ width: `${phase === "validating" ? 100 : progress * 100}%` }} />
              </div>
            )}
            <dl className="mt-3 grid grid-cols-3 gap-x-3 gap-y-2 text-xs tabular">
              <Meta k="Duration" v={duration != null ? fmtTime(duration) : "—"} />
              <Meta k="Resolution" v={res ? `${res[0]}×${res[1]}` : "—"} />
              <Meta k="Source FPS" v={inspection ? inspection.fps.toFixed(2) : phase === "error" ? "—" : "after validation"} />
              <Meta k="Frames" v={inspection ? String(inspection.frame_count) : "—"} />
              <Meta k="Codec" v={inspection ? inspection.codec_fourcc || inspection.sample_entries.join(",") : "—"} />
              <Meta k="Camera metadata" v={inspection ? (Object.keys(inspection.camera_related_tags).length ? "tags found" : "none") : "—"} />
            </dl>
          </div>
        )}
        {error && (
          <div className="mt-3 rounded-md border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-200" role="alert">
            <p className="font-medium">{error.message}</p>
            {error.hint && <p className="mt-1 text-red-200/80">{error.hint}</p>}
          </div>
        )}
      </form>
    </div>
  );
}

function Meta({ k, v }: { k: string; v: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">{k}</dt>
      <dd className="truncate font-medium text-foreground">{v}</dd>
    </div>
  );
}
