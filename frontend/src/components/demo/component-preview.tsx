import { useState } from "react";
import { SparklesPreview, SparklesPreviewColorful, SparklesPreviewDark } from "@/components/demo/sparkles-demo";
import { LiquidMetalButton, type LiquidMetalState } from "@/components/ui/liquid-metal-button";
import FileUpload05 from "@/components/ui/file-upload-1";

// Development preview of the integrated UI components (open /#/components). Not used by the dashboard flow.
const STATES: LiquidMetalState[] = ["disabled", "ready", "loading", "processing", "paused", "completed", "error"];

export default function ComponentPreview() {
  const [clicks, setClicks] = useState(0);
  return (
    <div className="mx-auto max-w-5xl space-y-8 p-6">
      <h1 className="text-lg font-semibold">Component preview</h1>
      <section className="space-y-3">
        <h2 className="text-sm text-muted-foreground">LiquidMetalButton — all states (clicks: {clicks})</h2>
        <div className="flex flex-wrap items-center gap-6">
          {STATES.map((s) => (
            <div key={s} className="flex flex-col items-center gap-2">
              <LiquidMetalButton label="START PROCESSING" state={s} width={230} onClick={() => setClicks((c) => c + 1)} />
              <span className="text-xs text-muted-foreground">{s}</span>
            </div>
          ))}
          <LiquidMetalButton viewMode="icon" />
        </div>
      </section>
      <section className="max-w-md">
        <FileUpload05 file={null} phase="idle" progress={0} localMeta={null} inspection={null} error={null} maxSizeMb={2048} onSelect={() => {}} onRemove={() => {}} />
      </section>
      <SparklesPreview />
      <SparklesPreviewDark />
      <SparklesPreviewColorful />
    </div>
  );
}
