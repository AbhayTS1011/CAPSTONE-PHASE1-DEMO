# Testing report

Test date: 2026-11-05. Machine: Intel Core i5-1035G1 (4 cores / 8 threads, 1.0 GHz base), 7.8 GB RAM, Windows 11 Home. Software: Python 3.10.4, Node 24.11, the in-app Chromium browser pane. All numbers below are **measured**. Raw outputs are in `test_results/`.

## Test videos

There was no independent consumer-dashcam recording on this machine. The test inputs were therefore built from **nuScenes v1.0-mini CAM_FRONT** images using `tools/build_test_videos.py`:
- all keyframes and sweeps of each scene, about 12 Hz;
- 1600×900;
- H.264.

The dataset itself was only read.

| File | Scene | Location / condition | Frames | FPS |
|---|---|---|---|---|
| `scene-0796_…_234f.mp4` | 0796 | Singapore Queenstown, day | 234 | 11.89 |
| `scene-0061_…_224f.mp4` | 0061 | Singapore One-North, day, left turn, roadworks | 224 | 11.64 |
| `scene-0103_…_229f.mp4` | 0103 | Boston Seaport, day (**different vehicle and calibration**) | 229 | 11.75 |
| `scene-1077_…_240f.mp4` | 1077 | Singapore Holland Village, **night** | 240 | 11.98 |
| `github/input/PSJ_01_Singapore_Queenstown_Input.mp4` | 0796 | the existing demo input (2 Hz keyframes, repeated) | 201 | 10.0 |
| `long_queenstown_x5_1170f.mp4` | 0796 ×5 | long-run and memory test only | 1170 | 11.89 |
| `scene-0553_…_237f.mp4` | 0553 | Boston, waiting at an intersection (**vehicle almost stationary**), pedestrians on a crosswalk | 237 | 11.86 |
| `scene-0655_…_237f.mp4` | 0655 | Boston, street with a zebra crossing, parked cars | 237 | 11.92 |
| `scene-0757_…_237f.mp4` | 0757 | Boston, busy intersection, **stops then turns** | 237 | 11.89 |
| `scene-0916_…_240f.mp4` | 0916 | Singapore Queenstown, **parking lot**, few markings | 240 | 12.01 |
| `scene-1094_…_232f.mp4` | 1094 | Singapore Holland Village, **night after rain** | 232 | 11.85 |
| `scene-1100_…_232f.mp4` | 1100 | Singapore Holland Village, **night, difficult lighting** | 232 | 11.85 |

The last six (added later) complete all 10 scenes of nuScenes v1.0-mini; their results are in section I.

The ground-truth calibration is written to `test_videos/ground_truth/`. Only `tools/evaluate_calibration.py` reads it, to score the estimate; the application never sees it.

No Queenstown map, pose or reference geometry is used anywhere. Every BEV and vector result comes from the video pixels.

---

## A. Upload and validation (through the browser UI)

| Test | Result |
|---|---|
| Drag and drop (`drop` event with a real `DataTransfer`) of an MP4 | ✅ uploaded, inspected, calibrated |
| Browse (`<input type=file>` change) | ✅ |
| Non-MP4 file (`notes.txt`) | ✅ rejected in the browser: *"is not an .mp4 file"* |
| Fake MP4 (11 bytes of text) | ✅ rejected by the backend: *"not an MP4/ISO-BMFF container"*, with a conversion hint |
| Truncated MP4 (first 3 MB of a clip) | ✅ rejected: *"no 'moov' (index) box; truncated or corrupted"* |
| Metadata shown: name, size, duration, resolution, FPS, frames, codec, camera tags | ✅ (codec h264, no camera tags in these files) |
| Remove video, and select another while one is active | ✅ the old session is stopped and its upload deleted; the new video is calibrated afresh |
| Size limit (client and server) | ⚠️ code path present; **not tested** with a file over 2 GB |
| Unsupported codec | ⚠️ handled by code (`cannot_open` / `cannot_decode`); **not tested** (no such file available) |
| Disk-full error | ⚠️ handled (`storage_error`, HTTP 507); **not tested** |

## B. Automatic calibration — accuracy against nuScenes ground truth

Source: `test_results/calibration_eval.json`. Calibration took 11–19 s per clip.

| Clip | Status | Road VP error (px, x / y) | Horizon error | **BEV shape error** after similarity alignment | 1 h equals | Non-pass checks |
|---|---|---|---|---|---|---|
| 0796 Queenstown | ok | +17.2 / −9.1 | −0.45° | **3.0 %** | 1.87 m (true h 1.51 m) | — |
| 0061 One-North | ok | −39.8 / −4.9 | −0.24° | **2.3 %** | 1.76 m (1.51 m) | VP dispersion 7.7°, stability 8.2° (warn — the left turn) |
| 0103 Boston | ok | +22.0 / −8.1 | −0.43° | **3.8 %** | 1.80 m (1.50 m) | — |
| 1077 Night | ok | +4.7 / −3.9 | −0.19° | **2.2 %** | 1.73 m (1.51 m) | ground-motion rigidity 0.92 (warn — noisy night tracks) |
| Existing demo input (2 Hz keyframes) | ok | (no ground-truth file) | | | | — |

**How the BEV shape error is computed.** A 13×7 ground grid (6–30 m ahead, ±6 m) is projected with the *true* camera. It is then mapped back with the *estimated* camera, and the best similarity transform (rotation, translation, scale) is fitted. The RMS residual is reported as a percentage of the 24 m grid length. This is the scale-free error the BEV actually suffers.

**Parameter-level results:**
- **Focal length: assumed on every clip** (70° prior gives 1142.5 px; truth 1266.4 or 1252.8 px, i.e. −9.8 % / −8.8 %).
  - Orthogonal vanishing points were *not usable*: these cameras are almost level, so the vertical VP is near infinity.
  - Turning was below the 3°-per-pair threshold, so rigidity could not constrain it either.
  - An earlier, looser version accepted an orthogonal-VP focal of +56 % on Boston. The conditioning gates were added because of that.
- **Roll: assumed 0 on every clip** after the gates (truth 0.05° / −0.71°). Vertical-structure roll was biased by 1.4–2.7° and is therefore reported only as a diagnostic.
- **Radial distortion:** "not detected" on all clips (k1 = 0). This is consistent with nuScenes images, which have negligible distortion.
- **Pitch angle vs truth:** −0.6° to −1.9°. This apparent error is almost entirely the **assumed principal point**: the truth is (816, 491), the assumption is (800, 450), and a 41 px row offset equals 1.85°. The detected horizon *row* is within 0.19–0.45°, as the table shows.
- **Yaw:** within 1.1–2.5° of the vehicle axis. Part of that is the road direction differing from the ego axis.

**Repeatability** (`test_results/calibration_repeatability.json`): four RANSAC seeds per clip gave a pitch range of 0.025° / 0.002°, a yaw range of ≤ 0.003°, and identical focal length and roll.

**Stability across sampled frames:** first-half vs second-half VP difference 0.19–0.86° on the straight clips, 8.2° on the turning clip (flagged *warn*).

**Fallback behaviour:**
- **Relative-scale mode:** used on every clip, with the message *"Automatic calibration completed — relative scale only. Absolute road distances could not be established."*
- **Failed mode (no BEV, original frames only, warning shown):** implemented and exercised through the code path. However, **no test clip actually failed calibration**, so the failed-mode UI was not observed with a real video.

## C. IPM

- The BEV is generated from each current frame with the estimated calibration. It is 480×720 px, the ROI is about 18 h ahead and ±6 h wide, and 73.6 % of the ROI is visible on these clips.
- Unobservable ground is hatched and never filled.
- Lane lines come out parallel and evenly spaced on straight roads, which visually confirms the horizon estimate. Raised objects smear, as expected for a flat-road IPM.
- No map geometry is used. The panel is labelled *RELATIVE SCALE · grid in camera heights (h)*.

## D. Vectorization

- Vectors are recomputed per frame. Their counts and shapes change with the content: for example 3 candidates on One-North frame 60 versus 9 on frame 150.
- Frames with at least one candidate:

  | Clip | Frames with candidates |
  |---|---|
  | 0061 | 216 / 224 |
  | 0103 | 216 / 229 |
  | 0796 | 234 / 234 |
  | 1077 | 240 / 240 |
  | Demo input | 201 / 201 |

  Frames without candidates show *"No reliable vector candidates detected in this frame."*
- Coordinates are reported in BEV ground units (h, X forward, Y left).
- **Observed failure modes** (visual inspection of the dumped panels):
  - false fragments on a white van directly ahead and on traffic cones (One-North frame 150);
  - kerb stripes reported as lane markings;
  - a washed-out yellow centre line missed (Boston, last frame);
  - Queenstown road arrows split into short pieces.

  The radial-smear test and the per-frame road-surface test were added to reduce false positives from vehicles and buildings. They reduce them but do not eliminate them.

## E. Synchronization and controls (browser UI, real WebSocket)

| Test | Result |
|---|---|
| All three panel headers sampled every 137–250 ms during runs (25 + 400 + 100 samples on three videos) | ✅ **0 desynchronized samples**; the image `alt` frame numbers matched too |
| Backend: every frame delivered exactly once, in order, all three panels present | ✅ on all 7 runs (201–1170 frames) |
| Pause → no frames for 2.5 s; resume continues from the next frame | ✅ (UI: paused at #20, resumed and reached #31; backend test: 0 frames received during the pause) |
| Stop → state *Stopped*, summary of frames processed | ✅ (31 / 234) |
| Reset → panels cleared, state *Ready to process* | ✅ |
| Re-run after *Completed* | ✅ |
| Liquid Metal Button states: disabled, ready, loading (calibrating / starting), processing, paused (→ RESUME), completed, error | ✅ rendered on `#/components`; ready, loading, processing, paused and completed also observed in the live flow |
| WebSocket loss → server pauses, client reconnects up to 6 times | ⚠️ implemented; **not tested** by forcibly breaking the connection |

## F. Performance (measured)

Backend-only runs (`tools/smoke_test_backend.py --no-pace`, 1600×900 input):

| Clip | Frames | Mean ms/frame | p95 ms | Compute FPS | Delivered FPS | Max RSS |
|---|---|---|---|---|---|---|
| 0061 | 224 | 59.1 | 72.3 | 16.9 | 13.9 | 164 MB |
| 0103 | 229 | 55.8 | 66.8 | 17.9 | 14.4 | 151 MB |
| 0796 | 234 | 75.4 | 103.2 | 13.3 | 10.8 | 138 MB |
| 1077 | 240 | 55.7 | 72.3 | 18.0 | 14.0 | 144 MB |
| Demo input | 201 | 64.0 | 73.0 | 15.6 | 12.7 | 149 MB |
| Long ×5 | 1170 | 123.1 | 165.5 | 8.1 | 6.7 | **134 MB** (flat — no growth) |
| 0796 paced + pause | 234 | 73.2 | 88.3 | 13.7 | 11.2 (source 11.89) | 146 MB |

With the dashboard open in the browser on the same laptop:

| Clip | Frames | Wall time | Mean ms/frame | p95 ms | Compute FPS | Delivered FPS |
|---|---|---|---|---|---|---|
| Queenstown (dev build) | 234 | 27.9 s | 95.5 | 132.8 | 10.5 | 8.4 |
| One-North | 224 | 23.5 s | 85.1 | 120.7 | 11.8 | 9.5 |
| Boston (production build) | 229 | 25.1 s | 90.6 | — | 11.0 | 9.1 |

The UI correctly showed *"Not keeping up with real time"* during these runs.

**Conclusions:**
- The prototype does **not** reach real time for 12 fps 1600×900 video on this laptop while the browser renders. Every frame is still processed.
- Run-to-run variation is large (the long run averaged 123 ms per frame versus 56–75 ms on the short runs), because of other load and CPU boost/thermal behaviour on this laptop. These are measurements on one machine, not a performance guarantee.
- Per-frame cost is mostly vectorization, rendering and JPEG encoding of three panels. The IPM remap costs about 3–5 ms.

## G. Generalization

- A fresh automatic calibration runs for every uploaded video. Nothing is cached between videos.
- Two nuScenes vehicles with different calibrations (Singapore, Boston), day and night, and a turning scene were tested.
- **Generalization across different cameras has NOT been independently validated.** Every test clip comes from the same nuScenes camera model with near-level mounting. A consumer dashcam (wide-angle, strong distortion, bonnet in view, burned-in timestamp) would exercise the distortion, static-mask and focal-length paths, which were not exercised here.

## H. Application

| Check | Result |
|---|---|
| `setup.bat` (venv, pip, `npm ci`, build) via `cmd.exe` | ✅ exit code 0 |
| `start_app.bat` starts the backend window, opens http://127.0.0.1:8000, health 200 | ✅ |
| `start_frontend.bat` serves http://127.0.0.1:5173, and `/api` is proxied to the backend | ✅ |
| `start_backend.bat` | ✅ (used by `start_app.bat`) |
| TypeScript check (`tsc -b`) | ✅ no errors |
| Production build (`npm run build`) | ✅ (warning only: 562 kB JS chunk > 500 kB) |
| Browser console during a full production run | ✅ no messages / errors |
| Backend errors during the test runs | ✅ none apart from OpenCV's harmless "OpenH264 library" notices, printed while *writing* test videos |
| ESLint / formatter | ⚠️ not configured in this new project, so not run |

## I. Second test round — the remaining six nuScenes mini scenes

Sources: `test_results/calibration_eval_new6.json` and `test_results/run_unpaced_scene-0553…1100*.json`. Panels were inspected visually on one frame per clip.

### Automatic calibration

| Clip | Status | Road VP error (px, x / y) | Horizon error | BEV shape error | 1 h equals | Checks not passed |
|---|---|---|---|---|---|---|
| 0553 intersection, stationary | ok | −29.5 / −10.2 | −0.53° | 4.0 % | 1.85 m | FOE vs VP and ground rigidity **not measurable**: only 1 frame pair had motion (warn) |
| 0655 street, crossing | ok | +4.6 / −12.4 | −0.63° | 3.7 % | 1.92 m | — (roll *estimated* here) |
| 0757 intersection, turns | **ok (wrong)** | **−227.5 / −30.8** | −1.56° | **6.7 %** | 2.39 m | temporal stability 3.3°, **FOE vs VP 15.5°** (both warn) |
| 0916 parking lot | low_confidence | +53.2 / −4.9 | −0.24° | 2.3 % | 1.76 m | VP detection rate 0.35, dispersion 3.3°, stability 8.6° (warn) |
| 1094 night after rain | ok | +4.9 / −8.5 | −0.42° | 2.9 % | 1.85 m | — |
| 1100 night, difficult | **failed** | — | — | — | — | Road VP found in only 3 / 38 sampled frames |

Findings:
- **The failed-calibration path works with a real video (1100).** The backend processed all 232 frames as original frames only, with no BEV and no vectors, at 4 ms per frame. Only the backend was run, so the failure messages were not observed in the browser UI.
- **The low-confidence path works (0916).** The parking lot has few road lines, so the vanishing point was found in only 35 % of sampled frames.
- **Status logic is too lenient (0757).** Yaw is about 10° wrong (227 px), because building edges along the cross street dominate while the car waits and then turns. The FOE-vs-VP check measured a 15.5° disagreement and flagged *warn*, but two warnings are below the three needed for `low_confidence`, so the UI reports **ok**. The BEV is visibly rotated.
- **Stationary vehicle (0553).** With almost no motion, the flow-based pitch refinement and the rigidity validation cannot run. The result is still reasonable (horizon −0.53°) because the vanishing point alone is used, but it is less validated than the status suggests.

### Processing (backend only, unpaced)

| Clip | Frames delivered, in order | Three panels in every frame | Frames with vector candidates | Mean / p95 ms | Max RSS |
|---|---|---|---|---|---|
| 0553 | 237 / 237 ✅ | ✅ | 233 | 95.1 / 125.7 | 166 MB |
| 0655 | 237 / 237 ✅ | ✅ | 237 | 95.8 / 134.2 | 149 MB |
| 0757 | 237 / 237 ✅ | ✅ | 237 | 89.9 / 129.7 | 148 MB |
| 0916 | 240 / 240 ✅ | ✅ | 228 | 92.7 / 134.2 | 143 MB |
| 1094 | 232 / 232 ✅ | ✅ | 212 | 99.7 / 134.5 | 140 MB |
| 1100 (calibration failed) | 232 / 232 ✅ | original only (by design) | 0 | 4.0 / 6.7 | 132 MB |

These per-frame times are about 1.5× those of the first round (56–75 ms) **with the same code**. This laptop's throughput varies between sessions, for example because of thermal state and background load.

### Visual inspection of the panels

- **1094, night after rain:** clean BEV; lane lines and a yellow line detected. This is the best result of the round.
- **0655:** BEV geometry correct (crossing stripes parallel, lane lines straight), but the **zebra crossing is not vectorized** and the faint yellow double centre line is missed.
- **0757:** BEV rotated (the yaw error above); the crossing is not vectorized.
- **0553:** BEV correct, with pedestrians smeared as expected. **Rendering bug:** the cyan BEV-region outline on panel 1 is drawn with spurious lines crossing the image.
- **0916:** BEV correct; only a single marking line is found, which is plausible on an unmarked parking lane.

**New limitation found: zebra crossings are never detected.** The stripes are about 0.3–0.4 h wide, wider than the 0.20 h "thin paint" limit and the 0.30 h top-hat kernel, so they are filtered out.

## Known issues

- Focal length, principal point and roll are assumed on typical forward-looking footage, which gives about 2–4 % BEV shape error on these clips.
- The relative unit h is distorted by the assumed focal length (1 h ≈ 1.73–1.87 m here, versus a 1.50 m camera height).
- Traditional-CV vectors include false positives on vehicles, cones and kerbs, and miss faded or yellow paint in some frames.
- Not real time on this laptop with the UI open.
- One calibration per video: pitch changes from bumps or braking are not followed.
- Second round (section I):
  - Calibration status can say `ok` when the FOE-vs-VP check shows a large disagreement (scene 0757, about 10° yaw error).
  - Zebra crossings are not vectorized.
  - The ROI outline on panel 1 is drawn incorrectly on scene 0553.
