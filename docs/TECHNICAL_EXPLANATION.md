# Technical explanation

Conventions:
- **Camera frame:** x right, y down, z forward (OpenCV).
- **Ground frame:** X forward, Y left, Z up. The origin is on the road directly below the camera.
- **Unit of length:** the camera height, written **h**. Absolute scale is unknown.

Each section names the code it describes.

Every quantity in this document is one of three kinds:
- **measured:** computed directly from the pixels;
- **estimated:** fitted from measurements under a stated model;
- **assumed:** a value the video cannot provide.

---

## 1. How the MP4 is decoded (`video_io.py`)

- **Upload.** The file is streamed to `backend/uploads/<session>.mp4` in 4 MB chunks, and the size limit is enforced while writing.
- **Container check.** A small ISO-BMFF box parser:
  - requires `ftyp` at the start and a `moov` index;
  - reads `mvhd` (duration), `stsd` (sample entry such as `avc1`/`hvc1`) and `hdlr`;
  - reads iTunes-style tags (`©mak`, `©mod`, `©xyz`, `©swr`…).
- **Decode check.** OpenCV's FFmpeg backend opens the stream, decodes the first frames, and seeks near the end to confirm that the stream decodes there too.
- **Errors.** Each failure gets a specific code and an actionable hint: `not_mp4`, `corrupted`, `cannot_open`, `cannot_decode`, `bad_fps`, `too_small`, `too_short`.
- **Metadata.** Camera intrinsics and extrinsics are **not** part of standard MP4 metadata. If camera tags exist they are reported, but they are never used to look up parameters, because no trustworthy database is available offline.
- **Processing.** `Mp4FrameSource.frames()` decodes **sequentially** with `cv2.VideoCapture.read()` and yields `(frame_index, timestamp, frame)`.
  - It never holds more than the current frame.
  - The timestamp is `CAP_PROP_POS_MSEC` when the container provides it, otherwise `index / fps`.

## 2. Sequential frame processing (`sessions.py`, `pipeline.py`)

A worker thread runs this loop: decode the next frame → `FramePipeline.process(frame)` → build one message → `put` it into a **bounded** asyncio queue (size 3).

- **Backpressure, not frame dropping.** When the browser is slower, `put` blocks the decoder, so every frame is processed and delivered in order.
- **Pacing (optional).** The worker sleeps until `t0 + (i − i0) / fps`. This makes the clip behave like a live camera. If processing is slower, it does not sleep and the UI reports that it is not keeping up.
- **Pause** uses a `threading.Event`, and the pacing clock restarts on resume. **Stop** sets a flag. **Reset** additionally increments the run id, so any queued frames of the old run are discarded.
- **Disconnects.** If the WebSocket disconnects mid-run, the session is paused, so frames do not pile up unseen.

## 3. Automatic estimation of camera parameters (`calibration.py`)

The estimate runs **once per video**, before processing, on roughly 40 bursts of 6 consecutive frames spread over the clip. The analysis resolution is 960 px wide; results are scaled back to full resolution.

| Parameter | How it is obtained | Kind |
|---|---|---|
| Principal point | Image centre | assumed (not observable without a calibration target) |
| Skew / pixel aspect | 0 / 1 | assumed |
| Radial distortion k1 | Plumb-line test. Long Canny edge chains are undistorted with candidate k1 ∈ [−0.30, 0.10]; chains that are straight for *some* k1 are kept; k1 is accepted only if at least 12 straight chains exist, the median straightness residual improves by ≥ 30 % over k1 = 0, and the optimum is not at the grid boundary. The radial model is never evaluated beyond 1.2 × the image-corner radius, where a polynomial folds far-away points back into the image. | estimated, or "not detected" (k1 = 0, assumed) |
| Focal length | (a) Orthogonality of the road-direction VP and the vertical VP: f² = −(v₁ − c)·(v₃ − c). Used only if it is well conditioned (vertical VP finite, lever arm ≥ 5 % of image height, IQR < 15 %). (b) BEV rigidity during turns (§5). Used only with ≥ 5 frame pairs turning > 3° and a sharp minimum. Otherwise, a prior of **70° HFOV**. | estimated, or assumed |
| Yaw | Robust median of the per-frame road-direction vanishing points | estimated |
| Pitch | Height of the road-direction VP, refined by ground-motion rigidity (§5) | estimated |
| Roll | Ground-motion rigidity, accepted only if the cost rises ≥ 20 % on **both** sides of the minimum at ±2°; otherwise 0 | estimated, or assumed |
| Camera height | Unknown → used as the unit of length | assumed (relative scale) |

**Why the vertical lines are not used for roll.** Vertical-VP directions are still computed and reported in the diagnostics. On the test clips they were biased by 1.4–2.7°, because tree trunks and poles are not truly vertical. The rigidity test does not have that bias.

## 4. Vanishing points and the horizon

- **Line segments.** LSD (`cv2.createLineSegmentDetector`) runs on each sampled grey frame. Segments on static pixels are removed.
- **Candidates for the road direction:**
  - midpoint in the lower 65 % of the image;
  - 12–80° from horizontal;
  - length ≥ 2.5 % of the image width.
- **RANSAC.** Hypotheses are intersections of two length-weighted random segments. The score is the length-weighted soft count of segments whose direction is within 2° of the line from their midpoint to the VP. The inliers are then refined by least squares (SVD on the normalised line equations).
- **Acceptance.** Inliers must come from **both** slope signs, with each side carrying ≥ 15 % of the support; this rejects a single bundle of parallel edges. The VP must lie inside a plausible image band.
- **Over time.** The robust median over frames, with a 3-MAD inlier gate, gives the camera's mean viewing direction relative to the road.
- **Motion consistency.** If at least 5 frames (and ≥ 25 %) have an optical-flow focus of expansion within 3° of their road VP, only those frames are used. The vehicle is then demonstrably travelling along the road, so frames dominated by cross-street lines while waiting or turning are dropped. The selection used is reported in the diagnostics.
- **Horizon.** Under the planar-ground model, the horizon is the vanishing line of the ground plane: l ∝ K⁻ᵀ n, where n is the ground normal. It passes through the road VP. Its tilt is the roll (assumed 0 unless measured).

## 5. Road-plane estimation and the rigidity test

**Model.** For a hypothesis (f, pitch, roll), ground point (X, Y) maps to the camera as P = X·fwd + Y·left − h·up, with h = 1. Pixels map back to the ground by intersecting the undistorted ray with the plane up·P = −h.

**Measurement:**
- Shi–Tomasi corners are taken in the road region: below the horizon, above the bonnet, and not on static pixels.
- They are tracked with pyramidal Lucas–Kanade, with a forward-backward check (< 0.8 px).
- For each burst, the frame gap is the first one with a median flow ≥ 6 px; repeated or static frames are skipped.

**Test:**
- Both ends of every track are mapped onto the hypothesised ground.
- Because the road is rigid, a correct plane makes the two point sets differ by a 2-D rotation plus translation.
- A RANSAC rigid fit, run once with the initial hypothesis, removes moving vehicles and off-ground points.
- The cost is the trimmed RMS residual divided by the mean displacement, with the median taken over frame pairs.

**What it constrains:**
- **Pitch is strongly constrained.** A wrong horizon makes near and far points move by different amounts. On the test clips, ±1° of pitch raised the cost by 25–90 %, and the minimum coincided with the ground truth when evaluated with the true camera (`tools/debug_rigidity.py`).
- **Roll is weakly constrained.**
- **Focal length is constrained only when the vehicle rotates.** For pure forward motion, any focal length gives rigid motion, so on straight driving it is not observable.

The search refines pitch on a grid of ±5° in 0.5° steps with a parabolic peak; roll uses a grid of ±4°.

## 6. Homography calculation and validation

- **Homography.** H_ground→image = K [fwd | left | −h·up] gives the undistorted image of the ground plane; the IPM additionally applies k1 through `cv2.projectPoints`.
- **Validation (all in `calibration.py`, shown in the UI):**
  - forward-VP detection rate ≥ 40 %;
  - VP dispersion < 3°;
  - first-half vs second-half stability < 2°;
  - optical-flow FOE vs road VP < 3°;
  - horizon inside −0.2…0.8 × image height (**fail**);
  - condition number of the normalised homography < 10⁴ and non-zero determinant (**fail**);
  - rigidity residual < 0.15;
  - visible ground ≥ 15 % of the ROI (**fail**).
- **Status:**
  - Any *fail* gives `failed`, which means no BEV.
  - Otherwise the result is `low_confidence` when any of these holds:
    - three or more *warn*;
    - an independent cross-check disagrees strongly (FOE vs VP > 6° or rigidity residual > 0.30);
    - neither cross-check could be measured because the vehicle hardly moves.
  - The reasons are listed as warnings in the UI. Otherwise the status is `ok`.

These are measured indicators, not a learned confidence score.

## 7. IPM generation (`ipm.py`)

The BEV grid is 480 × 720 px, with square cells:
- **Forward range:** 0 … x_max. Here x_max is the distance whose image row lies 7 % of the image height below the horizon, which avoids the extremely stretched rows near the horizon. On the test clips this was about 18 h.
- **Lateral range:** ±x_max/3.

For each BEV cell, the ground point is projected into the image once (including k1), producing `map_x`/`map_y` for `cv2.remap`. A cell is **valid** only if:
- its point is in front of the camera;
- it falls inside the image;
- it is above the bonnet row;
- it is not on a static-pixel region.

Invalid cells are hatched and never filled. Per frame, the cost is one remap, about 3–5 ms.

## 8. Road-feature detection (`vectorize.py`)

1. **Brightness.** L channel (Lab) white top-hat with a 0.30 h elliptical kernel, which keeps structures thinner than about 30 cm when h ≈ 1.5 m. The threshold is median + 5·MAD of the valid region, and at least 14.
2. **Yellow.** Lab b > 148, together with a weaker top-hat response.
3. **Clean-up.** Morphological opening (2×2), then closing (3×7, to join broken paint), then connected components.
4. **Shape tests per component (measured):**
   - length ≥ 0.45 h;
   - mean width ≤ 0.20 h;
   - PCA elongation ≥ 3.
5. **Radial-smear test.** A component that lies along a ray from the camera (its extension passes within 35 % of its lateral offset of the camera, and it is more than 0.8 h to the side) is the IPM smear of a vertical object, and is rejected.
6. **Road-surface context.**
   - The road appearance of *this frame* is the Lab median and MAD of the patch directly ahead (0–4 h, ±0.6 h).
   - A marking is kept only if ≥ 35 % of the ring around it is road-like. Paint sits on the road; edges of cars, buildings and vegetation usually do not.
7. **Road-edge candidates.**
   - Canny on the smoothed BEV, followed by probabilistic Hough.
   - Lines must be ≥ 2.5 h long, within 25° of the forward axis, and not radial.
   - They must not coincide with a marking.
   - Road-like pixels must be on the inner side and not on the outer side, with a lateral gradient ≥ 12.
   - At most one is kept per side: the outermost.

### Wide paint: pedestrian crossings and stop lines
Zebra stripes (about 0.4–0.6 m wide) and stop lines are wider than the thin-paint limit, so a second pass handles them:
1. A top-hat with a 1.0 h rectangular kernel runs at half BEV resolution.
2. **Directional openings** (0.5 h long along each BEV axis) split "ladder" crossings, whose stripes are joined by thin cross-bars, into individual stripes.
3. Each stripe must pass measured shape tests:
   - width 0.2–0.7 h;
   - length ≥ 0.6 h;
   - length/width ≥ 2;
   - fill of its minimum-area rectangle ≥ 0.6;
   - the road-surface context test.
4. Stripes that are parallel (within 15°), side by side (0.25–1.3 h apart across, overlapping along) are grouped with union-find. A group of **≥ 3** becomes a `pedestrian_crossing`, drawn as the minimum-area rectangle around the group. Thin-pass fragments inside a crossing are discarded.
5. A lone wide bar ≥ 1.2 h long and within 30° of transverse becomes a stop-line candidate (`transverse_marking`).

## 9. Producing the vector polylines

- **Centreline.** For each component, the pixels are projected onto the principal axis and binned every 0.4 h; the bin centroids give the centreline.
- **Merging dashes.** Longitudinal pieces are joined into one polyline when the gap is ≤ 4 h, the direction change is < 10°, and the lateral offset of the next piece from the extrapolated line is < 0.25 h. `parts_merged` records how many pieces were joined.
- **Simplification.** Douglas–Peucker with ε = 0.05 h.
- **Classification by direction:**
  - < 30° from forward: lane marking (white or yellow);
  - > 60°: transverse marking (stop line or crossing rail);
  - in between: oblique.
- **Coordinates.** Polylines are converted to ground coordinates in h (X forward, Y left) and sent as `vector_candidates`.
- **Score.** `cv_score` = min(1, length / 3h) × min(1, contrast / 2). It is a heuristic, not a probability. Candidates scoring < 0.2 are not reported.
- The per-frame path described here is used as evidence. The rendered output uses the tracked version (section 9a).

## 9a. Temporal tracking for smooth output (`vector_tracker.py`)

Per-frame polylines (section 9) flicker, and dashes stay fragmented. The rendered panel C therefore uses a short temporal fusion.

**1. Ego-motion.**
- Shi–Tomasi corners are taken on the image pixels that see valid ground, on a half-resolution grey image, and tracked with pyramidal Lucas–Kanade.
- Both ends of each track are mapped to the ground plane with the calibrated camera.
- A RANSAC 2-point rigid fit (200 hypotheses, vectorised) and a Kabsch refinement give g_now = R·g_prev + t.
- Implausible motions (> 15° or > 3 h per frame) are rejected.
- If the fit fails, the last good motion is reused for up to 3 frames; otherwise the memory restarts.

**2. Evidence fusion.**
- Four BEV maps are kept: accepted paint, yellow paint, strict road-surface likeness, and kerb-like edges. Kerb-like edges have a strong lateral gradient, a weak gradient along the road, and are away from paint.
- Each map is warped with cv2.warpAffine by the BEV-pixel form of (R, t), then blended: map = 0.8·warped + 0.2·current.
- Current evidence is zero outside the visible ground, so the unseen ground only decays and is never filled.

**3. Lane dividers.**
- A column histogram of fused paint (≥ 0.30) gives peaks; sliding windows (1 h × 0.7 h) follow each peak from near to far.
- Each line is fitted as col = quadratic(row), or linear when shorter than 5 h. If the residual is above 0.15 h, a smoothed polyline is used instead.
- Lines shorter than 2.4 h, or with paint on less than 25 % of their length, are rejected. This removes road arrows.
- **Solid vs dashed** is measured on the CURRENT frame's paint, over the nearer 60 % of the range, because fusion smears dashes into streaks.

**4. Road boundaries.**
- On each side, the outermost long solid line (≥ 4 h, ≥ 80 % painted now, more than 0.8 h from the centre line) is the road edge line.
- Otherwise, each BEV row is scanned outwards from the vehicle's lane until 0.3 h of non-road, or a kerb-like edge not near paint.
- The stop points are median-filtered, split at jumps, and fitted with smooth curves.
- Segments are dropped if they are shorter than 2.5 h or lie inside the vehicle's own lane.

**5. Consistency rules.**
- A dashed line is never a boundary: a scan boundary along it is dropped.
- Paint within 0.7 h of a boundary is the edge line itself, not a divider.
- Dividers must lie on the scanned drivable surface and not beyond a boundary.

**6. Line persistence (smooth frame-to-frame output).**
- Each drawn line is a *track* kept in ground coordinates, sampled as Y every 0.4 h of forward distance X.
- Every frame, each track is first moved by the measured ego-motion (its points are transformed by (R, t)), so a line stays put on the road as the vehicle advances.
- The frame's freshly extracted lines are matched to tracks of the same class by lateral distance (median gap below 0.9 h over their overlap). A match updates the track with an exponential blend (new weight 0.35), so the shape changes slowly and smoothly; newly revealed far/near ground is taken directly.
- An unmatched detection starts a new track; a track with no match fades and is dropped after 6 frames. A track is drawn only once confirmed over 2 frames, and only over ground the camera currently sees.
- The result is that lines glide with the vehicle and keep their identity between frames, instead of being redrawn from scratch each frame.

**Accuracy** is measured against the map-derived geometry of the earlier demo by `tools/benchmark_vectors.py` (evaluation only; the application never reads the map). The results are in the testing report.

## 10. Keeping the three panels synchronized

All three panels are rendered from the same decoded `frame` inside one `process()` call. They travel in **one** WebSocket message, with one `frame_index`, one `timestamp_seconds` and one `run_id`.

The React app keeps that message as a single state value, and all three `<img>` elements and frame labels read from it. It is therefore impossible for one panel to show frame 125 while another shows frame 126.

Frames from a stopped or reset run are filtered out by `run_id`, both on the server and before they are sent.

## 11. Processing metrics

| Metric | Definition |
|---|---|
| Processing time | Wall time of `FramePipeline.process` for this frame: IPM + vectorize + render + JPEG encode. Decode time is reported separately. |
| Processing FPS | 1000 / mean processing time of the last 30 frames (compute throughput) |
| Playback (effective) FPS | Frames actually handed to the stream per wall-clock second, over the last 60 frames. Pauses are excluded. |
| Source FPS | From the container |
| Keeping up | Processing FPS ≥ source FPS, and (when paced) effective FPS ≥ 95 % of source |
| Progress | (frame_index + 1) / frame count, i.e. frames actually processed |
| Memory | Backend process RSS (psutil) |
| Run summary | Frames processed, wall time excluding pauses, mean and p95 ms, compute and delivered FPS |

## 12. Assumptions, estimates and measurements — summary

| Kind | Items |
|---|---|
| **Assumed** | Flat road; one fixed camera pose for the whole video; principal point at the image centre; zero skew and square pixels; focal length = 70° HFOV prior unless measured; roll = 0 unless measured; camera height used as the unit (relative scale) |
| **Estimated** | Yaw and pitch (always, when calibration succeeds); k1, roll and focal length only when the measurements constrain them; ROI far limit (derived from the estimated horizon) |
| **Measured** | Resolution, FPS, frame count, codec; line segments; vanishing points; optical-flow tracks; FOE; static pixels; all quality-check values; per-frame marking contrast, length and width; timing and memory |
