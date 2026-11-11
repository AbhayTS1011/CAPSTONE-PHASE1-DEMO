# Dashcam → Automatic Calibration → IPM → Vectorized BEV (interactive prototype)

**PSJ_01 — Edge-Deployed Monocular Vectorized HD Map Generation for Reliable Mapping on Unstructured Roads**

This is a local, browser-based prototype. You drag one dashcam `.mp4` into the page. The program then works out the camera orientation and the road plane from the footage itself; nothing has to be typed in. Press **START PROCESSING** and it plays the video frame by frame as a *simulated live camera feed*. For every frame it shows three panels side by side, all from that same frame:

| Panel | What it shows |
|---|---|
| **01 — Original dashcam feed** | The decoded frame, with the estimated horizon and the outline of the BEV region drawn on it |
| **02 — Inverse perspective mapping** | The same frame warped onto the estimated ground plane |
| **03 — Vectorized road representation** | Polylines for lane markings and road-edge candidates, found in that frame's BEV with traditional computer vision |

Constraints the prototype keeps:
- **Traditional computer vision only.** No neural networks, pretrained models or downloaded weights, and no external APIs.
- **No map data.** It never uses nuScenes maps, ego poses or any precomputed geometry.
- **All processing stays on your computer.**

> **What this is not.** This is a local, vectorized BEV of what the camera currently sees. It is *not* an HD map. Evidence is fused over only about 5 frames, using measured ego-motion, to make the lines smooth and stable; there is no persistent world model and no map data.
>
> **Scale is relative.** Distances are in units of camera height (h), because a single camera cannot measure metres without a physical reference.

---

## 1. Quick start (Windows, Command Prompt)

Prerequisites:
- **Python 3.10, 3.11 or 3.12.** Check with `py -0`.
- **Node.js 20+ with npm.** Check with `node -v`.

FFmpeg is **not** required. OpenCV's wheel includes its own FFmpeg decoder.

```bat
cd C:\Abhay\PESU\CAPSTONE\PROJECT\interactive_prototype
setup.bat
start_app.bat
```

`setup.bat` is needed once. It:
1. creates `backend\.venv`;
2. installs `backend\requirements.txt`;
3. runs `npm ci`;
4. builds the frontend.

`start_app.bat` opens a "PSJ_01 backend" window and your browser at **http://127.0.0.1:8000**.

Then:
1. Drag and drop a dashcam MP4 onto **"Drag and drop your dashcam MP4 here"**, or click *browse*.
2. Wait while the program inspects and calibrates the video. The stages tick off in the *Automatic calibration* card; this took 11–19 s on the test clips.
3. Click **START PROCESSING**.
4. Watch the three synchronized panels. You can use **Pause / Resume / Stop / Reset**.
   Tick **Record demo video** before starting to get a downloadable 1920×1080 MP4 in the same layout as the PSJ_01 final-demo video. The **Download demo video** button appears when the run ends.
5. To stop the application, close the "PSJ_01 backend" window.

| Script | Purpose |
|---|---|
| `setup.bat` | One-time install and production build |
| `start_app.bat` | Starts the backend in its own window and opens the browser |
| `start_backend.bat` | Backend only, on http://127.0.0.1:8000 (it also serves the built UI) |
| `start_frontend.bat` | Development only: the Vite dev server with hot reload on **http://127.0.0.1:5173**. It proxies `/api` and `/ws` to the backend, so run `start_backend.bat` as well. |

### Run the development servers separately

After running `setup.bat`, open two Command Prompt windows from the project folder:

```bat
start_backend.bat
```

```bat
start_frontend.bat
```

Open **http://127.0.0.1:5173** for the Vite development build. The production build served by FastAPI is available at **http://127.0.0.1:8000**. The development frontend requires the backend to be running because Vite forwards API and WebSocket requests to port 8000.

### Manual setup (Windows Command Prompt)

Use this if you prefer not to run the setup script. Run each block from the project root:

```bat
py -3.10 -m venv backend\.venv
backend\.venv\Scripts\python -m pip install --upgrade pip
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt
cd frontend
npm ci
npm run build
cd ..
start_app.bat
```

The backend supports Python 3.10–3.12. If `py -3.10` is unavailable, use an installed supported version, for example `py -3.11` or `py -3.12`, in the first command. Node.js 20 or newer and npm are required. The supplied launch scripts target Windows; on another operating system, create the virtual environment, install the Python requirements and frontend dependencies with the equivalent commands, then launch Uvicorn and Vite as described below.

### Manual backend and frontend commands

From the project root, start the backend in one terminal:

```bash
cd backend
python -m venv .venv
# Windows PowerShell: .venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux:        .venv/bin/python -m pip install -r requirements.txt
# Windows PowerShell: .venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# macOS/Linux:        .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For the built-in production UI, first build the frontend in a second terminal:

```bash
cd frontend
npm ci
npm run build
```

For frontend development with hot reload, run `npm run dev` from `frontend/` instead of opening the production UI. Keep the backend running in the first terminal and browse to **http://127.0.0.1:5173**.

Manual equivalent (CMD):
```bat
cd interactive_prototype\backend
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
```bat
cd interactive_prototype\frontend
npm ci
npm run build
```

Settings (environment variables, set before `start_backend.bat`):
- `set PSJ_MAX_UPLOAD_MB=4096` changes the upload limit. The default is 2048 MB.
- `set PSJ_UPLOAD_DIR=D:\tmp\psj` changes the temporary upload folder. It is emptied at start-up, and a video is deleted when it is removed or replaced.

The isolated UI component preview is at **http://127.0.0.1:8000/#/components**.

---

## 2. Architecture

```
Browser (React + TypeScript + Tailwind, shadcn structure)
  ├─ MP4 drag & drop (XHR upload with real progress)   POST /api/videos   ─┐
  ├─ calibration stages / results / frames  ◄── WebSocket /ws/{session} ──┤
  └─ commands start / pause / resume / stop / reset ──► (same socket)      │
                                                                           ▼
FastAPI backend (Python, OpenCV, NumPy)                                    
  video_io.py      MP4 box parser, validation, metadata, FrameSource (sequential decoder)
  calibration.py   automatic camera + road-plane estimation, quality checks (runs once per video)
  camera_model.py  pinhole + flat ground in camera-height units
  ipm.py           precomputed remap tables -> per-frame BEV, visibility mask
  vectorize.py     per-frame paint / road-surface / crossing evidence
  vector_tracker.py ego-motion, evidence fusion, smooth divider + boundary curves
  demo_video.py    optional 1920x1080 demo MP4 (final-demo layout)
  render.py        the three panels (figure style of the final-demo video)
  pipeline.py      FramePipeline.process(frame): input-agnostic per-frame processing
  sessions.py      worker threads, bounded queue (backpressure), pacing, metrics
  main.py          REST + WebSocket endpoints; serves frontend/dist
```

`FramePipeline` only needs `process(frame)`. `Mp4FrameSource` implements the small `FrameSource` protocol (`frames()` and `close()`). A future webcam or RTSP source only needs that same protocol (for example `cv2.VideoCapture(0)` or an RTSP URL), so the computer-vision code would not change.

### Technology stack
| Layer | Used |
|---|---|
| Backend | Python 3.10, FastAPI 0.115, Uvicorn 0.32, OpenCV 4.11 (bundled FFmpeg), NumPy 1.26, psutil |
| Frontend | React 19.1, TypeScript 5.8, Vite 6.3, Tailwind CSS 4.1, shadcn-style `components/ui` (Radix Slot and Label, cva, clsx, tailwind-merge), lucide-react |
| Supplied components | `file-upload-1.tsx` (adapted MP4 uploader), `liquid-metal-button.tsx` (`@paper-design/shaders`), `sparkles.tsx` (`@tsparticles` v3 and `framer-motion` 12) |

### Directory structure
```
interactive_prototype/
  README.md                      this file
  setup.bat  start_app.bat  start_backend.bat  start_frontend.bat
  backend/
    requirements.txt
    app/  main.py sessions.py pipeline.py video_io.py calibration.py camera_model.py ipm.py vectorize.py render.py
    uploads/                     temporary, cleaned automatically (git-ignored)
  frontend/
    package.json  package-lock.json  vite.config.ts  components.json  tsconfig*.json  index.html
    src/  App.tsx  main.tsx  index.css
          lib/        api.ts utils.ts (cn)
          components/ calibration-card.tsx processing-panels.tsx
          components/ui/   button.tsx label.tsx file-upload-1.tsx liquid-metal-button.tsx sparkles.tsx
          components/demo/ component-preview.tsx sparkles-demo.tsx
  docs/  TECHNICAL_EXPLANATION.md  TESTING_REPORT.md
  tools/ build_test_videos.py evaluate_calibration.py calibration_repeatability.py
         smoke_test_backend.py run_test_suite.sh dump_panels.py debug_rigidity.py
  test_videos/   test clips built from nuScenes (git-ignored; ground truth is used ONLY by tools/evaluate_calibration.py)
  test_results/  measured results quoted in docs/TESTING_REPORT.md
```

---

## 3. Automatic calibration (summary)

There are no input fields, no clicking of points and no calibration wizard. For each video, `calibration.py` does the following:

1. **Inspects** the MP4 (resolution, FPS, frame count, codec, container tags). Standard MP4 files do not carry intrinsics, so the app reports any camera tags it finds but never looks parameters up from them.
2. **Samples** about 40 short bursts of frames spread across the whole video.
3. **Finds static pixels** (bonnet or burned-in overlays) and excludes them from the ground.
4. **Tests lens straightness** with a plumb-line check. It estimates k1 only if straight edges become measurably straighter with it; otherwise it uses k1 = 0 and labels it *assumed*.
5. **Finds vanishing points.** It runs LSD line segments and RANSAC per frame to get the road-direction vanishing point, and takes a robust median over frames. This gives **yaw** and the **horizon height**.
6. **Tracks optical flow** (Lucas–Kanade):
   - the **focus of expansion** cross-checks yaw;
   - tracks on the road surface feed the **ground-motion rigidity test**. The tracked points are mapped onto the hypothesised ground plane; if the plane is right, consecutive frames differ by a rigid 2-D motion. This refines **pitch**, and estimates **roll** and the **focal length** only when the measurements clearly constrain them.
7. **Validates** with measured checks:
   - VP detection rate and dispersion;
   - first-half vs second-half stability;
   - FOE vs VP;
   - horizon band;
   - homography conditioning;
   - rigidity residual;
   - visible ground fraction.

   The result is `ok`, `low_confidence` or `failed`.
8. **Falls back** when needed:
   - With no metric reference (always, for an arbitrary MP4), it gives a **relative-scale BEV** (units = camera height).
   - With no trustworthy road plane, it shows **no BEV**: the original frames are still processed, and the panels show an honest warning.

The UI labels every parameter **estimated** or **assumed**. On the test clips, the following were **assumed** because the footage did not constrain them:
- focal length (prior 70° horizontal field of view);
- principal point (image centre);
- roll (0).

The UI says so, and *Technical diagnostics (read-only)* shows the measurements behind each decision. Details are in [docs/TECHNICAL_EXPLANATION.md](docs/TECHNICAL_EXPLANATION.md).

---

## 4. Processing pipeline (per frame)

Each frame goes through these steps:
1. Sequential decode (`Mp4FrameSource`).
2. `cv2.remap` with the precomputed IPM tables.
3. Paint evidence from this frame: top-hat and Lab colour extraction, the road-surface context test and shape filters (thin paint, plus a wide-paint pass for pedestrian crossings).
4. **Ego-motion:** Lucas–Kanade tracks on the road, mapped to the ground plane, then a RANSAC rigid 2-D fit gives how the ground moved since the previous frame.
5. **Evidence fusion:** paint, road-surface and kerb-edge evidence maps are warped by that motion and blended with this frame's evidence (about 5 frames of memory).
6. **Lane dividers:** column histogram of the fused paint, then sliding windows along the road, then a smooth quadratic curve. Dashed or solid is measured on the current frame.
7. **Road boundaries:** the outermost solid edge line on each side, or else a per-row scan outwards from the vehicle's lane until the road surface ends or a kerb edge is met. The edge points are median-filtered and fitted with smooth curves.
8. Rendering of the three panels in the style of the earlier final-demo video. Optionally the 1920×1080 demo frame is written to an MP4 (*Record demo video*).
9. **One** WebSocket message `frame_update` containing all three images, the shared `frame_index` and `timestamp_seconds`, the vector candidates (in h units) and the measured metrics.

The frame queue is bounded at 3. When the browser falls behind, the decoder **waits**, so frames are never dropped.

*Pace to source FPS* (on by default) limits delivery to the video's frame rate, to simulate a live camera. When the computer is slower than the source, the UI says **"Not keeping up with real time"** and shows the measured FPS.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| "Backend not reachable" banner | Start `start_backend.bat`, or check that port 8000 is free: `netstat -ano \| findstr :8000` |
| `setup.bat`: "Python 3.10, 3.11 or 3.12 is required" | Install Python 3.10–3.12. NumPy 1.26 has no wheels for 3.13. |
| Upload rejected: "not an MP4/ISO-BMFF container" / "no 'moov' box" | The file is not a real MP4, or it is truncated. Re-export it, or convert it: `ffmpeg -i in.mov -c:v libx264 -pix_fmt yuv420p out.mp4` |
| "could not be decoded" | Unsupported codec (for example some AV1/ProRes files). Convert to H.264 with the command above. Installing FFmpeg is optional and only needed for this conversion: `winget install Gyan.FFmpeg`, then check with `ffmpeg -version`. |
| Calibration `failed` | The road lines are not visible or consistent enough (for example a stationary car, a dirt road, night without markings). Only the original frames are shown; this is intentional, so the app does not show a misleading BEV. |
| "Not keeping up with real time" | The computer processes fewer frames per second than the video contains. All frames are still processed. Turn off *Pace to source FPS* to see the maximum throughput. |
| Stream lost | Reload the page. The server pauses processing when the WebSocket disconnects. |
| Port 5173 / 8000 in use | Close other dev servers, or change `--port` in `start_backend.bat` and `vite.config.ts`. |

## 6. Limitations (read before presenting)

- **Assumptions:** monocular vision, a flat road and a single fixed calibration per video. Raised objects (vehicles, kerbs, poles) smear in the IPM. Pitch changes from braking or bumps are not modelled.
- **Relative scale only:**
  - 1 h ≈ the camera height, but on the nuScenes clips the fitted unit came out at 1.73–1.87 m (true camera height 1.50 m), because the assumed focal length is about 9% short.
  - The BEV *shape* error after scale alignment was 2.2–3.8% (see the testing report).
- **The focal length is usually not observable** from ordinary dashcam footage without strong turns or a clearly pitched camera. The 70° prior is used and labelled.
- **Panel C is video-derived and cannot equal the map-derived panel C of the earlier demo.**
  - The earlier demo drew HD-map geometry (ground truth), including kerbs hidden behind parked cars and side streets the camera cannot see.
  - Measured against that map on Queenstown (40 keyframes), points within 1 m of the map:

    | Class | Precision | Recall |
    |---|---|---|
    | Road boundary | 47% (75% at 1.5 m) | 14% |
    | Lane divider | 58% | 19% |

    Before tracking, the figures were 11% / 0.2% for boundaries and 19% / 39% for dividers.
  - Lines that are drawn are mostly right. Many map lines that are faint, occluded or out of view are not drawn.
  - Pedestrian crossings of the Queenstown "dotted" type are not detected.
- **Traditional CV detection is unreliable** in some conditions. It struggles with vehicles directly ahead, cones, strong shadows, faded paint and unmarked roads. When nothing is found, the panel says *"No reliable vector candidates detected in this frame."*
- **Generalization is limited.** It was tested on all 10 nuScenes v1.0-mini CAM_FRONT scenes (Singapore and Boston, two vehicles, day and night): 8 calibrated `ok` (one of them visibly wrong, scene 0757), 1 `low_confidence`, 1 `failed`. It has **not** been validated on an independent consumer dashcam.
- **Performance is measured on this laptop** (i5-1035G1, 8 GB):
  - with the tracked vectorizer, about 125 ms per frame for processing alone (≈ 8 fps);
  - more with the browser open or with *Record demo video* on.

  **Real time is not reached** for 12 fps 1600×900 video (see the testing report).
