"""Processing sessions: one uploaded video -> inspection -> automatic calibration -> sequential processing.

Threads: calibration and frame processing run in worker threads; results reach the WebSocket through
asyncio queues.  The frame queue is BOUNDED: when the browser falls behind, the decoder blocks
(backpressure) instead of dropping frames, so every frame is processed and delivered in order.
"""
import asyncio, collections, json, os, shutil, threading, time, traceback, uuid
import numpy as np
import psutil

from .video_io import Mp4FrameSource, VideoError, inspect_video
from .calibration import calibrate, CalibrationFailed
from .pipeline import FramePipeline, jpeg_b64
from .demo_video import DemoRecorder

FRAME_QUEUE_SIZE = 3
_PROC = psutil.Process(os.getpid())


class Session:
    def __init__(self, sid, path, filename, loop):
        self.id, self.path, self.filename, self.loop = sid, path, filename, loop
        self.state = "uploaded"
        self.inspection = None
        self.calibration = None
        self.stages = []
        self.error = None
        self.ctrl_q: asyncio.Queue = asyncio.Queue()
        self.frame_q: asyncio.Queue = asyncio.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.pipeline = None
        self.worker = None
        self.run_id = 0
        self.pause_evt = threading.Event(); self.pause_evt.set()      # set = running
        self.stop_flag = False
        self.pace = True
        self.record = False
        self.demo_path = None
        self.last_frame_index = -1
        self.summary = None
        self.client_connected = False

    # ---------------------------------------------------------------- messaging
    def emit(self, msg):
        msg.setdefault("session_id", self.id)
        self.loop.call_soon_threadsafe(self.ctrl_q.put_nowait, msg)

    def set_state(self, state, **extra):
        self.state = state
        self.emit({"type": "status", "state": state, **extra})

    def snapshot(self):
        return {"type": "session_state", "session_id": self.id, "state": self.state, "filename": self.filename,
                "inspection": self.inspection, "calibration_stages": self.stages, "calibration": self._public_cal(),
                "error": self.error, "last_frame_index": self.last_frame_index, "summary": self.summary, "pace": self.pace}

    def _public_cal(self):
        if self.calibration is None:
            return None
        return {k: v for k, v in self.calibration.items() if k != "static_mask_small"}

    # ---------------------------------------------------------------- calibration (automatic, no user input)
    def start_analysis(self):
        threading.Thread(target=self._analyse, daemon=True, name=f"calib-{self.id}").start()

    def _stage(self, stage, detail=""):
        rec = {"stage": stage, "detail": detail, "t": round(time.time(), 3)}
        self.stages.append(rec)
        self.emit({"type": "calibration_stage", **rec})

    def _analyse(self):
        try:
            self.set_state("calibrating")
            t0 = time.time()
            try:
                cal = calibrate(self.path, self.inspection, progress=self._stage)
            except CalibrationFailed as e:
                cal = {"status": "failed", "scale_mode": "none", "reason": str(e), "warnings": [
                    "Automatic road-plane estimation failed. A reliable BEV transformation could not be established for this video; "
                    "only the original frames will be shown."], "checks": [], "parameters": [], "elapsed_s": round(time.time() - t0, 2)}
            self.calibration = cal
            if cal["status"] != "failed":
                self._stage("building_projection", "precomputing IPM remap tables")
                self.pipeline = FramePipeline(cal)
                cal["bev_valid_fraction"] = round(self.pipeline.projector.valid_fraction, 3)
                cal["preview"] = self._preview()
            else:
                self.pipeline = FramePipeline(None)
            self._stage("calibration_completed", cal["status"])
            self.emit({"type": "calibration_result", "calibration": self._public_cal()})
            self.set_state("ready")
        except VideoError as e:
            self.fail(e.to_dict())
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.fail({"code": "calibration_error", "message": f"Automatic calibration crashed: {e}", "hint": "See the backend console."})

    def _preview(self):
        import cv2
        cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.inspection["frame_count"] // 2)
        ok, fr = cap.read(); cap.release()
        if not ok:
            return None
        self.pipeline.reset()
        out, vecs, _, _ = self.pipeline.process(fr)
        self.pipeline.reset()
        return {"frame_index": self.inspection["frame_count"] // 2, "original": out["original_frame"], "bev": out["bev_frame"]}

    def fail(self, err):
        self.error = err
        self.set_state("error", error=err)

    # ---------------------------------------------------------------- processing control
    def start(self, pace=True, record=False):
        if self.state not in ("ready", "completed", "stopped"):
            return False
        self.pace = bool(pace)
        self.record = bool(record) and self.pipeline is not None and self.pipeline.bev_enabled
        self.stop_flag = False
        self.pause_evt.set()
        self.run_id += 1
        # drop frames still queued from a previous run so panels never mix runs
        while not self.frame_q.empty():
            self.frame_q.get_nowait()
        self.summary = None
        self.pipeline.reset()
        self.worker = threading.Thread(target=self._process, args=(self.run_id,), daemon=True, name=f"proc-{self.id}")
        self.worker.start()
        return True

    def pause(self):
        if self.state == "processing":
            self.pause_evt.clear(); self.set_state("paused")

    def resume(self):
        if self.state == "paused":
            self.pause_evt.set(); self.set_state("processing")

    def stop(self):
        self.stop_flag = True
        self.pause_evt.set()

    def _put_frame(self, msg, run_id):
        """Blocking put into the bounded frame queue (backpressure). Returns False if the run was stopped."""
        while not self.stop_flag and run_id == self.run_id:
            fut = asyncio.run_coroutine_threadsafe(asyncio.wait_for(self.frame_q.put(msg), 0.5), self.loop)
            try:
                fut.result()
                return True
            except (asyncio.TimeoutError, TimeoutError):
                continue
        return False

    def _process(self, run_id):
        info = self.inspection
        fps = info["fps"]
        src = Mp4FrameSource(self.path, fps, info["frame_count"])
        proc_ms = collections.deque(maxlen=30)
        delivered = collections.deque(maxlen=60)
        n_done, t_start, paused_s = 0, time.perf_counter(), 0.0
        all_proc, n_vec_frames = [], 0
        self.set_state("processing")
        pace_t0, pace_i0 = None, None
        recorder = None
        if self.record:
            self.demo_path = os.path.join(os.path.dirname(self.path), "..", "exports", f"{self.id}_demo.mp4")
            name = self.filename if len(self.filename) <= 48 else self.filename[:45] + "..."
            recorder = DemoRecorder(self.demo_path, fps, f"PSJ_01  |  {name}  |  camera frame -> IPM BEV -> video-derived vectorized BEV",
                                    self.calibration)
        try:
            t_dec = time.perf_counter()
            for idx, ts, frame in src.frames():
                decode_ms = (time.perf_counter() - t_dec) * 1e3
                if not self.pause_evt.is_set():
                    tp = time.perf_counter()
                    self.pause_evt.wait()
                    paused_s += time.perf_counter() - tp
                    pace_t0 = None                                     # restart pacing clock after a pause
                    delivered.clear()
                if self.stop_flag or run_id != self.run_id:
                    break
                t0 = time.perf_counter()
                panels, vecs, vinfo, tm = self.pipeline.process(frame)
                p_ms = (time.perf_counter() - t0) * 1e3
                if recorder is not None:
                    tr = time.perf_counter()
                    recorder.write(idx, info["frame_count"], ts, *self.pipeline.last_panels, vecs)
                    tm["record_ms"] = (time.perf_counter() - tr) * 1e3
                proc_ms.append(p_ms); all_proc.append(p_ms)
                if self.pace:
                    now = time.perf_counter()
                    if pace_t0 is None:
                        pace_t0, pace_i0 = now, idx
                    target = pace_t0 + (idx - pace_i0) / fps
                    if target > now:
                        time.sleep(target - now)
                    elif now - target > 1.0:
                        pace_t0, pace_i0 = now, idx                     # behind by > 1 s: re-anchor (no frames are skipped)
                now = time.perf_counter()
                delivered.append(now)
                eff = (len(delivered) - 1) / (delivered[-1] - delivered[0]) if len(delivered) > 1 and delivered[-1] > delivered[0] else None
                proc_fps = 1000.0 / float(np.mean(proc_ms))
                msg = {
                    "type": "frame_update", "session_id": self.id, "run_id": run_id,
                    "frame_index": idx, "timestamp_seconds": round(ts, 4), "total_frames": info["frame_count"],
                    **panels,
                    "vector_candidates": vecs, "vector_info": vinfo,
                    "metrics": {
                        "processing_time_ms": round(p_ms, 1), "decode_ms": round(decode_ms, 1),
                        **{k: round(v, 1) for k, v in tm.items()},
                        "processing_fps": round(proc_fps, 2),
                        "effective_fps": None if eff is None else round(eff, 2),
                        "source_fps": fps,
                        "keeping_up_with_source": None if eff is None else bool(proc_fps >= fps and (eff >= 0.95 * fps or not self.pace)),
                        "paced_to_source": self.pace,
                        "progress": round((idx + 1) / info["frame_count"], 4),
                        "memory_rss_mb": round(_PROC.memory_info().rss / 1e6, 1),
                        "queue_depth": self.frame_q.qsize(),
                    },
                }
                if not self._put_frame(msg, run_id):
                    break
                self.last_frame_index = idx
                n_done += 1
                n_vec_frames += bool(vecs)
                t_dec = time.perf_counter()
            if recorder is not None:
                recorder.close()                                       # finalise the MP4 before announcing it
            wall = time.perf_counter() - t_start - paused_s
            stopped = self.stop_flag or run_id != self.run_id
            if run_id == self.run_id:
                self.summary = {
                    "frames_processed": n_done, "total_frames": info["frame_count"], "wall_time_s": round(wall, 2), "paused_time_s": round(paused_s, 2),
                    "mean_processing_ms": round(float(np.mean(all_proc)), 2) if all_proc else None,
                    "p95_processing_ms": round(float(np.percentile(all_proc, 95)), 2) if all_proc else None,
                    "mean_processing_fps": round(1000 / float(np.mean(all_proc)), 2) if all_proc else None,
                    "effective_fps": round(n_done / wall, 2) if wall > 0 else None,
                    "frames_with_vector_candidates": n_vec_frames, "paced_to_source": self.pace,
                    "demo_video": (f"/api/videos/{self.id}/demo.mp4" if recorder is not None and recorder.frames else None),
                    "demo_video_frames": recorder.frames if recorder is not None else 0,
                }
                self.set_state("stopped" if stopped else "completed", summary=self.summary)
        except VideoError as e:
            self.fail(e.to_dict())
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self.fail({"code": "processing_error", "message": f"Processing failed: {e}", "hint": "See the backend console."})
        finally:
            src.close()
            if recorder is not None:
                recorder.close()

    def reset(self):
        self.stop()
        self.run_id += 1
        while not self.frame_q.empty():
            self.frame_q.get_nowait()
        self.last_frame_index = -1
        self.summary = None
        if self.calibration is not None and self.state != "error":
            self.set_state("ready")

    def close(self):
        self.stop()
        self.run_id += 1


class SessionManager:
    def __init__(self, upload_dir):
        self.upload_dir = upload_dir
        self.sessions = {}
        os.makedirs(upload_dir, exist_ok=True)
        exports = os.path.join(upload_dir, "..", "exports")
        os.makedirs(exports, exist_ok=True)
        for d in (upload_dir, exports):                       # temporary files from a previous run
            for f in os.listdir(d):
                try:
                    os.remove(os.path.join(d, f))
                except OSError:
                    pass

    def new_id(self):
        return uuid.uuid4().hex[:12]

    def path_for(self, sid):
        return os.path.join(self.upload_dir, f"{sid}.mp4")

    MAX_SESSIONS = 3

    def create(self, sid, filename, loop):
        # Each browser tab removes its own previous video (DELETE); other tabs/clients keep theirs.
        # Only when too many accumulate is the oldest session that is not processing evicted.
        while len(self.sessions) >= self.MAX_SESSIONS:
            idle = [k for k, v in self.sessions.items() if v.state not in ("processing", "calibrating")]
            self.delete((idle or list(self.sessions))[0])
        s = Session(sid, self.path_for(sid), filename, loop)
        self.sessions[sid] = s
        return s

    def get(self, sid):
        return self.sessions.get(sid)

    def delete(self, sid):
        s = self.sessions.pop(sid, None)
        if s:
            s.close()
            if s.demo_path and os.path.exists(s.demo_path):
                try:
                    os.remove(s.demo_path)
                except OSError:
                    pass
            for _ in range(20):
                try:
                    if os.path.exists(s.path):
                        os.remove(s.path)
                    break
                except PermissionError:                        # decoder may still hold the file briefly
                    time.sleep(0.1)
        return s is not None
