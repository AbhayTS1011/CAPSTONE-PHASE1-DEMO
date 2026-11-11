"""FastAPI server: upload/validation REST endpoints + one WebSocket per session for synchronized frames.

Runs entirely on this computer; videos are never sent anywhere else.
  uvicorn app.main:app --host 127.0.0.1 --port 8000
"""
import asyncio, json, os, shutil

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .sessions import SessionManager
from .video_io import VideoError, inspect_video

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.environ.get("PSJ_UPLOAD_DIR", os.path.join(BASE, "uploads"))
MAX_UPLOAD_MB = int(os.environ.get("PSJ_MAX_UPLOAD_MB", "2048"))
FRONTEND_DIST = os.path.join(BASE, "..", "frontend", "dist")
VERSION = "0.1.0"

app = FastAPI(title="Dashcam -> Automatic Calibration -> IPM -> Vectorized BEV (local prototype)", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
manager = SessionManager(UPLOAD_DIR)


def err(status, code, message, hint=""):
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, "hint": hint}})


@app.get("/api/health")
def health():
    return {"ok": True, "version": VERSION, "max_upload_mb": MAX_UPLOAD_MB, "accepted": [".mp4"]}


@app.post("/api/videos")
async def upload(request: Request, file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if not name:
        return err(400, "no_file", "No video was selected.")
    if not name.lower().endswith(".mp4"):
        return err(415, "unsupported_extension", f"'{name}' is not an .mp4 file.", "Only MP4 videos are accepted.")
    clen = request.headers.get("content-length")
    if clen and int(clen) > (MAX_UPLOAD_MB + 1) * 1024 * 1024:
        return err(413, "too_large", f"The upload exceeds the {MAX_UPLOAD_MB} MB limit.", "Trim the video or set PSJ_MAX_UPLOAD_MB.")
    sid = manager.new_id()
    session = manager.create(sid, name, asyncio.get_running_loop())
    written = 0
    try:
        with open(session.path, "wb") as out:
            while True:
                chunk = await file.read(4 * 1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(413)
                out.write(chunk)
    except HTTPException:
        manager.delete(sid)
        return err(413, "too_large", f"The upload exceeds the {MAX_UPLOAD_MB} MB limit.", "Trim the video or set PSJ_MAX_UPLOAD_MB.")
    except OSError as e:
        manager.delete(sid)
        return err(507, "storage_error", f"Could not store the upload: {e.strerror or e}", f"Check free disk space in {UPLOAD_DIR}.")
    if written == 0:
        manager.delete(sid)
        return err(400, "empty_file", "The uploaded file is empty.")
    session.state = "inspecting"
    try:
        session.inspection = await asyncio.to_thread(inspect_video, session.path)
    except VideoError as e:
        manager.delete(sid)
        return err(422, e.code, e.message, e.hint)
    session.inspection["filename"] = name
    session.start_analysis()                      # automatic calibration starts immediately
    return {"session_id": sid, "filename": name, "inspection": session.inspection}


@app.get("/api/videos/{sid}")
def get_video(sid: str):
    s = manager.get(sid)
    if not s:
        raise HTTPException(404, "unknown session")
    return s.snapshot()


@app.get("/api/videos/{sid}/demo.mp4")
def demo_video(sid: str):
    s = manager.get(sid)
    if not s or not s.demo_path or not os.path.exists(s.demo_path) or s.state == "processing":
        raise HTTPException(404, "no finished demo video for this session")
    base = os.path.splitext(s.filename)[0]
    return FileResponse(s.demo_path, media_type="video/mp4", filename=f"{base}_PSJ01_BEV_demo.mp4")


@app.delete("/api/videos/{sid}")
def delete_video(sid: str):
    return {"deleted": manager.delete(sid)}


@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    s = manager.get(sid)
    if not s:
        await ws.send_text(json.dumps({"type": "error", "error": {"code": "unknown_session", "message": "This video session no longer exists."}}))
        await ws.close()
        return
    while not s.ctrl_q.empty():                    # the snapshot below supersedes queued control messages
        s.ctrl_q.get_nowait()
    s.client_connected = True
    await ws.send_text(json.dumps(s.snapshot()))
    if s.state == "paused" and getattr(s, "paused_by_disconnect", False):
        s.paused_by_disconnect = False

    async def receiver():
        while True:
            msg = json.loads(await ws.receive_text())
            cmd = msg.get("cmd")
            if cmd == "start":
                if not s.start(pace=msg.get("pace", True), record=msg.get("record", False)):
                    s.emit({"type": "notice", "message": f"Cannot start while the session is '{s.state}'."})
            elif cmd == "pause":
                s.pause()
            elif cmd == "resume":
                s.resume()
            elif cmd == "stop":
                s.stop()
            elif cmd == "reset":
                s.reset()
            elif cmd == "set_pace":
                s.pace = bool(msg.get("pace", True))
                s.emit({"type": "notice", "message": f"Pacing to source FPS {'on' if s.pace else 'off'}."})

    async def sender():
        while True:
            gc = asyncio.ensure_future(s.ctrl_q.get())
            gf = asyncio.ensure_future(s.frame_q.get())
            try:
                done, pending = await asyncio.wait({gc, gf}, return_when=asyncio.FIRST_COMPLETED)
            finally:                                   # also on cancellation, so no orphaned get() tasks remain
                for p in (gc, gf):
                    if not p.done():
                        p.cancel()
            for d in (gc, gf):                      # control first, then the frame (both if both completed)
                if d in done:
                    m = d.result()
                    if m.get("type") == "frame_update" and m.get("run_id") != s.run_id:
                        continue                      # frame from a stopped/reset run
                    await ws.send_text(json.dumps(m))

    tasks = [asyncio.create_task(receiver()), asyncio.create_task(sender())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            if t.exception() and not isinstance(t.exception(), WebSocketDisconnect):
                print("websocket task error:", repr(t.exception()))
    finally:
        for t in tasks:
            t.cancel()
        s.client_connected = False
        if s.state == "processing":                 # connection lost: pause so no frames pile up unseen
            s.pause()
            s.paused_by_disconnect = True


if os.path.isdir(FRONTEND_DIST):                    # production build served from the same origin
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = os.path.join(FRONTEND_DIST, path)
        if path and os.path.isfile(f):
            return FileResponse(f)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
