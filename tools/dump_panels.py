"""Debug helper: calibrate a video and write side-by-side panel images for selected frame indices."""
import sys, os, base64, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.video_io import inspect_video, Mp4FrameSource
from app.calibration import calibrate
from app.pipeline import FramePipeline
v, out_dir, idx = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
os.makedirs(out_dir, exist_ok=True)
info = inspect_video(v); P = FramePipeline(calibrate(v, info))
for i, t, fr in Mp4FrameSource(v, info["fps"], info["frame_count"]).frames():
    out, vecs, vi, tm = P.process(fr)          # process EVERY frame so the temporal tracker has full history
    if i in idx:
        ims = [cv2.imdecode(np.frombuffer(base64.b64decode(out[k]), np.uint8), 1) for k in ("original_frame", "bev_frame", "vectorized_bev_frame")]
        ims = [cv2.resize(im, (int(im.shape[1] * 720 / im.shape[0]), 720)) for im in ims]
        cv2.imwrite(os.path.join(out_dir, f"{os.path.basename(v)[:10]}_{i:04d}.jpg"), np.hstack(ims))
        print(i, [(x["type"], x["length_h"], x["cv_score"]) for x in vecs], vi)
    if i >= max(idx):
        break
