"""Repeatability of the automatic calibration: re-run with different RANSAC seeds and report the spread."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.video_io import inspect_video
from app.calibration import calibrate
out = {}
for v in sys.argv[1:-1]:
    info = inspect_video(v)
    rows = [calibrate(v, info, seed=s)["camera"] for s in range(4)]
    a = {k: [r[k] for r in rows] for k in ("pitch_deg", "yaw_deg", "roll_deg", "focal_px")}
    out[os.path.basename(v)] = {k: {"values": np.round(x, 3).tolist(), "range": round(float(np.ptp(x)), 3)} for k, x in a.items()}
    print(os.path.basename(v), {k: out[os.path.basename(v)][k]["range"] for k in a})
json.dump(out, open(sys.argv[-1], "w"), indent=1)
