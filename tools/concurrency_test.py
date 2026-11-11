"""Regression test for the crash of 2026-11-05: two automatic calibrations running at the same time."""
import glob, os, sys, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.video_io import inspect_video
from app.calibration import calibrate
vids = sys.argv[1:] or sorted(glob.glob("test_videos/scene-0796*.mp4") + glob.glob("test_videos/scene-0655*.mp4"))
res = {}
def run(v):
    try:
        res[v] = calibrate(v, inspect_video(v))["status"]
    except Exception as e:  # noqa: BLE001
        res[v] = f"EXCEPTION {e!r}"
ts = [threading.Thread(target=run, args=(v,)) for v in vids]
[t.start() for t in ts]; [t.join() for t in ts]
for v, r in res.items():
    print(os.path.basename(v), "->", r)
print("PASS" if all(not r.startswith("EXCEPTION") for r in res.values()) else "FAIL")
