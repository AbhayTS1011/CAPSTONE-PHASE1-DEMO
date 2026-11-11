"""Accuracy of the video-derived vectorized BEV against the map-derived reference of the earlier demo.

EVALUATION ONLY: the reference polylines (nuScenes Map Expansion, clipped to the camera-visible ROI) and the
nuScenes timestamps are read here to score the output; the application never sees them.

The app works in camera heights (h). For scoring, h is converted to metres with the scale fitted by
tools/evaluate_calibration.py (projection-consistent metres per h for this clip), so the score measures
geometry, not the unknown absolute scale.

Metric (per class, MapTR-style point matching): reference and predicted polylines are resampled every 0.5 m;
precision = share of predicted points within T of a reference point of the same class,
recall    = share of reference points within T of a predicted point of the same class, for T = 1.0 m and 1.5 m.

  python tools/benchmark_vectors.py test_videos/scene-0796_singapore-queenstown_CAM_FRONT_234f.mp4 --out test_results/vector_benchmark.json
"""
import argparse, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "backend"))
from app.video_io import inspect_video, Mp4FrameSource          # noqa: E402
from app.calibration import calibrate                            # noqa: E402
from app.pipeline import FramePipeline                           # noqa: E402

PROJECT = os.path.abspath(os.path.join(HERE, "..", ".."))
REF_DIR = os.path.join(PROJECT, "github", "data", "current_split", "vectorized_bev", "geometry")
CLASS_OF = {"lane_divider": "divider", "lane_divider_yellow": "divider", "road_boundary": "boundary", "lane_marking": "divider", "lane_marking_yellow": "divider", "divider": "divider",
            "road_edge_candidate": "boundary", "boundary": "boundary",
            "pedestrian_crossing": "ped_crossing", "ped_crossing": "ped_crossing"}


def resample(poly, step=0.5):
    p = np.asarray(poly, float)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return p[:1]
    t = np.arange(0, s[-1] + 1e-9, step)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def points(polys):
    pts = [resample(p) for p in polys if len(p) >= 2]
    return np.vstack(pts) if pts else np.zeros((0, 2))


def match(a, b, T):
    if len(a) == 0:
        return np.zeros(0, bool)
    if len(b) == 0:
        return np.zeros(len(a), bool)
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
    return d < T


def keyframe_indices(video):
    """frame index in the 12 Hz test clip for each reference keyframe (matched by nuScenes timestamp)."""
    sd = json.load(open(os.path.join(PROJECT, "v1.0-mini", "v1.0-mini", "sample_data.json"), encoding="utf-8"))
    refs = {}
    for f in glob.glob(os.path.join(REF_DIR, "*.json")):
        r = json.load(open(f))
        refs[r["timestamp_us"]] = r
    scene_log = None
    for r in sd:
        if r["timestamp"] in refs:
            scene_log = r["filename"].split("/")[-1].split("__")[0]
            break
    cams = sorted([r for r in sd if "CAM_FRONT/" in r["filename"] and r["filename"].split("/")[-1].startswith(scene_log)],
                  key=lambda r: r["timestamp"])
    ts = [r["timestamp"] for r in cams]
    lo, hi = min(refs), max(refs)
    # restrict to the scene: the test clip covers the scene's camera frames, which bracket the keyframes
    first = max(i for i, t in enumerate(ts) if t <= lo + 1) if any(t <= lo + 1 for t in ts) else 0
    out = {}
    for t, r in refs.items():
        i = ts.index(t)
        out[i - first] = r
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--out"); ap.add_argument("--m-per-h", type=float)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    kf = keyframe_indices(a.video)
    info = inspect_video(a.video)
    cal = calibrate(a.video, info)
    mph = a.m_per_h
    if mph is None:
        ev = json.load(open(os.path.join(HERE, "..", "test_results", "calibration_eval.json")))
        mph = [r for r in ev if r["video"] == os.path.basename(a.video)][0]["geometry_error"]["fitted_metres_per_h"]
    P = FramePipeline(cal)
    per_frame = []
    tot = {c: {T: [0, 0, 0, 0] for T in (1.0, 1.5)} for c in ("divider", "boundary", "ped_crossing")}
    jitter = []
    prev = None
    for i, ts, fr in Mp4FrameSource(a.video, info["fps"], info["frame_count"]).frames():
        _, vecs, _, _ = P.process(fr)
        if i not in kf:
            continue
        ref = kf[i]["visible"]
        pred = {}
        for v in vecs:
            c = CLASS_OF.get(v["type"])
            if c:
                pred.setdefault(c, []).append(np.asarray(v["points_h"]) * mph)
        row = {"frame": i}
        for c in tot:
            rp, pp = points(ref.get(c, [])), points(pred.get(c, []))
            for T in tot[c]:
                pm, rm = match(pp, rp, T), match(rp, pp, T)
                tot[c][T][0] += int(pm.sum()); tot[c][T][1] += len(pp); tot[c][T][2] += int(rm.sum()); tot[c][T][3] += len(rp)
            row[c] = {"pred_pts": len(pp), "ref_pts": len(rp)}
        per_frame.append(row)
    res = {"video": os.path.basename(a.video), "label": a.label, "keyframes_scored": len(per_frame), "metres_per_h_used": mph,
           "calibration_status": cal["status"], "scores": {}}
    for c, d in tot.items():
        res["scores"][c] = {f"T={T}m": {"precision": round(v[0] / v[1], 3) if v[1] else None, "recall": round(v[2] / v[3], 3) if v[3] else None,
                                        "pred_points": v[1], "ref_points": v[3]} for T, v in d.items()}
    print(json.dumps(res["scores"], indent=1))
    if a.out:
        res["per_frame"] = per_frame
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
