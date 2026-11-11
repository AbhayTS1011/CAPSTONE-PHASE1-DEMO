"""Debug (evaluation only): draw the map-derived reference polylines (converted to h) on our IPM for keyframes."""
import json, os, sys, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend")); sys.path.insert(0, os.path.dirname(__file__))
from app.video_io import inspect_video, Mp4FrameSource
from app.calibration import calibrate
from app.pipeline import FramePipeline
from benchmark_vectors import keyframe_indices
v, out_dir, mph = sys.argv[1], sys.argv[2], float(sys.argv[3]); want = [int(x) for x in sys.argv[4].split(",")]
os.makedirs(out_dir, exist_ok=True)
kf = keyframe_indices(v); info = inspect_video(v); P = FramePipeline(calibrate(v, info)); Pr = P.projector
idx = sorted(kf)
col = {"divider": (0, 140, 255), "boundary": (60, 200, 60), "ped_crossing": (255, 120, 40)}
pcol = {"lane_divider": (0, 255, 255), "lane_divider_yellow": (0, 255, 255), "road_boundary": (255, 0, 255), "pedestrian_crossing": (255, 255, 0)}
for i, ts, fr in Mp4FrameSource(v, info["fps"], info["frame_count"]).frames():
    _, vecs, vi, _ = P.process(fr)
    if i not in kf or idx.index(i) not in want: continue
    bev = P.projector.warp(fr)
    road = (P.vectorizer.road > 0.55).astype(np.uint8) * 255; paint = (P.vectorizer.paint > 0.3).astype(np.uint8) * 255
    dbg = np.dstack([paint, road // 2, np.zeros_like(road)])
    for vv in vecs:
        q = np.asarray(vv["points_h"]); cc, rr = Pr.ground_to_px(q[:, 0], q[:, 1])
        cv2.polylines(dbg, [np.column_stack([cc, rr]).round().astype(np.int32)], vv["type"] == "pedestrian_crossing", pcol.get(vv["type"], (255, 255, 255)), 3, cv2.LINE_AA)
    print(i, vi.get("ego_motion"), vi.get("ego_forward_h"), vi.get("boundaries"), vi.get("dividers"))
    for c, polys in kf[i]["visible"].items():
        for p in polys:
            p = np.asarray(p) / mph
            cc, rr = Pr.ground_to_px(p[:, 0], p[:, 1])
            cv2.polylines(bev, [np.column_stack([cc, rr]).round().astype(np.int32)], False, col[c], 2, cv2.LINE_AA)
    for c, polys in kf[i]["visible"].items():
        for p in polys:
            p = np.asarray(p) / mph
            cc, rr = Pr.ground_to_px(p[:, 0], p[:, 1])
            cv2.polylines(dbg, [np.column_stack([cc, rr]).round().astype(np.int32)], False, col[c], 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(out_dir, f"overlay_kf{idx.index(i):02d}_f{i}.jpg"), np.hstack([bev, dbg]))
