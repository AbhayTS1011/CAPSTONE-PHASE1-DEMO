"""Score the automatic calibration against nuScenes ground truth (EVALUATION ONLY).

The application never sees the ground truth; this script runs the same inspect+calibrate code the server
runs and then compares.  usage: python tools/evaluate_calibration.py test_videos/*.mp4 [--out report.json]
"""
import argparse, json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from app.video_io import inspect_video                      # noqa: E402
from app.calibration import calibrate, CalibrationFailed     # noqa: E402
from app.camera_model import CameraModel                     # noqa: E402


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def truth_angles(gt):
    R = quat_to_R(gt["cam_to_ego_rotation_wxyz"])          # camera -> ego; rows = ego axes in camera coords
    K = np.array(gt["camera_intrinsic"])
    w, h = gt["image_size"]
    cam = CameraModel(w, h, K[0, 0], K[0, 2], K[1, 2], 0.0, R[0], R[1], R[2])
    return cam.angles(), K[0, 0], gt["cam_to_ego_translation_m"][2], cam


def bev_shape_error(cam_true, cam_est, h_true):
    """Ground grid (metres) -> true pixels -> estimated ground (camera heights); best similarity alignment;
    returns RMS residual as % of the grid extent (scale-free BEV fidelity) and the fitted metres-per-h."""
    X, Y = np.meshgrid(np.linspace(6, 30, 13), np.linspace(-6, 6, 7))
    cam_true.h = h_true
    uv, z = cam_true.project_ground(X.ravel(), Y.ravel())
    g, ok = cam_est.pixel_to_ground(uv)
    A = np.column_stack([X.ravel(), Y.ravel()])[ok]; B = g[ok]
    ca, cb = A.mean(0), B.mean(0); A0, B0 = A - ca, B - cb
    U, S, Vt = np.linalg.svd(B0.T @ A0)
    Rm = U @ Vt; sc = S.sum() / (B0 ** 2).sum()
    res = np.linalg.norm((B0 @ Rm) * sc - A0, axis=1)
    return 100 * float(np.sqrt((res ** 2).mean())) / float(np.ptp(A[:, 0])), float(sc)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--out")
    a = ap.parse_args()
    rows = []
    for v in a.videos:
        gtp = os.path.join(os.path.dirname(v), "ground_truth", os.path.basename(v).replace(".mp4", ".json"))
        t = time.time()
        info = inspect_video(v)
        try:
            cal = calibrate(v, info)
        except CalibrationFailed as e:
            rows.append({"video": os.path.basename(v), "status": "failed", "reason": str(e)}); print(rows[-1]); continue
        r = {"video": os.path.basename(v), "status": cal["status"], "elapsed_s": round(time.time() - t, 1),
             "est": {"f_px": round(cal["camera"]["focal_px"], 1), **{k: round(cal["camera"][k], 2) for k in ("pitch_deg", "yaw_deg", "roll_deg")}},
             "sources": {p["name"]: p["source"] for p in cal["parameters"]},
             "checks": {c["name"]: [c["value"], c["status"]] for c in cal["checks"]},
             "roi_units_h": cal["roi_units_h"], "diag": {k: cal["diagnostics"][k] for k in ("sampled_frames", "forward_vp_frames", "vertical_vp_frames", "flow_pairs", "foe_pairs")},
             "focal_diag": cal["diagnostics"]["focal"], "pitch_diag": cal["diagnostics"]["pitch_refinement"].get("refined")}
        if os.path.exists(gtp):
            gt = json.load(open(gtp))
            ang, f_true, h_true, cam_true = truth_angles(gt)
            from app.camera_model import CameraModel as CM
            ce = cal["camera"]
            cam_est = CM(ce["image_size"][0], ce["image_size"][1], ce["focal_px"], *ce["principal_point_px"], ce["k1"],
                         np.array(ce["fwd"]), np.array(ce["left"]), np.array(ce["up"]))
            vpt = cam_true.K @ cam_true.fwd; vpt = vpt[:2] / vpt[2]
            vpe = np.array(cal["diagnostics"]["vanishing_point_px"])
            hz_t = -(cam_true.horizon_line() @ [cam_true.width / 2, 0, 1]) / cam_true.horizon_line()[1]
            hz_e = -(cam_est.horizon_line() @ [cam_est.width / 2, 0, 1]) / cam_est.horizon_line()[1]
            shape, mph = bev_shape_error(cam_true, cam_est, h_true)
            r["truth"] = {"f_px": round(f_true, 1), **{k: round(v_, 2) for k, v_ in ang.items()}, "camera_height_m": round(h_true, 3)}
            r["error"] = {"f_rel": round(cal["camera"]["focal_px"] / f_true - 1, 3),
                          **{k: round(cal["camera"][k] - ang[k], 2) for k in ang}}
            r["roi_x_max_m_if_true_height"] = round(cal["roi_units_h"]["x_max"] * h_true, 1)
            r["geometry_error"] = {"vanishing_point_px": np.round(vpe - vpt, 1).tolist(),
                                   "horizon_row_at_centre_px": round(float(hz_e - hz_t), 1),
                                   "horizon_row_deg": round(float(np.degrees(np.arctan((hz_e - hz_t) / ce["focal_px"]))), 2),
                                   "bev_shape_rms_pct_after_similarity": round(shape, 2),
                                   "fitted_metres_per_h": round(mph, 3), "true_camera_height_m": round(h_true, 3),
                                   "note": "angle errors include the ASSUMED principal point (image centre); the pixel/BEV errors are what the BEV depends on"}
        rows.append(r)
        print(json.dumps({k: r[k] for k in r if k not in ("checks", "focal_diag")}))
        print("   checks:", r["checks"])
        print("   focal:", json.dumps(r["focal_diag"])[:600])
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
