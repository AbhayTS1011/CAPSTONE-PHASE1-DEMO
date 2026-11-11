"""Build continuous CAM_FRONT test videos (keyframes + sweeps, ~12 Hz) from nuScenes v1.0-mini.

The dataset is only read. The ground-truth calibration of each scene is written to a SEPARATE
file under test_videos/ground_truth/ and is used ONLY by tools/evaluate_calibration.py to score the
automatic estimate; the application never reads it.

  python tools/build_test_videos.py --dataroot ../v1.0-mini --out test_videos scene-0796 scene-0061 scene-0103
"""
import argparse, json, os
import cv2, numpy as np


def load(root, name):
    return json.load(open(os.path.join(root, "v1.0-mini", name + ".json"), encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--out", default="test_videos")
    ap.add_argument("scenes", nargs="+")
    a = ap.parse_args()
    scenes = {s["name"]: s for s in load(a.dataroot, "scene")}
    logs = {l["token"]: l for l in load(a.dataroot, "log")}
    samples = {s["token"]: s for s in load(a.dataroot, "sample")}
    sensors = {s["token"]: s for s in load(a.dataroot, "sensor")}
    cal = {c["token"]: c for c in load(a.dataroot, "calibrated_sensor")}
    sd = load(a.dataroot, "sample_data")
    os.makedirs(os.path.join(a.out, "ground_truth"), exist_ok=True)
    for name in a.scenes:
        sc = scenes[name]
        toks = set()
        t = sc["first_sample_token"]
        while t:
            toks.add(t); t = samples[t]["next"]
        cams = [r for r in sd if r["sample_token"] in toks and sensors[cal[r["calibrated_sensor_token"]]["sensor_token"]]["channel"] == "CAM_FRONT"]
        cams.sort(key=lambda r: r["timestamp"])
        ts = np.array([r["timestamp"] for r in cams]) / 1e6
        fps = round((len(ts) - 1) / float(ts[-1] - ts[0]), 2)   # mean capture rate (intervals are irregular)
        loc = logs[sc["log_token"]]["location"]
        out = os.path.join(a.out, f"{name}_{loc}_CAM_FRONT_{len(cams)}f.mp4")
        first = cv2.imread(os.path.join(a.dataroot, cams[0]["filename"]))
        h, w = first.shape[:2]
        vw = None
        for cc in ("avc1", "mp4v"):
            vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*cc), fps, (w, h))
            if vw.isOpened():
                break
        for r in cams:
            vw.write(cv2.imread(os.path.join(a.dataroot, r["filename"])))
        vw.release()
        c = cal[cams[0]["calibrated_sensor_token"]]
        gt = {"scene": name, "location": loc, "frames": len(cams), "fps_written": fps, "codec": cc,
              "camera_intrinsic": c["camera_intrinsic"], "cam_to_ego_rotation_wxyz": c["rotation"],
              "cam_to_ego_translation_m": c["translation"], "image_size": [w, h],
              "note": "Ground truth for EVALUATION ONLY. Never read by the application."}
        json.dump(gt, open(os.path.join(a.out, "ground_truth", os.path.basename(out).replace(".mp4", ".json")), "w"), indent=2)
        print(out, len(cams), "frames", fps, "fps", cc, f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
