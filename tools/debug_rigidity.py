"""Debug: does BEV rigidity of ground tracks prefer the ground-truth pitch/roll/focal? (evaluation only)"""
import sys, os, json, math, numpy as np
sys.path.insert(0, "backend"); sys.path.insert(0, "tools")
from app import calibration as C
from app.video_io import inspect_video
from app.camera_model import CameraModel, unit
from evaluate_calibration import quat_to_R
v = sys.argv[1]
gt = json.load(open(os.path.join("test_videos/ground_truth", os.path.basename(v).replace(".mp4", ".json"))))
info = inspect_video(v); rng = np.random.default_rng(0)
b = C.sample_bursts(v, info); an = sorted(b); W0 = info["resolution"][0]; s = 960 / W0
H, W = b[an[0]][0].shape[:2]
R = quat_to_R(gt["cam_to_ego_rotation_wxyz"]); K = np.array(gt["camera_intrinsic"]) * s; K[2, 2] = 1
mask = np.zeros((H, W), bool); mask[int(0.55 * H):] = True
pairs = []
for a in an:
    seq = b[a]
    for g in range(1, len(seq)):
        tr = C.track_pair(seq[0], seq[g], mask)
        if tr is None: continue
        m = np.median(np.linalg.norm(tr[1] - tr[0], axis=1))
        if m < 3: continue
        if m >= 6 or g == len(seq) - 1: pairs.append(tr); break
M = C.RigidityModel(pairs, W, H)
def cam_rot(f, dp=0, dr=0):
    fwd, left, up = R[0], R[1], R[2]
    def rot(ax, ang):
        ax = unit(ax); a = math.radians(ang)
        return lambda v: v * math.cos(a) + np.cross(ax, v) * math.sin(a) + ax * np.dot(ax, v) * (1 - math.cos(a))
    rp = rot(left, dp); fwd, up = rp(fwd), rp(up)
    rr = rot(fwd, dr); left, up = rr(left), rr(up)
    return CameraModel(W, H, f, K[0, 2], K[1, 2], 0, fwd, left, up)
ct = cam_rot(K[0, 0])
print("pairs", M.select_inliers(ct, 15, rng))
print("truth cost", round(M.cost(ct)[0], 4))
print("pitch offsets:", [(d, round(M.cost(cam_rot(K[0, 0], dp=d))[0], 4)) for d in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)])
print("roll offsets:", [(d, round(M.cost(cam_rot(K[0, 0], dr=d))[0], 4)) for d in (-3, -2, -1, 0, 1, 2, 3)])
print("focal x:", [(x, round(M.cost(cam_rot(K[0, 0] * x))[0], 4)) for x in (0.7, 0.85, 1, 1.15, 1.3, 1.5)])
_, n, rots = M.cost(ct); print("rotation deg per pair:", np.round(np.sort(rots)[-8:], 2))
