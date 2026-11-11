"""Automatic monocular camera / road-plane calibration with traditional computer vision.

Evidence used (all measured from the uploaded video):
  * static-pixel analysis      -> bonnet/hood and burned-in overlays excluded from the ground
  * plumb-line straightness    -> radial distortion k1 (only if measurably better than k1 = 0)
  * LSD line segments + RANSAC -> forward (road-direction) vanishing point, vertical vanishing point
  * pyramidal Lucas-Kanade     -> focus of expansion (cross-check of camera yaw) and ground tracks
  * BEV rigidity of tracks     -> refines the horizon (pitch) and, when the vehicle turns, the focal length

Nothing the video cannot determine is invented: the principal point and zero skew are ASSUMED, the focal
length is ASSUMED (documented prior) unless a measurement constrains it, and absolute scale is never
claimed: the ground plane is expressed in units of the (unknown) camera height.
"""
import math, threading, time
import numpy as np
import cv2

from .camera_model import CameraModel, unit
from .video_io import read_frames_at

ANALYSIS_WIDTH = 960
PRIOR_HFOV_DEG = 70.0        # documented prior; only used when the focal length is not observable
FOV_SEARCH_DEG = (40.0, 120.0)
HORIZON_MARGIN_FRAC = 0.07   # far ROI limit: ground rows at least this far (x image height) below the horizon
BEV_ASPECT = 2.57            # ROI forward length : half width (matches the 560x720 BEV grid)
MAX_ROI_X_H = 13.5           # cap on the forward range (camera heights). ~25 m for a typical ~1.85 m/h dashcam;
#                             keeps the BEV to the near field, where the flat-road IPM and detection are stable.


class CalibrationFailed(Exception):
    pass


def _f_from_hfov(width, hfov):
    return 0.5 * width / math.tan(math.radians(hfov) / 2)


def _hfov_from_f(width, f):
    return math.degrees(2 * math.atan(0.5 * width / f))


# =========================================================================== sampling
def sample_bursts(path, info, n_anchor=40, burst=6):
    n = info["frame_count"]
    n_anchor = max(8, min(n_anchor, n // burst))
    lo, hi = int(0.02 * n), max(int(0.98 * n) - burst, int(0.02 * n) + 1)
    anchors = np.unique(np.linspace(lo, hi, n_anchor).astype(int))
    if n <= 3000:   # short clips: one sequential pass is cheaper than seeking
        need = {a + k: (a, k) for a in anchors for k in range(burst)}
        out = {int(a): [] for a in anchors}
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        i, last = 0, max(need)
        while i <= last:
            if i in need:
                ok, fr = cap.read()
                if not ok:
                    break
                s = ANALYSIS_WIDTH / fr.shape[1]
                if s < 1:
                    fr = cv2.resize(fr, (ANALYSIS_WIDTH, int(round(fr.shape[0] * s))), interpolation=cv2.INTER_AREA)
                out[need[i][0]].append(fr)
            elif not cap.grab():
                break
            i += 1
        cap.release()
        return {a: v for a, v in out.items() if v}
    return read_frames_at(path, anchors, burst=burst, max_width=ANALYSIS_WIDTH)


# =========================================================================== static regions (hood / overlays)
def static_mask(frames):
    small = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (240, 135)).astype(np.float32) for f in frames])
    std = small.std(0)
    m = (std < 5.0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    rows = m.mean(1)
    hood_top = 135
    for r in range(134, 80, -1):           # contiguous block of mostly-static rows at the bottom
        if rows[r] > 0.6:
            hood_top = r
        else:
            break
    H, W = frames[0].shape[:2]
    full = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    hood_row = int(round(hood_top / 135 * H)) if hood_top < 135 else H
    return full, hood_row, float(m.mean())


# =========================================================================== distortion (plumb line)
def estimate_distortion(frames, f0, cx, cy, static):
    contours = []
    for fr in frames[:: max(1, len(frames) // 16)]:
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        e = cv2.Canny(g, 60, 150)
        e[static] = 0
        cs, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            if len(c) >= 0.12 * fr.shape[1]:
                p = c.reshape(-1, 2).astype(np.float64)
                span = np.ptp(p, 0).max()
                if span > 0.1 * fr.shape[1]:
                    contours.append(p[:: max(1, len(p) // 60)])
    ks = np.round(np.arange(-0.30, 0.1001, 0.02), 3)
    if len(contours) < 10:
        return 0.0, {"status": "not_observable", "reason": f"only {len(contours)} long edge chains", "contours": len(contours)}
    def resid(p, k):
        x = (p - [cx, cy]) / f0
        r2 = (x ** 2).sum(1, keepdims=True)
        xu = x * (1 + k * r2)               # first-order inverse of the k1 model (adequate for |k1| <= 0.3)
        xu = xu - xu.mean(0)
        s = np.linalg.svd(xu, compute_uv=False)
        return s[-1] / math.sqrt(len(p)) * f0
    R = np.array([[resid(p, k) for k in ks] for p in contours])
    straight = R.min(1) < 0.8               # chains that are straight lines for SOME k (world-straight edges)
    if straight.sum() < 12:
        return 0.0, {"status": "not_observable", "reason": f"only {int(straight.sum())} straight edge chains", "contours": len(contours)}
    cost = np.median(R[straight], 0)
    kb = float(ks[int(np.argmin(cost))])
    c0 = float(cost[ks == 0][0])
    gain = c0 / max(float(cost.min()), 1e-6)
    info = {"contours": int(straight.sum()), "k1_grid_best": kb, "residual_px_k0": round(c0, 3),
            "residual_px_best": round(float(cost.min()), 3), "improvement_ratio": round(gain, 3)}
    if abs(kb) < 0.02 or gain < 1.30 or kb in (ks[0], ks[-1]):
        info.update(status="not_detected", reason="straightness is not measurably improved by a non-zero k1")
        return 0.0, info
    info.update(status="estimated")
    return kb, info


# =========================================================================== line segments + vanishing points
_TLS = threading.local()        # OpenCV's LSD object is not thread-safe: one detector per worker thread


def segments(gray, static):
    lsd = getattr(_TLS, "lsd", None)
    if lsd is None:
        lsd = _TLS.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(gray)[0]
    if lines is None:
        return np.zeros((0, 4))
    s = lines.reshape(-1, 4).astype(np.float64)
    mid = ((s[:, :2] + s[:, 2:]) / 2).astype(int)
    H, W = gray.shape
    mid[:, 0] = mid[:, 0].clip(0, W - 1); mid[:, 1] = mid[:, 1].clip(0, H - 1)
    return s[~static[mid[:, 1], mid[:, 0]]]


def _seg_props(s):
    d = s[:, 2:] - s[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    ang = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180          # 0..180, image y down
    mid = (s[:, :2] + s[:, 2:]) / 2
    hom = np.cross(np.column_stack([s[:, :2], np.ones(len(s))]), np.column_stack([s[:, 2:], np.ones(len(s))]))
    return L, ang, mid, hom, d / L[:, None]


def _angle_err(vp, mid, dirs):
    """Angle (deg) between each segment direction and the direction from its midpoint to homogeneous vp."""
    if abs(vp[2]) < 1e-9:
        to = np.tile(vp[:2], (len(mid), 1))
    else:
        to = vp[:2] / vp[2] - mid
    to = to / (np.linalg.norm(to, axis=1, keepdims=True) + 1e-12)
    c = np.abs((to * dirs).sum(1)).clip(0, 1)
    return np.degrees(np.arccos(c))


def ransac_vp(s, rng, iters=300, thr=2.0):
    L, ang, mid, hom, dirs = _seg_props(s)
    if len(s) < 4:
        return None
    p = L / L.sum()
    best, best_score = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(s), 2, replace=False, p=p)
        vp = np.cross(hom[i], hom[j])
        if not np.isfinite(vp).all() or np.linalg.norm(vp) < 1e-12:
            continue
        vp = vp / np.linalg.norm(vp)
        e = _angle_err(vp, mid, dirs)
        score = (L * np.clip(1 - e / thr, 0, None)).sum()
        if score > best_score:
            best, best_score = vp, score
    if best is None:
        return None
    inl = _angle_err(best, mid, dirs) < thr
    # least-squares refinement: vp minimising sum of squared (normalised) line residuals of inliers
    A = hom[inl] / np.linalg.norm(hom[inl][:, :2], axis=1, keepdims=True) * np.sqrt(L[inl])[:, None]
    if len(A) >= 2:
        vp = np.linalg.svd(A)[2][-1]
        best = vp / np.linalg.norm(vp)
        inl = _angle_err(best, mid, dirs) < thr
    return best, inl, L, ang


def forward_vp(s, W, H, rng):
    L, ang, mid, _, _ = _seg_props(s)
    a = np.minimum(ang, 180 - ang)                       # angle from horizontal, 0..90
    keep = (a > 12) & (a < 80) & (L > 0.025 * W) & (mid[:, 1] > 0.35 * H)
    if keep.sum() < 6:
        return None
    r = ransac_vp(s[keep], rng)
    if r is None:
        return None
    vp, inl, L2, ang2 = r
    if abs(vp[2]) < 1e-9:
        return None
    x, y = vp[:2] / vp[2]
    slope_neg = (ang2 < 90) & inl                        # image y down: ang<90 is a '\' segment
    slope_pos = (ang2 > 90) & inl
    wl, wr = L2[slope_neg].sum(), L2[slope_pos].sum()
    tot = wl + wr
    if inl.sum() < 4 or tot == 0 or min(wl, wr) / tot < 0.15:
        return None
    if not (-0.5 * W < x < 1.5 * W and -0.3 * H < y < 0.85 * H):
        return None
    return {"x": float(x), "y": float(y), "inliers": int(inl.sum()), "support": float(tot)}


def vertical_vp(s, W, H, rng):
    L, ang, mid, _, _ = _seg_props(s)
    keep = (np.abs(ang - 90) < 12) & (L > 0.03 * H)
    if keep.sum() < 6:
        return None
    r = ransac_vp(s[keep], rng, thr=1.5)
    if r is None:
        return None
    vp, inl, _, _ = r
    if inl.sum() < 6:
        return None
    if vp[2] < 0 or (abs(vp[2]) < 1e-12 and vp[1] < 0):
        vp = -vp
    return {"vp": vp.tolist(), "inliers": int(inl.sum())}


# =========================================================================== optical flow
def track_pair(f0, f1, mask):
    g0, g1 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(g0, 500, 0.005, 6, mask=mask.astype(np.uint8) * 255, blockSize=7)
    if p0 is None or len(p0) < 20:
        return None
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
    pb, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
    ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1) < 0.8)
    if ok.sum() < 20:
        return None
    return p0.reshape(-1, 2)[ok].astype(np.float64), p1.reshape(-1, 2)[ok].astype(np.float64)


def focus_of_expansion(p0, p1, rng, iters=200, thr_deg=3.0):
    d = p1 - p0
    m = np.linalg.norm(d, axis=1)
    keep = m > 1.5
    if keep.sum() < 15:
        return None
    p0, d, m = p0[keep], d[keep], m[keep]
    dirs = d / m[:, None]
    hom = np.cross(np.column_stack([p0, np.ones(len(p0))]), np.column_stack([p0 + d, np.ones(len(p0))]))
    best, bs = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(p0), 2, replace=False)
        e = np.cross(hom[i], hom[j])
        if abs(e[2]) < 1e-9:
            continue
        e = e / e[2]
        to = p0 - e[:2]                                       # expansion: flow points away from FOE
        to /= np.linalg.norm(to, axis=1, keepdims=True) + 1e-12
        cosang = (to * dirs).sum(1)
        score = (cosang > math.cos(math.radians(thr_deg))).sum()
        if score > bs:
            best, bs = e[:2], score
    if best is None or bs < 0.4 * len(p0):
        return None
    return {"x": float(best[0]), "y": float(best[1]), "inlier_ratio": float(bs / len(p0))}


# =========================================================================== rigidity of ground motion in the BEV
def _rigid_fit(a, b):
    ca, cb = a.mean(0), b.mean(0)
    Hm = (a - ca).T @ (b - cb)
    U, _, Vt = np.linalg.svd(Hm)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    t = cb - R @ ca
    return R, t


def _rigid_ransac(a, b, rng, thr, iters=150):
    best = None
    n = len(a)
    for _ in range(iters):
        i = rng.choice(n, 2, replace=False)
        if np.linalg.norm(a[i[0]] - a[i[1]]) < 1e-3:
            continue
        R, t = _rigid_fit(a[i], b[i])
        e = np.linalg.norm(a @ R.T + t - b, axis=1)
        inl = e < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    return best


class RigidityModel:
    """Scores a camera hypothesis by how rigid the ground-track motion is after mapping to the ground plane."""

    def __init__(self, pairs, W, H):
        self.pairs, self.W, self.H = pairs, W, H

    def camera(self, f, k1_px, vp, dpitch=0.0, droll=0.0):
        """Camera looking along the road VP with zero roll, then rotated by dpitch (about 'left') and droll (about 'fwd')."""
        cx, cy = self.W / 2, self.H / 2
        cam = CameraModel.from_vanishing(self.W, self.H, f, cx, cy, k1_px * f * f, vp, up_vector(f, cx, cy, vp, (0.0, 1.0)))
        return rotate_camera(cam, dpitch, droll)

    def select_inliers(self, cam, x_eval, rng):
        self.sel = []
        for p0, p1 in self.pairs:
            g0, v0 = cam.pixel_to_ground(p0)
            g1, v1 = cam.pixel_to_ground(p1)
            ok = v0 & v1 & (g0[:, 0] > 0.5) & (g0[:, 0] < x_eval) & (np.abs(g0[:, 1]) < 0.6 * x_eval)
            if ok.sum() < 12:
                self.sel.append(None); continue
            disp = np.median(np.linalg.norm(g1[ok] - g0[ok], axis=1))
            inl = _rigid_ransac(g0[ok], g1[ok], rng, thr=max(0.08 * disp, 0.02))
            idx = np.flatnonzero(ok)[inl] if inl is not None else None
            self.sel.append(idx if idx is not None and len(idx) >= 10 else None)
        return sum(s is not None for s in self.sel)

    def cost(self, cam, min_rot=None):
        vals, rots = [], []
        for (p0, p1), idx in zip(self.pairs, self.sel):
            if idx is None:
                continue
            g0, v0 = cam.pixel_to_ground(p0[idx])
            g1, v1 = cam.pixel_to_ground(p1[idx])
            ok = v0 & v1
            if ok.sum() < 8:
                vals.append(np.nan); rots.append(0); continue
            a, b = g0[ok], g1[ok]
            R, t = _rigid_fit(a, b)
            e = np.linalg.norm(a @ R.T + t - b, axis=1)
            keep = e <= np.quantile(e, 0.8)                  # trimmed refit
            R, t = _rigid_fit(a[keep], b[keep])
            e = np.linalg.norm(a[keep] @ R.T + t - b[keep], axis=1)
            disp = np.mean(np.linalg.norm(b[keep] - a[keep], axis=1))
            vals.append(float(np.sqrt((e ** 2).mean()) / max(disp, 1e-6)))
            rots.append(abs(math.degrees(math.atan2(R[1, 0], R[0, 0]))))
        vals, rots = np.array(vals), np.array(rots)
        if min_rot is not None:
            vals = vals[rots > min_rot]
        vals = vals[np.isfinite(vals)]
        return (float(np.median(vals)) if len(vals) else np.nan), len(vals), rots


def _rodrigues(axis, deg):
    a = math.radians(deg); k = unit(axis)
    return lambda v: v * math.cos(a) + np.cross(k, v) * math.sin(a) + k * np.dot(k, v) * (1 - math.cos(a))


def rotate_camera(cam, dpitch, droll):
    """Rotate the ground frame relative to the camera: +dpitch tilts the road direction up, +droll rotates about it."""
    fwd, left, up = cam.fwd, cam.left, cam.up
    if dpitch:
        r = _rodrigues(left, -dpitch); fwd, up = r(fwd), r(up)
    if droll:
        r = _rodrigues(fwd, droll); left, up = r(left), r(up)
    return CameraModel(cam.width, cam.height, cam.f, cam.cx, cam.cy, cam.k1, unit(fwd), unit(left), unit(up))


def _parabolic(xs, ys, i):
    a, b, c = ys[i - 1], ys[i], ys[i + 1]
    den = a - 2 * b + c
    return xs[i] + (0.5 * (a - c) / den) * (xs[1] - xs[0]) if den > 0 else xs[i]


def up_vector(f, cx, cy, vp, roll_vec):
    """Ground normal (up) from the forward VP and the image direction of verticals (roll).
    roll_vec = (sin r, cos r): image direction of world-vertical lines near the image centre."""
    hdir = np.array([roll_vec[1], -roll_vec[0]])                  # horizon direction is perpendicular to verticals
    p1 = np.array([vp[0], vp[1], 1.0]); p2 = np.array([vp[0] + hdir[0], vp[1] + hdir[1], 1.0])
    return CameraModel.up_from_horizon(f, cx, cy, np.cross(p1, p2))


# =========================================================================== ROI / visibility
def choose_roi(cam):
    """Far limit: the forward distance whose image row is HORIZON_MARGIN_FRAC*H below the horizon."""
    hl = cam.horizon_line()
    v_h = -(hl[0] * cam.cx + hl[2]) / hl[1]
    target = v_h + HORIZON_MARGIN_FRAC * cam.height
    lo, hi = 0.5, 400.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        uv, z = cam.project_ground(np.array([mid]), np.array([0.0]))
        if z[0] > 0 and uv[0, 1] > target:
            lo = mid
        else:
            hi = mid
    x_max = float(min(lo, MAX_ROI_X_H))        # cap the far limit at the near field (steadier, less IPM stretch)
    return {"x_min": 0.0, "x_max": x_max, "y_max": x_max / BEV_ASPECT, "horizon_row_px": float(v_h)}


# =========================================================================== main entry
def calibrate(path, info, progress=lambda stage, detail="": None, seed=0):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    W0, H0 = info["resolution"]
    s = min(1.0, ANALYSIS_WIDTH / W0)
    progress("sampling", "decoding sample frames across the video")
    bursts = sample_bursts(path, info)
    anchors = sorted(bursts)
    if len(anchors) < 6:
        raise CalibrationFailed("Too few frames could be decoded for calibration.")
    firsts = [bursts[a][0] for a in anchors]
    H, W = firsts[0].shape[:2]
    cx, cy = W / 2, H / 2
    f_prior = _f_from_hfov(W, PRIOR_HFOV_DEG)
    checks, warnings = [], []

    def check(name, value, ok, detail, warn_only=True):
        checks.append({"name": name, "value": value, "status": "pass" if ok else ("warn" if warn_only else "fail"), "detail": detail})
        return ok

    progress("analysing_geometry", "static regions (bonnet / overlays) and lens straightness")
    static, hood_row, static_frac = static_mask(firsts)
    k1_grid, dist_info = estimate_distortion(firsts, f_prior, cx, cy, static)
    k1_px = k1_grid / (f_prior ** 2)
    def undist(fr):
        if k1_px == 0:
            return fr
        return cv2.undistort(fr, np.array([[f_prior, 0, cx], [0, f_prior, cy], [0, 0, 1]]), np.array([k1_grid, 0, 0, 0, 0]))

    progress("vanishing_points", f"line segments and RANSAC vanishing points in {len(anchors)} sampled frames")
    fvps, vvps = [], []
    for a, fr in zip(anchors, firsts):
        g = cv2.cvtColor(undist(fr), cv2.COLOR_BGR2GRAY)
        seg = segments(g, static)
        fv = forward_vp(seg, W, H, rng)
        if fv:
            fv["anchor"] = int(a); fvps.append(fv)
        vv = vertical_vp(seg, W, H, rng)
        if vv:
            vvps.append(vv)
    rate = len(fvps) / len(anchors)
    if len(fvps) < max(3, 0.2 * len(anchors)):
        raise CalibrationFailed(f"The road-direction vanishing point was found in only {len(fvps)} of {len(anchors)} sampled frames; "
                                "the road lines are not visible or not consistent enough to estimate the camera orientation.")
    def aggregate(sel):
        xy = np.array([[v["x"], v["y"]] for v in sel])
        med = np.median(xy, 0)
        mad = np.median(np.abs(xy - med), 0) * 1.4826
        near = np.linalg.norm(xy - med, axis=1) < max(3 * np.linalg.norm(mad), 0.03 * W)
        vp = xy[near].mean(0) if near.sum() >= 3 else med
        half = len(sel) // 2
        vp_a = np.median(xy[:half], 0) if half >= 2 else vp
        vp_b = np.median(xy[half:], 0) if len(sel) - half >= 2 else vp
        return vp, mad, vp_a, vp_b

    vp, mad, vp_a, vp_b = aggregate(fvps)
    vp_selection = f"all {len(fvps)} frames with a road vanishing point"

    # vertical vanishing directions: reported as a diagnostic (tree trunks / leaning poles bias them by degrees,
    # so roll is NOT taken from them; see the rigidity-based roll test below)
    roll_spread, roll_vertical = None, None
    if len(vvps) >= 4:
        dirs = []
        for v in vvps:
            p = np.array(v["vp"])
            d = p[:2] - np.array([cx, cy]) * p[2]
            d = d / (np.linalg.norm(d) + 1e-12)
            dirs.append(d if d[1] > 0 else -d)
        dirs = np.array(dirs)
        ang = np.degrees(np.arctan2(dirs[:, 0], dirs[:, 1]))
        roll_vertical = float(np.median(ang)); roll_spread = float(np.median(np.abs(ang - roll_vertical)) * 1.4826)

    progress("camera_orientation", "optical-flow focus of expansion and ground tracks")
    road_mask = np.zeros((H, W), bool)
    road_mask[int(min(max(vp[1] + 0.06 * H, 0), H - 1)):min(hood_row, H)] = True
    road_mask &= ~static
    pairs, foes = [], []
    for a in anchors:
        seq = bursts[a]
        for g in range(1, len(seq)):
            tr = track_pair(seq[0], seq[g], road_mask)      # raw pixels: the camera model applies k1 itself
            if tr is None:
                continue
            mflow = float(np.median(np.linalg.norm(tr[1] - tr[0], axis=1)))
            if mflow < 3.0:
                continue                                         # repeated / static frames: no information
            if mflow > 45 and g > 1:
                break
            if mflow >= 6.0 or g == len(seq) - 1:
                pairs.append(tr)
                fo = focus_of_expansion(tr[0], tr[1], rng)
                if fo:
                    foes.append({**fo, "anchor": int(a)})
                break

    # Prefer frames where the vehicle is demonstrably travelling ALONG the road: the optical-flow focus of
    # expansion and the road vanishing point agree.  This drops frames dominated by cross-street lines while
    # the vehicle waits or turns (they otherwise bias the yaw by many degrees).
    foe_by_anchor = {fo["anchor"]: fo for fo in foes}
    yaw_of = lambda x: math.degrees(math.atan((x - cx) / f_prior))
    agree = [v for v in fvps if v["anchor"] in foe_by_anchor and abs(yaw_of(v["x"]) - yaw_of(foe_by_anchor[v["anchor"]]["x"])) < 3.0]
    if len(agree) >= max(5, 0.25 * len(fvps)) and len(agree) < len(fvps):
        vp, mad, vp_a, vp_b = aggregate(agree)
        vp_selection = (f"{len(agree)} of {len(fvps)} frames in which the optical-flow focus of expansion agrees with the "
                        "road vanishing point (vehicle travelling along the road)")

    progress("road_plane", f"ground-plane rigidity over {len(pairs)} tracked frame pairs")
    model = RigidityModel(pairs, W, H)
    f_est, f_src, f_detail = f_prior, "assumed", {"prior_hfov_deg": PRIOR_HFOV_DEG}
    # (a) orthogonality of forward and vertical vanishing directions -- only well conditioned when the
    #     camera is clearly pitched (vertical VP finite and the forward VP well away from the principal point)
    f_orth = []
    for v in vvps:
        p = np.array(v["vp"])
        if abs(p[2]) > 1e-9:
            x3, y3 = p[:2] / p[2]
            d3 = np.array([x3 - cx, y3 - cy]); n3 = np.linalg.norm(d3)
            lever = abs(np.dot(vp - [cx, cy], d3 / (n3 + 1e-12)))
            if lever < 0.05 * H or n3 > 30 * f_prior:
                continue
            f2 = -np.dot(vp - [cx, cy], d3)
            if f2 > 0:
                f_orth.append(math.sqrt(f2))
    f_detail["orthogonal_vp"] = {"usable_frames": len(f_orth)}
    if len(f_orth) >= 5:
        q1, q2, q3 = np.percentile(f_orth, [25, 50, 75])
        f_detail["orthogonal_vp"].update(median_px=round(float(q2) / s, 1), iqr_rel=round(float((q3 - q1) / q2), 3))
        lo_f, hi_f = _f_from_hfov(W, FOV_SEARCH_DEG[1]), _f_from_hfov(W, FOV_SEARCH_DEG[0])
        if (q3 - q1) / q2 < 0.15 and lo_f < q2 < hi_f:
            f_est, f_src = float(q2), "estimated (orthogonal vanishing points)"
    else:
        f_detail["orthogonal_vp"]["note"] = ("camera pitch too small: the vertical vanishing point is (near) infinity, "
                                             "so orthogonality does not constrain the focal length")

    dp, dr, roll_src = 0.0, 0.0, "assumed"
    rig0 = rig_best = np.nan
    cam0 = model.camera(f_est, k1_px, vp)
    roi0 = choose_roi(cam0)
    n_sel = model.select_inliers(cam0, roi0["x_max"] * 0.8, rng) if pairs else 0
    pitch_detail = {"pairs_used": n_sel, "refined": False}
    roll_detail = {"vertical_vp_roll_deg": None if roll_vertical is None else round(roll_vertical, 2),
                   "vertical_vp_spread_deg": None if roll_spread is None else round(roll_spread, 2)}
    if n_sel >= 5:
        rig0 = model.cost(cam0)[0]
        # pitch: wrong pitch makes near and far ground points move by different amounts -> non-rigid BEV motion
        dps = np.arange(-5.0, 5.01, 0.5)
        pc = np.array([model.cost(model.camera(f_est, k1_px, vp, d))[0] for d in dps])
        i = int(np.nanargmin(pc))
        pitch_detail.update(dpitch_grid_deg=dps.round(2).tolist(), costs=[round(float(c), 4) for c in pc])
        if 0 < i < len(dps) - 1 and min(pc[i - 1], pc[i + 1]) > 1.03 * pc[i]:
            dp = float(_parabolic(dps, pc, i)); pitch_detail["refined"] = True
        pitch_detail["dpitch_deg"] = round(dp, 3)
        # roll: accepted only when the rigidity cost rises clearly on BOTH sides of its minimum
        drs = np.arange(-4.0, 4.01, 1.0)
        rc = np.array([model.cost(model.camera(f_est, k1_px, vp, dp, d))[0] for d in drs])
        j = int(np.nanargmin(rc))
        roll_detail.update(droll_grid_deg=drs.tolist(), costs=[round(float(c), 4) for c in rc])
        if 1 < j < len(drs) - 2 and rc[j - 2] > 1.20 * rc[j] and rc[j + 2] > 1.20 * rc[j]:
            dr = float(_parabolic(drs, rc, j)); roll_src = "estimated"
        roll_detail["droll_deg"] = round(dr, 3)
        # (b) focal from rigidity during clear turns (a wrong focal makes rotating ground motion non-rigid)
        fs = [_f_from_hfov(W, hv) for hv in np.arange(FOV_SEARCH_DEG[0], FOV_SEARCH_DEG[1] + 0.1, 5.0)]
        fc = [model.cost(model.camera(fv, k1_px, vp, dp, dr), min_rot=3.0) for fv in fs]
        cvals = np.array([c[0] for c in fc]); nturn = max(c[1] for c in fc)
        f_detail["rigidity_turns"] = {"turning_pairs_gt_3deg": int(nturn), "hfov_grid_deg": [round(_hfov_from_f(W, x), 1) for x in fs],
                                      "costs": [None if not np.isfinite(c) else round(float(c), 4) for c in cvals]}
        if f_src == "assumed" and nturn >= 5 and np.isfinite(cvals).all():
            j = int(np.argmin(cvals))
            if 3 <= j <= len(fs) - 4 and cvals[j - 3] > 1.2 * cvals[j] and cvals[j + 3] > 1.2 * cvals[j]:
                hf = float(_parabolic([_hfov_from_f(W, x) for x in fs], cvals, j))
                f_est, f_src = _f_from_hfov(W, hf), "estimated (BEV rigidity of ground motion during turns)"
            else:
                f_detail["rigidity_turns"]["note"] = "cost minimum not sharp enough: focal not observable"
        elif f_src == "assumed":
            f_detail["rigidity_turns"]["note"] = "fewer than 5 frame pairs with > 3 deg of turning: focal not observable"
        rig_best = model.cost(model.camera(f_est, k1_px, vp, dp, dr))[0]
    else:
        pitch_detail["note"] = "not enough moving ground tracks to refine the horizon"

    cam = model.camera(f_est, k1_px, vp, dp, dr)
    progress("validating", "homography validity, visible ground area and stability checks")
    roi = choose_roi(cam)

    # ---------------- validation (all measured) ----------------
    check("vanishing_point_detection_rate", round(rate, 3), rate >= 0.4,
          f"forward vanishing point found in {len(fvps)}/{len(anchors)} sampled frames")
    disp_deg = math.degrees(math.atan(float(np.linalg.norm(mad)) / f_est))
    check("vanishing_point_dispersion_deg", round(disp_deg, 2), disp_deg < 3.0,
          "robust spread of per-frame vanishing points (turns and lane changes increase it)")
    stab = math.degrees(math.atan(float(np.linalg.norm(vp_a - vp_b)) / f_est))
    check("temporal_stability_deg", round(stab, 2), stab < 2.0, "first-half vs second-half vanishing point difference")
    if foes:
        fx = np.median([f["x"] for f in foes])
        yaw_diff = abs(math.degrees(math.atan((fx - cx) / f_est) - math.atan((vp[0] - cx) / f_est)))
        check("flow_foe_vs_vanishing_point_deg", round(yaw_diff, 2), yaw_diff < 3.0,
              f"optical-flow focus of expansion vs road vanishing point (yaw), {len(foes)} frame pairs")
    else:
        check("flow_foe_vs_vanishing_point_deg", None, False, "no reliable focus of expansion (vehicle stationary or too little texture)")
    v_h = roi["horizon_row_px"]
    check("horizon_inside_plausible_band", round(v_h / H, 3), -0.2 * H < v_h < 0.8 * H, "horizon row / image height", warn_only=False)
    Hg = cam.ground_homography()
    Kn = np.diag([1 / f_est, 1 / f_est, 1.0]); Hn = Kn @ Hg
    cond = float(np.linalg.cond(Hn)); det = float(np.linalg.det(Hn))
    check("homography_conditioning", round(cond, 2), np.isfinite(cond) and cond < 1e4 and abs(det) > 1e-9,
          "condition number of the normalised ground homography", warn_only=False)
    if np.isfinite(rig_best):
        check("ground_motion_rigidity", round(float(rig_best), 4), rig_best < 0.15,
              "median non-rigid residual of tracked ground points in the BEV / displacement (0 = perfectly planar & rigid)")
    else:
        check("ground_motion_rigidity", None, False, "not enough moving ground tracks to test the road-plane hypothesis")
    # visible ground inside the ROI
    gx = np.linspace(roi["x_min"], roi["x_max"], 120); gy = np.linspace(-roi["y_max"], roi["y_max"], 80)
    GX, GY = np.meshgrid(gx, gy)
    uv, z = cam.project_ground(GX, GY)
    vis = (z > 0) & (uv[..., 0] >= 0) & (uv[..., 0] < W) & (uv[..., 1] >= 0) & (uv[..., 1] < min(H, hood_row))
    inb = np.zeros_like(vis)
    ui = uv[..., 0].clip(0, W - 1).astype(int); vi = uv[..., 1].clip(0, H - 1).astype(int)
    inb[vis] = ~static[vi[vis], ui[vis]]
    vis_frac = float(inb.mean())
    check("visible_ground_fraction_of_roi", round(vis_frac, 3), vis_frac > 0.15, "share of the BEV ROI observed by the camera", warn_only=False)

    fails = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    if fails:
        status = "failed"
    else:
        # independent cross-checks that disagree strongly (or could not be measured at all) lower the confidence
        severe_limits = {"flow_foe_vs_vanishing_point_deg": 6.0, "ground_motion_rigidity": 0.30}
        severe = [c for c in warns if c["name"] in severe_limits and c["value"] is not None and c["value"] > severe_limits[c["name"]]]
        unvalidated = all(c["value"] is None for c in checks if c["name"] in severe_limits)
        conf_reasons = []
        if len(warns) >= 3:
            conf_reasons.append(f"{len(warns)} quality checks below threshold")
        for c in severe:
            conf_reasons.append(f"'{c['name']}' = {c['value']} is far beyond its threshold (independent estimates disagree)")
        if unvalidated:
            conf_reasons.append("the vehicle hardly moves, so the orientation could not be cross-checked with optical flow")
        status = "low_confidence" if conf_reasons else "ok"
        for r in conf_reasons:
            warnings.append("Low confidence: " + r + ".")
    if f_src == "assumed":
        warnings.append(f"Focal length not observable from this video (no usable orthogonal vanishing points and too little turning). "
                        f"A documented prior of {PRIOR_HFOV_DEG:.0f} deg horizontal field of view is ASSUMED; the BEV's forward/lateral aspect depends on it.")
    if roll_src == "assumed":
        warnings.append("Camera roll is not clearly constrained by the ground-motion measurements; zero roll is ASSUMED.")
    warnings.append("Relative scale only: no physical reference is available, so distances are in units of the (unknown) camera height, not metres.")
    for c in warns:
        warnings.append(f"Check '{c['name']}' did not meet its threshold ({c['detail']}).")

    # ---------------- scale back to full resolution ----------------
    sc = 1.0 / s
    cam_full = CameraModel(W0, H0, f_est * sc, cx * sc, cy * sc, cam.k1, cam.fwd, cam.left, cam.up)
    vp_final = cam_full.K @ cam_full.fwd
    vp_final = vp_final[:2] / vp_final[2]
    ang = cam_full.angles()
    params = [
        {"name": "Focal length", "value": round(cam_full.f, 1), "unit": "px",
         "extra": f"HFOV {_hfov_from_f(W0, cam_full.f):.1f} deg", "source": "estimated" if f_src != "assumed" else "assumed", "method": f_src},
        {"name": "Principal point", "value": [round(cam_full.cx, 1), round(cam_full.cy, 1)], "unit": "px",
         "source": "assumed", "method": "image centre (not observable without a calibration target)"},
        {"name": "Skew / pixel aspect", "value": "0 / 1", "unit": "", "source": "assumed", "method": "standard digital sensor assumption"},
        {"name": "Radial distortion k1", "value": round(cam_full.k1, 4), "unit": "",
         "source": "estimated" if dist_info.get("status") == "estimated" else "assumed",
         "method": "plumb-line straightness of long edge chains" + ("" if dist_info.get("status") == "estimated" else f" -> {dist_info.get('status')}, k1 = 0 used")},
        {"name": "Yaw (road direction)", "value": round(ang["yaw_deg"], 2), "unit": "deg", "source": "estimated",
         "method": "robust median of per-frame road vanishing points (LSD + RANSAC)"},
        {"name": "Pitch", "value": round(ang["pitch_deg"], 2), "unit": "deg", "source": "estimated",
         "method": "road vanishing point" + (f" refined by ground-motion rigidity ({dp:+.2f} deg)" if pitch_detail.get("refined") else "")},
        {"name": "Roll", "value": round(ang["roll_deg"], 2), "unit": "deg", "source": roll_src,
         "method": "ground-motion rigidity minimum" if roll_src == "estimated" else "not constrained by the measurements -> 0"},
        {"name": "Camera height", "value": 1.0, "unit": "h", "source": "assumed",
         "method": "unknown; used as the unit of length (relative scale)"},
    ]
    return {
        "status": status,
        "scale_mode": "relative",
        "scale_note": "Distances are expressed in camera heights (h). Absolute metres are not established.",
        "method": "LSD line segments + RANSAC vanishing points, plumb-line distortion test, LK optical flow, BEV rigidity refinement",
        "camera": cam_full.to_dict(),
        "parameters": params,
        "checks": checks,
        "warnings": warnings,
        "roi_units_h": {k: round(v, 3) for k, v in roi.items() if k != "horizon_row_px"},
        "horizon_row_px": round(roi["horizon_row_px"] * sc, 1),
        "hood_top_row_px": int(round(hood_row * sc)) if hood_row < H else None,
        "static_pixel_fraction": round(static_frac, 4),
        "static_mask_small": cv2.resize(static.astype(np.uint8), (240, 135), interpolation=cv2.INTER_NEAREST).tolist(),
        "diagnostics": {
            "sampled_frames": len(anchors), "forward_vp_frames": len(fvps), "vertical_vp_frames": len(vvps),
            "flow_pairs": len(pairs), "foe_pairs": len(foes),
            "vanishing_point_px": [round(float(vp_final[0]), 1), round(float(vp_final[1]), 1)],
            "vanishing_point_frame_selection": vp_selection,
            "vanishing_point_line_based_px": [round(float(vp[0] * sc), 1), round(float(vp[1] * sc), 1)],
            "roll": roll_detail,
            "distortion": dist_info, "focal": f_detail, "pitch_refinement": pitch_detail,
            "rigidity_initial": None if not np.isfinite(rig0) else round(float(rig0), 4),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
