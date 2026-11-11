"""Temporally tracked vectorized BEV (traditional CV, video-derived).

Per frame:
  1. ego-motion: Lucas-Kanade tracks on the road surface between the previous and current frame are mapped to
     the ground plane; a RANSAC rigid 2-D fit gives how the ground moved relative to the camera.
  2. evidence fusion: two BEV evidence maps, painted-marking evidence and road-surface evidence, are warped by
     that motion and blended with the current frame's evidence (exponential memory of a few frames). Dashed
     lines and intermittent detections become continuous and stable; flicker is suppressed.
  3. extraction, inside the CURRENTLY visible ground only:
       road_boundary       per BEV row, walk outwards from the vehicle's lane until the road surface ends;
                           left/right edge points are median-filtered and fitted with smooth curves
       lane_divider        column histogram of the fused paint -> sliding windows along the road -> smooth curve;
                           painted edge lines beside a boundary are treated as part of the boundary
       pedestrian_crossing stripe groups (per frame, see vectorize.py)
Nothing is taken from maps or poses, and nothing is drawn outside the ground the camera sees in the current frame.
"""
import math
import numpy as np
import cv2

from .vectorize import Vectorizer, _length

ALPHA = 0.8               # evidence memory per frame (about 5 frames)
PAINT_ON = 0.30           # fused paint evidence needed for a paint pixel
ROAD_ON = 0.55            # fused road-surface evidence needed for a road pixel
MIN_DIVIDER_H = 1.2
MIN_BOUNDARY_H = 2.5
EDGE_LINE_GAP_H = 0.7     # paint closer than this to a road boundary is an edge line (part of the boundary)
FLOW_SCALE = 0.5          # ego-motion is measured on a half-resolution grey image

# ---- line-level temporal tracking (smooth frame-to-frame output) ----
TRACK_GRID_H = 0.4        # a tracked line is stored as Y sampled every this many h of forward distance X
TRACK_BLEND = 0.20        # per-frame weight of a new detection against the carried-over line (lower = steadier)
TRACK_MATCH_H = 0.9       # a detection matches a track if their median lateral gap over the overlap is below this
TRACK_MIN_AGE = 2         # a track is drawn only after it has been confirmed in this many frames
TRACK_MAX_MISS = 8        # a track is dropped after this many frames without a matching detection
TRACK_ENDS_BLEND = 0.35   # update of a line's near/far ends, where new ground is revealed


def _smooth_run(p, win):
    if len(p) < 3 or win < 3:
        return p
    k = np.ones(win) / win
    pad = win // 2
    q = np.pad(p, ((pad, pad), (0, 0)), mode="edge")
    return np.column_stack([np.convolve(q[:, 0], k, "valid"), np.convolve(q[:, 1], k, "valid")])[:len(p)]


def _fit_curve(rows, cols, step, res):
    """Smooth curve col(row): quadratic (linear when short) if it fits within 0.15 h, else a smoothed polyline."""
    deg = 2 if (rows.max() - rows.min()) * res > 5 else 1
    coef = np.polyfit(rows, cols, deg)
    r = np.abs(np.polyval(coef, rows) - cols)
    keep = r < max(2.5 * np.median(r), 2.0)
    if keep.sum() >= deg + 2:
        coef = np.polyfit(rows[keep], cols[keep], deg)
    if float(np.median(np.abs(np.polyval(coef, rows) - cols))) * res < 0.15:
        fine = np.arange(rows.min(), rows.max() + 1e-6, step)
        return np.column_stack([np.polyval(coef, fine), fine])
    order = np.argsort(rows)
    return _smooth_run(np.column_stack([cols[order], rows[order]]), 7)


class TrackedVectorizer:
    def __init__(self, projector, cam):
        self.P, self.cam = projector, cam
        self.vz = Vectorizer(projector)
        P = projector
        res = P.res
        self.A = np.array([[0.0, -res], [-res, 0.0]])                 # BEV px (col,row) -> ground (X,Y)
        self.c = np.array([P.roi["x_max"] - 0.5 * res, P.y_max - 0.5 * res])
        self.Ainv = np.linalg.inv(self.A)
        self.valid = projector.valid
        H, W = cam.height, cam.width
        m = np.zeros((H, W), np.uint8)                              # image pixels that see valid ground
        ok = projector.valid
        m[projector.map_y[ok].astype(int).clip(0, H - 1), projector.map_x[ok].astype(int).clip(0, W - 1)] = 1
        m = cv2.dilate(m, np.ones((15, 15), np.uint8))
        self.track_mask = cv2.resize(m, (int(W * FLOW_SCALE), int(H * FLOW_SCALE)), interpolation=cv2.INTER_NEAREST)
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.paint = np.zeros(self.valid.shape, np.float32)
        self.yellow = np.zeros(self.valid.shape, np.float32)
        self.road = np.zeros(self.valid.shape, np.float32)
        self.edge = np.zeros(self.valid.shape, np.float32)
        self.cur_paint = np.zeros(self.valid.shape, bool)
        self.last_motion, self.coasted = None, 0
        self.tracks = []                                        # persistent, temporally smoothed output lines
        self._next_id = 0
        self.frames = 0
        self.Xg = np.arange(0.0, self.P.roi["x_max"] + 1e-6, TRACK_GRID_H)

    # ------------------------------------------------------------------ ego-motion
    def _ego_motion(self, gray):
        """Returns ((R, t), info) with ground coords g_now = R g_prev + t, or (None, info) if unknown."""
        static = (np.eye(2), np.zeros(2))
        if self.prev_gray is None:
            return None, {"ego_motion": "first frame"}
        p0 = cv2.goodFeaturesToTrack(self.prev_gray, 400, 0.01, 6, mask=self.track_mask, blockSize=7)
        if p0 is None or len(p0) < 15:
            return static, {"ego_motion": "too few features (assumed static)"}
        lk = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None, **lk)
        ok = st.ravel() == 1
        p0, p1 = p0.reshape(-1, 2)[ok] / FLOW_SCALE, p1.reshape(-1, 2)[ok] / FLOW_SCALE
        if len(p0) < 15:
            return None, {"ego_motion": "tracking lost"}
        if np.median(np.linalg.norm(p1 - p0, axis=1)) < 0.3:
            return static, {"ego_motion": "static"}
        g0, v0 = self.cam.pixel_to_ground(p0)
        g1, v1 = self.cam.pixel_to_ground(p1)
        keep = v0 & v1 & (g0[:, 0] < self.P.roi["x_max"] * 0.6) & (np.abs(g0[:, 1]) < self.P.y_max)
        g0, g1 = g0[keep], g1[keep]
        if len(g0) < 10:
            return None, {"ego_motion": "too few ground tracks"}
        thr = max(0.05, 0.15 * float(np.median(np.linalg.norm(g1 - g0, axis=1))))
        # RANSAC over 2-point rigid hypotheses, evaluated for all hypotheses at once (vectorised)
        n_h = 200
        i = self.rng.integers(0, len(g0), (n_h, 2))
        da, db = g0[i[:, 1]] - g0[i[:, 0]], g1[i[:, 1]] - g1[i[:, 0]]
        ok = np.linalg.norm(da, axis=1) > 0.2
        if not ok.any():
            return None, {"ego_motion": "degenerate ground tracks"}
        i, da, db = i[ok], da[ok], db[ok]
        ang = np.arctan2(db[:, 1], db[:, 0]) - np.arctan2(da[:, 1], da[:, 0])
        c, s_ = np.cos(ang), np.sin(ang)
        Rs = np.stack([np.stack([c, -s_], 1), np.stack([s_, c], 1)], 1)            # (n, 2, 2)
        ma, mb = (g0[i[:, 0]] + g0[i[:, 1]]) / 2, (g1[i[:, 0]] + g1[i[:, 1]]) / 2
        ts = mb - np.einsum("nij,nj->ni", Rs, ma)
        pred = np.einsum("nij,mj->nmi", Rs, g0) + ts[:, None, :]                  # (n, m, 2)
        counts = (np.linalg.norm(pred - g1[None], axis=2) < thr).sum(1)
        k = int(np.argmax(counts))
        if counts[k] < max(8, 0.25 * len(g0)):
            return None, {"ego_motion": "no consistent ground motion"}
        R, t = Rs[k], ts[k]
        inl = np.linalg.norm(g0 @ R.T + t - g1, axis=1) < thr
        a, b = g0[inl], g1[inl]                                     # least-squares refinement (Kabsch)
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        R = (U @ Vt).T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = (U @ Vt).T
        t = cb - R @ ca
        if abs(math.degrees(math.atan2(R[1, 0], R[0, 0]))) > 15 or np.linalg.norm(t) > 3.0:
            return None, {"ego_motion": "implausible motion rejected"}
        return (R, t), {"ego_motion": "ok", "ego_forward_h": round(float(-t[0]), 3),
                        "ego_yaw_deg": round(math.degrees(math.atan2(R[1, 0], R[0, 0])), 2), "ground_tracks": int(inl.sum())}

    def _warp(self, motion):
        R, t = motion
        M = self.Ainv @ R @ self.A
        b = self.Ainv @ (R @ self.c + t - self.c)
        T = np.hstack([M, b[:, None]]).astype(np.float32)
        size = (self.valid.shape[1], self.valid.shape[0])
        self.paint = cv2.warpAffine(self.paint, T, size, flags=cv2.INTER_LINEAR, borderValue=0)
        self.yellow = cv2.warpAffine(self.yellow, T, size, flags=cv2.INTER_LINEAR, borderValue=0)
        self.road = cv2.warpAffine(self.road, T, size, flags=cv2.INTER_LINEAR, borderValue=0)
        self.edge = cv2.warpAffine(self.edge, T, size, flags=cv2.INTER_LINEAR, borderValue=0)

    # ------------------------------------------------------------------ per frame
    def update(self, frame, bev):
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        motion, minfo = self._ego_motion(gray)
        if motion is None and self.last_motion is not None and self.coasted < 3:
            motion, self.coasted = self.last_motion, self.coasted + 1  # constant-velocity coasting for a few frames
            minfo["ego_motion"] += " -> previous motion reused"
        elif motion is not None:
            self.last_motion, self.coasted = motion, 0
        a = ALPHA if (self.frames and motion is not None) else 0.0  # unknown motion: start the memory again
        if a and motion is not None:
            self._warp(motion)
        self.prev_gray = gray
        raw, info = self.vz.extract(bev, evidence_only=True)
        last = self.vz.last
        if last is None:
            self.reset()
            return [], {**info, **minfo}
        vis = last["valid"]
        cur_paint = (last["paint"] > 0).astype(np.float32)
        cur_yel = (last["paint"] == 2).astype(np.float32)
        cur_road = last["road_like"].astype(np.float32) if last["road_like"] is not None else np.zeros_like(cur_paint)
        cur_edge = self._long_edges(bev, last["paint"] > 0, vis)
        self.cur_paint = last["paint"] > 0
        # where the camera currently sees the ground, blend; elsewhere only decay
        # current evidence is zero outside the visible ground, so a plain blend = blend inside, decay outside
        for name, cur in (("paint", cur_paint), ("yellow", cur_yel), ("road", cur_road), ("edge", cur_edge)):
            setattr(self, name, cv2.addWeighted(getattr(self, name), a, cur, 1 - a, 0))
        self.frames += 1
        lines = self._paint_lines(vis)
        solid_sides = self._solid_edge_lines(lines)
        boundaries, region = self._boundaries(vis, solid_sides)
        dividers, boundaries = self._dividers(lines, boundaries, region)
        crossings = [v for v in raw if v["type"] == "pedestrian_crossing"]
        # detections this frame, as ground polylines (X forward, Y left, in h)
        dets = []
        for b in boundaries:
            dets.append({"type": "road_boundary", "xy": self._px_to_xy(b["px"]), "score": 1.0})
        for d in dividers:
            dets.append({"type": "lane_divider_yellow" if d["yellow"] else "lane_divider",
                         "xy": self._px_to_xy(d["px"]), "score": d["score"]})
        out = self._track_lines(motion if (a and motion is not None) else None, dets, vis)
        out += crossings
        info.update(minfo)
        info.update({"tracked_frames": self.frames, "boundaries": len(boundaries), "dividers": len(dividers),
                     "active_tracks": len(self.tracks), "drawn_lines": len(out) - len(crossings)})
        return out, info

    def _px_to_xy(self, px):
        X, Y = self.P.px_to_ground(px[:, 0], px[:, 1])
        return np.column_stack([X, Y])

    def _emit(self, typ, px, score):
        X, Y = self.P.px_to_ground(px[:, 0], px[:, 1])
        return {"type": typ, "points_h": np.column_stack([X, Y]).round(3).tolist(), "px": px,
                "length_h": round(_length(px) * self.P.res, 2), "parts_merged": 1, "cv_score": round(float(score), 2),
                "contrast_ratio": None}

    # ------------------------------------------------------------------ line-level temporal tracking
    def _resample(self, xy):
        """A ground polyline -> Y sampled on the fixed forward grid Xg (NaN outside the line's X range)."""
        x, y = xy[:, 0], xy[:, 1]
        o = np.argsort(x)
        x, y = x[o], y[o]
        keep = np.concatenate([[True], np.diff(x) > 1e-6])
        x, y = x[keep], y[keep]
        if len(x) < 2:
            return np.full(self.Xg.shape, np.nan)
        return np.interp(self.Xg, x, y, left=np.nan, right=np.nan)

    def _track_lines(self, motion, dets, vis):
        """Carry each line over from the previous frame: move it by the ego-motion, then blend the new
        detection into it (an exponential update). This makes the drawn lines glide smoothly instead of being
        re-detected from scratch every frame. Lines not seen for a few frames fade out."""
        if motion is None:                                      # ego-motion unknown: can't relate to the past
            self.tracks = []
        else:
            R, t = motion
            for tr in self.tracks:                              # move last frame's lines into this frame
                xy = np.column_stack([self.Xg, tr["y"]])
                valid = np.isfinite(tr["y"])
                xy[valid] = xy[valid] @ R.T + t
                tr["y"] = self._resample(xy[valid]) if valid.sum() >= 2 else np.full(self.Xg.shape, np.nan)
        det_grids = [(d, self._resample(d["xy"])) for d in dets]
        used = [False] * len(det_grids)
        for tr in self.tracks:
            tv = np.isfinite(tr["y"])
            best, best_gap = None, TRACK_MATCH_H
            for i, (d, dy) in enumerate(det_grids):
                if used[i] or d["type"] != tr["type"]:
                    continue
                both = tv & np.isfinite(dy)
                if both.sum() < 3:
                    continue
                gap = float(np.median(np.abs(tr["y"][both] - dy[both])))
                if gap < best_gap:
                    best, best_gap = i, gap
            if best is None:
                tr["missed"] += 1
                continue
            used[best] = True
            d, dy = det_grids[best]
            tr["missed"] = 0
            tr["age"] += 1
            tr["type"] = d["type"]                              # yellow/white can flip as colour evidence firms up
            tr["score"] = 0.7 * tr["score"] + 0.3 * d["score"]
            dv = np.isfinite(dy)
            new = tr["y"].copy()
            both = tv & dv
            new[both] = (1 - TRACK_BLEND) * tr["y"][both] + TRACK_BLEND * dy[both]
            reveal = dv & ~tv                                   # ground the line now reaches for the first time
            new[reveal] = dy[reveal]
            # a short gap between the carried line and a detection in front/behind is bridged
            new[tv & ~dv] = tr["y"][tv & ~dv]
            tr["y"] = new
        for i, (d, dy) in enumerate(det_grids):                # unmatched detections start new tracks
            if not used[i]:
                self.tracks.append({"id": self._next_id, "type": d["type"], "y": dy,
                                    "age": 1, "missed": 0, "score": d["score"]})
                self._next_id += 1
        self.tracks = [tr for tr in self.tracks if tr["missed"] <= TRACK_MAX_MISS and np.isfinite(tr["y"]).sum() >= 2]
        return self._render_tracks(vis)

    def _render_tracks(self, vis):
        """Draw each confirmed track, but only over ground the camera currently sees (never the hatched area)."""
        P = self.P
        H, W = vis.shape
        out = []
        for tr in self.tracks:
            if tr["age"] < TRACK_MIN_AGE:
                continue
            y = tr["y"]
            ok = np.isfinite(y)
            if ok.sum() < 2:
                continue
            ys = self._smooth_nan(y)
            cols, rows = P.ground_to_px(self.Xg, np.nan_to_num(ys))
            visible = ok.copy()                                 # keep only points on currently visible ground
            ri = np.clip(np.nan_to_num(rows).astype(int), 0, H - 1)
            ci = np.clip(np.nan_to_num(cols).astype(int), 0, W - 1)
            visible &= vis[ri, ci]
            for seg in self._runs(visible):
                if len(seg) < 2:
                    continue
                px = np.column_stack([cols[seg], rows[seg]])
                out.append({"type": tr["type"], "points_h": np.column_stack([self.Xg[seg], ys[seg]]).round(3).tolist(),
                            "px": px, "length_h": round(_length(px) * P.res, 2), "parts_merged": 1,
                            "cv_score": round(float(min(1.0, tr["score"])), 2), "contrast_ratio": None, "track_id": tr["id"]})
        return out

    @staticmethod
    def _runs(mask):
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return []
        return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)

    def _smooth_nan(self, y):
        v = np.isfinite(y)
        out = y.copy()
        for seg in self._runs(v):
            if len(seg) >= 3:
                out[seg] = np.convolve(np.pad(y[seg], 1, mode="edge"), [1 / 3, 1 / 3, 1 / 3], "valid")
        return out

    # ------------------------------------------------------------------ longitudinal edges (kerbs)
    def _long_edges(self, bev, paint, vis):
        """Edges that run ALONG the road (strong lateral gradient, weak gradient along the road), away from paint.
        Kerbs and road-surface borders produce them consistently; shadow edges have mixed orientations."""
        g = cv2.GaussianBlur(cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        e = (np.abs(gx) > 40) & (np.abs(gx) > 2.0 * np.abs(gy)) & vis
        e &= ~cv2.dilate(paint.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        return e.astype(np.float32)

    # ------------------------------------------------------------------ paint lines (histogram + sliding windows)
    def _paint_lines(self, vis):
        P = self.P
        px = lambda h: max(1, int(round(h / P.res)))
        paint = ((self.paint > PAINT_ON) & vis).astype(np.float32)
        H, W = paint.shape
        vis_rows = np.flatnonzero(vis.any(1))
        if len(vis_rows) == 0:
            return []
        w = px(0.3)
        hist = np.convolve(paint.sum(0), np.ones(w) / w, "same")
        peaks = []
        for c in np.argsort(hist)[::-1]:
            if hist[c] < px(0.8):
                break
            if all(abs(c - q) > px(0.9) for q in peaks):
                peaks.append(int(c))
        win_h, win_w = px(1.0), px(0.35)
        lines = []
        for c0 in peaks:
            pts, c, misses = [], float(c0), 0
            for top in range(int(vis_rows.max()), int(vis_rows.min()), -win_h):    # near -> far
                r0 = max(top - win_h, 0)
                lo, hi = int(max(c - win_w, 0)), int(min(c + win_w + 1, W))
                ys, xs = np.nonzero(paint[r0:top, lo:hi])
                if len(xs) >= 3:
                    c = float(xs.mean() + lo)
                    pts += list(zip(xs + lo, ys + r0))
                    misses = 0
                else:
                    misses += 1
                    if misses > 4:                                    # > 4 h without paint: the line ended
                        break
            if not pts:
                continue
            p = np.array(pts, float)
            span = (p[:, 1].max() - p[:, 1].min()) * P.res
            cover = len(np.unique((p[:, 1] // px(0.25)).astype(int))) * 0.25 / max(span, 1e-6)
            if span < 2 * MIN_DIVIDER_H or cover < 0.25:
                continue                                              # too short or too sparse (arrows, blobs)
            line = _fit_curve(p[:, 1], p[:, 0], px(0.25), P.res)
            yi, xi = p[:, 1].astype(int), p[:, 0].astype(int)
            yellow = float(self.yellow[yi, xi].mean()) > 0.5 * float(self.paint[yi, xi].mean() + 1e-6)
            _, Y = P.px_to_ground(line[:, 0], line[:, 1])
            # solid vs dashed from the CURRENT frame only (fusion smears dashes into continuous streaks)
            hits = []
            near_row = H - 1 - int(0.6 * H)                           # nearer 60 % of the range: dashes are sharp there
            for cc, rr in line:
                r_, c_ = int(rr), int(cc)
                if near_row <= r_ < H and 0 <= c_ < W and vis[r_, c_]:
                    hits.append(self.cur_paint[r_, max(c_ - win_w // 2, 0):c_ + win_w // 2 + 1].any())
            cover_now = float(np.mean(hits)) if hits else 0.0
            lines.append({"px": line, "yellow": yellow, "span": span, "cover": cover, "y_mean": float(np.mean(Y)),
                          "cover_now": cover_now, "solid": cover_now >= 0.8 and span >= 4.0,
                          "score": min(1.0, span / 4.0) * min(1.0, cover / 0.5)})
        return lines

    def _solid_edge_lines(self, lines):
        """Outermost long SOLID (not dashed) paint line on each side of the vehicle: a road edge line."""
        out = {}
        for side, sign in (("left", 1), ("right", -1)):
            cand = [l for l in lines if l["solid"] and sign * l["y_mean"] > 0.8]
            if cand:
                out[side] = max(cand, key=lambda l: sign * l["y_mean"])
        return out

    # ------------------------------------------------------------------ road boundary
    def _boundaries(self, vis, solid_sides):
        """Each side: the outermost solid edge line if there is one; otherwise walk outwards along every BEV row
        from the vehicle's lane until the road surface ends (>= 0.3 h of non-road, or a kerb-like edge).
        Rows where the walk reaches the edge of the visible ground give no boundary (the road may continue)."""
        P = self.P
        px = lambda h: max(1, int(round(h / P.res)))
        road = (self.road > ROAD_ON) | (self.paint > PAINT_ON)
        k = px(0.15) | 1
        road = cv2.morphologyEx(road.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)).astype(bool)
        # kerb-like edges, ignoring anything near paint seen in recent frames (dash and arrow outlines)
        near_paint = cv2.dilate((self.paint > 0.08).astype(np.uint8), np.ones((px(0.3) | 1, px(0.3) | 1), np.uint8)).astype(bool)
        kerb = (self.edge > 0.35) & ~near_paint
        H, W = road.shape
        c_ego = int(round(P.ground_to_px(np.array([0.0]), np.array([0.0]))[0][0]))
        run = px(0.3)
        sides = {"left": [], "right": []}
        region = np.zeros_like(road)
        for r in range(0, H, 2):
            if not vis[r, c_ego] or not road[r, max(c_ego - px(0.5), 0):c_ego + px(0.5)].any():
                continue
            for name, step in (("left", -1), ("right", 1)):
                c, miss, last_road, stop = c_ego + step * px(0.6), 0, c_ego, False
                while 0 <= c < W and vis[r, c]:
                    if kerb[r, c]:
                        stop, last_road = True, c - step
                        break
                    if road[r, c]:
                        miss, last_road = 0, c
                    else:
                        miss += 1
                        if miss >= run:
                            stop = True
                            break
                    c += step
                lo, hi = sorted((c_ego, last_road))
                region[r:r + 2, lo:hi + 1] = True
                if stop and 0 <= c < W and vis[r, c]:                 # stopped inside the view
                    sides[name].append((last_road + 0.5 * step, r))
        out = []
        for name, pts in sides.items():
            if name in solid_sides:
                line = solid_sides[name]["px"]
                out.append({"px": line, "src": "edge_line", "side": name})
                ri, ci = line[:, 1].astype(int).clip(0, H - 1), line[:, 0].astype(int).clip(0, W - 1)
                for r, c in zip(ri, ci):                              # drivable region extends to the edge line
                    lo, hi = sorted((c_ego, c))
                    region[r, lo:hi + 1] = True
                continue
            if len(pts) < 4:
                continue
            p = np.array(pts, float)
            # split where the boundary jumps or is interrupted (e.g. a side street)
            breaks = np.flatnonzero((np.abs(np.diff(p[:, 0])) > px(0.6)) | (np.diff(p[:, 1]) > px(1.0))) + 1
            for seg in np.split(p, breaks):
                if len(seg) < 4 or (seg[:, 1].max() - seg[:, 1].min()) * P.res < MIN_BOUNDARY_H:
                    continue
                kk = max(3, min(len(seg) // 2 * 2 - 1, px(0.8) // 2 * 2 + 1))
                cols = seg[:, 0].copy()
                padc = np.pad(cols, kk // 2, mode="edge")             # median filter removes single-row spikes
                cols = np.median(np.lib.stride_tricks.sliding_window_view(padc, kk), axis=1)
                fit = _fit_curve(seg[:, 1], cols, px(0.25), P.res)
                _, Yb = P.px_to_ground(fit[:, 0], fit[:, 1])
                if np.median(np.abs(Yb)) < 0.7:
                    continue                                          # inside the vehicle's own lane: not a road edge
                out.append({"px": fit, "src": "scan", "side": name})
        return out, region

    # ------------------------------------------------------------------ dividers
    def _dividers(self, lines, boundaries, region):
        """Paint lines inside the road become dividers. A dashed line is never a road edge: if a scan-derived
        boundary runs along a dashed line, that boundary is dropped instead. Returns (dividers, boundaries)."""
        P = self.P
        def dist(a, b):
            return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1) * P.res
        def beyond(line, b):
            """share of the line's points lying OUTSIDE boundary b (on b's side of the vehicle)"""
            rows = b["px"][:, 1]
            if len(rows) < 2 or rows.max() - rows.min() < 1:
                return 0.0
            order = np.argsort(rows)
            bc = np.interp(line[:, 1], rows[order], b["px"][order, 0], left=np.nan, right=np.nan)
            ok = np.isfinite(bc)
            if not ok.any():
                return 0.0
            sign = -1 if b["side"] == "left" else 1                  # left side = smaller column
            return float(np.mean(sign * (line[ok, 0] - bc[ok]) > 0))
        drop = set()
        out = []
        H, W = region.shape
        for l in lines:
            line = l["px"]
            ri, ci = line[:, 1].astype(int).clip(0, H - 1), line[:, 0].astype(int).clip(0, W - 1)
            if region[ri, ci].mean() < 0.5:
                continue                                              # not on the scanned drivable surface
            near = [i for i, b in enumerate(boundaries) if np.median(dist(line, b["px"])) < EDGE_LINE_GAP_H]
            if near:
                if l["solid"] or any(boundaries[i]["src"] == "edge_line" for i in near):
                    continue                                          # the edge line itself, or paint right beside it
                drop.update(i for i in near if boundaries[i]["src"] == "scan")
            if any(beyond(line, b) > 0.5 for i, b in enumerate(boundaries) if i not in drop and i not in near):
                continue                                              # outside the road (e.g. kerb stripes)
            out.append(l)
        return out, [b for i, b in enumerate(boundaries) if i not in drop]
