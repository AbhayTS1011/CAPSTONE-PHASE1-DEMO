"""Traditional-CV extraction of road-marking and road-edge candidates from ONE BEV frame, as polylines.

Lane / transverse markings : bright thin structures -> white top-hat + adaptive threshold (+ yellow test in Lab),
                             morphological cleaning, connected components filtered by measured shape
                             (length, thinness, elongation), principal-axis centreline, Douglas-Peucker.
Road-edge candidates       : long, roughly longitudinal intensity edges (Canny + probabilistic Hough) that do
                             not coincide with a marking; at most one per side (the outermost well-supported).
Nothing is carried over between frames and nothing is drawn when the measurements do not support it.
All lengths are in camera heights (h); thresholds are expressed in h so they do not depend on BEV resolution.
"""
import math
import numpy as np
import cv2

MARK_KERNEL_H = 0.30          # top-hat kernel diameter (markings are much thinner than this)
MIN_MARK_LEN_H = 0.45
MAX_MARK_WIDTH_H = 0.20
MIN_ELONGATION = 3.0
MERGE_ANGLE_DEG = 10.0
MERGE_OFFSET_H = 0.25
MERGE_GAP_H = 4.0
EDGE_MIN_LEN_H = 2.5
MIN_ROAD_CONTEXT = 0.35        # share of a marking's surrounding ring that must look like this frame's road surface
MIN_CV_SCORE = 0.2
WIDE_KERNEL_H = 1.0            # second, coarser top-hat for wide paint (zebra stripes, stop lines)
WIDE_WIDTH_H = (0.2, 0.7)      # wide-paint width range (thin markings are < 0.2 h)
CROSSING_MIN_STRIPES = 3              # length/3h x contrast/2, both clipped to 1 (a heuristic, not a probability)


def _pca_axis(pts):
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    w, v = np.linalg.eigh(cov)
    return c, v[:, 1], v[:, 0], w[1], max(w[0], 1e-9)


def _centreline(pts, c, major, step):
    t = (pts - c) @ major
    lo, hi = t.min(), t.max()
    nb = max(2, int((hi - lo) / step) + 1)
    edges = np.linspace(lo, hi + 1e-9, nb + 1)
    out = []
    idx = np.digitize(t, edges) - 1
    for b in range(nb):
        sel = idx == b
        if sel.sum() >= 2:
            out.append(pts[sel].mean(0))
    return np.array(out)


def _simplify(poly, eps):
    if len(poly) < 3:
        return poly
    return cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), eps, False).reshape(-1, 2).astype(float)


def _length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def _is_radial(P, poly):
    """True if the polyline lies along a ray from the camera: the signature of a vertical object (pole,
    vehicle side, wall) smeared by the flat-ground IPM, not of something painted on the road."""
    X, Y = P.px_to_ground(poly[[0, -1], 0], poly[[0, -1], 1])
    if abs(X[1] - X[0]) < 1e-6:
        return False
    ymid = (Y[0] + Y[1]) / 2
    y0 = Y[0] - X[0] * (Y[1] - Y[0]) / (X[1] - X[0])            # lateral offset where the line meets X = 0
    return abs(ymid) > 0.8 and abs(y0) < 0.35 * abs(ymid)


def _dir_deg(p):
    """Direction of a polyline relative to the forward axis (BEV image 'up'), 0 = longitudinal, 90 = transverse."""
    d = p[-1] - p[0]
    return math.degrees(math.atan2(abs(d[0]), abs(d[1]) + 1e-9))


class Vectorizer:
    def __init__(self, projector):
        self.P = projector
        r = projector.res
        k = max(5, int(round(MARK_KERNEL_H / r)) | 1)
        self.tophat_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.valid_er = cv2.erode(projector.valid.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        self.px = lambda h: h / r

    def extract(self, bev, evidence_only=False):
        P = self.P
        lab = cv2.cvtColor(bev, cv2.COLOR_BGR2LAB)
        L, bch = lab[..., 0], lab[..., 2]
        valid = self.valid_er
        if valid.sum() < 500:
            self.last = None
            return [], {"reason": "insufficient visible ground"}
        road_like, road_info = self._road_model(lab, valid)
        th = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, self.tophat_k)
        vals = th[valid]
        med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1.0
        thr = max(med + 5.0 * mad, 14.0)
        mask = (th > thr) & valid
        yellow = (bch.astype(int) > 148) & (th > 0.6 * thr) & valid
        mask |= yellow
        m8 = mask.astype(np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
        n, lab_img, stats, _ = cv2.connectedComponentsWithStats(m8, 8)
        cands = []
        accepted = np.zeros(L.shape, np.uint8)                     # pixels of accepted paint (evidence for tracking)
        min_area = max(12, int(self.px(MIN_MARK_LEN_H) * 2))
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            x, y, w, h = stats[i, :4]
            sub = lab_img[y:y + h, x:x + w] == i
            ys, xs = np.nonzero(sub)
            pts = np.column_stack([xs + x, ys + y]).astype(float)
            c, major, minor, l1, l2 = _pca_axis(pts)
            elong = math.sqrt(l1 / l2)
            t = (pts - c) @ major
            length_px = float(t.max() - t.min())
            width_px = len(pts) / max(length_px, 1)
            length_h, width_h = length_px * P.res, width_px * P.res
            if length_h < MIN_MARK_LEN_H or width_h > MAX_MARK_WIDTH_H or elong < MIN_ELONGATION:
                continue
            line = _centreline(pts, c, major, self.px(0.4))
            if len(line) < 2 or _is_radial(P, line):
                continue
            if road_like is not None:                               # paint lies ON the road: check its surroundings
                y0, y1, x0, x1 = max(y - 6, 0), min(y + h + 6, lab_img.shape[0]), max(x - 6, 0), min(x + w + 6, lab_img.shape[1])
                comp = (lab_img[y0:y1, x0:x1] == i).astype(np.uint8)
                ring = cv2.dilate(comp, np.ones((11, 11), np.uint8)).astype(bool) & ~comp.astype(bool) & valid[y0:y1, x0:x1]
                if ring.sum() == 0 or road_like[y0:y1, x0:x1][ring].mean() < MIN_ROAD_CONTEXT:
                    continue
            contrast = float(th[ys + y, xs + x].mean() / thr)
            is_yellow = float(yellow[ys + y, xs + x].mean()) > 0.5
            cands.append({"px": line, "length_h": length_h, "width_h": width_h, "contrast": contrast,
                          "yellow": is_yellow, "elongation": elong, "parts": 1})
            accepted[ys + y, xs + x] = 2 if is_yellow else 1
        crossings, stop_lines, cross_polys = self._wide_paint(L, valid, road_like)
        self.last = {"paint": accepted, "road_like": road_like, "road_relaxed": getattr(self, "road_relaxed", None),
                     "crossing_polys": cross_polys, "valid": valid}
        if cross_polys:                                             # thin fragments on crossing stripe edges are not lane lines
            cands = [c for c in cands if not any(cv2.pointPolygonTest(cp, tuple(map(float, c["px"].mean(0))), False) >= 0 for cp in cross_polys)]
        if evidence_only:                                           # the tracker only needs evidence maps + crossings
            marks, edges = [], []
        else:
            marks = self._merge(cands) + stop_lines
            edges = self._road_edges(L, valid, marks, road_like)
        out = []
        for m in crossings:
            X, Y = P.px_to_ground(m["px"][:, 0], m["px"][:, 1])
            score = min(1.0, m["stripes"] / 4.0) * min(1.0, m["contrast"] / 2.0)
            out.append({"type": "pedestrian_crossing", "points_h": np.column_stack([X, Y]).round(3).tolist(), "px": m["px"],
                        "length_h": round(m["length_h"], 2), "parts_merged": m["stripes"], "stripes": m["stripes"],
                        "cv_score": round(float(score), 2), "contrast_ratio": round(m["contrast"], 2)})
        for m in marks + edges:
            poly = _simplify(m["px"], 0.05 / P.res)
            X, Y = P.px_to_ground(poly[:, 0], poly[:, 1])
            ang = _dir_deg(poly)
            if m.get("edge"):
                typ = "road_edge_candidate"
            elif ang > 60:
                typ = "transverse_marking"
            elif ang < 30:
                typ = "lane_marking_yellow" if m["yellow"] else "lane_marking"
            else:
                typ = "oblique_marking"
            score = min(1.0, m["length_h"] / 3.0) * min(1.0, m["contrast"] / 2.0)
            if score < MIN_CV_SCORE:
                continue                                           # too short / too faint to report
            out.append({"type": typ, "points_h": np.column_stack([X, Y]).round(3).tolist(), "px": poly,
                        "length_h": round(m["length_h"], 2), "parts_merged": m["parts"],
                        "cv_score": round(float(score), 2), "contrast_ratio": round(m["contrast"], 2)})
        return out, {"tophat_threshold": round(thr, 1), "components": int(n - 1), "marking_candidates": len(marks),
                     "edge_candidates": len(edges), "crossings": len(crossings), "stop_lines": len(stop_lines), **road_info}

    # ---- wide paint: zebra-crossing stripe groups and stop lines ----
    def _wide_paint(self, L, valid, road_like):
        P = self.P
        small = cv2.resize(L, (L.shape[1] // 2, L.shape[0] // 2), interpolation=cv2.INTER_AREA)
        vs = cv2.resize(valid.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        k = max(5, int(round(WIDE_KERNEL_H / (2 * P.res))) | 1)
        th = cv2.morphologyEx(small, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        vals = th[vs]
        if len(vals) < 200:
            return [], [], []
        med = float(np.median(vals)); mad = float(np.median(np.abs(vals - med))) * 1.4826 + 1.0
        thr = max(med + 5.0 * mad, 18.0)
        m8 = ((th > thr) & vs).astype(np.uint8)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        res2 = 2 * P.res
        stripes = []
        # Directional openings split "ladder" crossings (stripes joined by thin cross-bars) into their stripes:
        # a long kernel along one BEV axis keeps paint elongated in that direction and removes thin bars across it.
        kl = max(5, int(round(0.5 / res2)))
        comps = []
        for kern, axis in ((np.ones((kl, 1), np.uint8), np.array([0.0, 1.0])), (np.ones((1, kl), np.uint8), np.array([1.0, 0.0]))):
            mo = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kern)
            n, lab_img, stats, _ = cv2.connectedComponentsWithStats(mo, 8)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= 10:
                    ys, xs = np.nonzero(lab_img == i)
                    comps.append((ys, xs, axis))
        for ys, xs, axis in comps:
            pts = np.column_stack([xs, ys]).astype(np.float32)
            (cx_, cy_), (rw, rh), ang = cv2.minAreaRect(pts)
            length_h, width_h = max(rw, rh) * res2, min(rw, rh) * res2
            rect_fill = len(pts) / max(rw * rh, 1.0)
            if not (WIDE_WIDTH_H[0] < width_h <= WIDE_WIDTH_H[1]) or length_h < 0.6 or length_h / width_h < 2.0 or rect_fill < 0.6:
                continue
            full = pts * 2.0 + 0.5                                      # back to full-resolution BEV pixels
            if road_like is not None:
                x0, y0 = np.maximum(full.min(0).astype(int) - 8, 0)
                x1, y1 = np.minimum(full.max(0).astype(int) + 9, [L.shape[1], L.shape[0]])
                comp = np.zeros((y1 - y0, x1 - x0), np.uint8)
                fi = full.astype(int)
                comp[(fi[:, 1] - y0).clip(0, y1 - y0 - 1), (fi[:, 0] - x0).clip(0, x1 - x0 - 1)] = 1
                comp = cv2.dilate(comp, np.ones((3, 3), np.uint8))
                ring = cv2.dilate(comp, np.ones((15, 15), np.uint8)).astype(bool) & ~comp.astype(bool) & valid[y0:y1, x0:x1]
                if ring.sum() == 0 or road_like[y0:y1, x0:x1][ring].mean() < MIN_ROAD_CONTEXT:
                    continue
            c, major, _, _, _ = _pca_axis(full.astype(float))
            if abs(float(np.dot(major, axis))) < math.cos(math.radians(35)):
                continue                                            # not elongated along this opening's axis
            if any(np.linalg.norm(c - o["c"]) < 0.3 / P.res for o in stripes):
                continue                                            # already found by the other opening
            stripes.append({"c": c, "dir": major, "len_px": length_h / P.res, "full": full, "length_h": length_h,
                            "contrast": float(th[ys, xs].mean() / thr)})
        # group parallel, side-by-side stripes (union-find)
        parent = list(range(len(stripes)))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for a in range(len(stripes)):
            for b in range(a + 1, len(stripes)):
                A, B = stripes[a], stripes[b]
                if abs(float(np.dot(A["dir"], B["dir"]))) < math.cos(math.radians(15)):
                    continue
                d = B["c"] - A["c"]
                along = abs(float(np.dot(d, A["dir"]))) * P.res
                across = abs(float(np.cross(A["dir"], d))) * P.res
                if 0.25 < across < 1.3 and along < 0.6 * max(A["length_h"], B["length_h"]):
                    parent[find(a)] = find(b)
        groups = {}
        for i in range(len(stripes)):
            groups.setdefault(find(i), []).append(i)
        crossings, stop_lines, polys = [], [], []
        for idx in groups.values():
            if len(idx) >= CROSSING_MIN_STRIPES:
                allp = np.vstack([stripes[i]["full"] for i in idx]).astype(np.float32)
                box = cv2.boxPoints(cv2.minAreaRect(allp))
                poly = np.vstack([box, box[:1]]).astype(float)
                polys.append(box.astype(np.float32).reshape(-1, 1, 2))
                crossings.append({"px": poly, "stripes": len(idx), "length_h": _length(poly) / 2 * P.res,
                                  "contrast": float(np.mean([stripes[i]["contrast"] for i in idx]))})
            else:
                for i in idx:                                       # a lone wide transverse bar = stop-line candidate
                    st = stripes[i]
                    if st["length_h"] >= 1.2 and abs(st["dir"][1]) < math.sin(math.radians(30)):
                        e = st["dir"] * st["len_px"] / 2
                        stop_lines.append({"px": np.array([st["c"] - e, st["c"] + e]), "length_h": st["length_h"],
                                           "contrast": st["contrast"], "yellow": False, "parts": 1})
        return crossings, stop_lines, polys

    # ---- this frame's road-surface appearance (patch directly ahead of the vehicle) ----
    def _road_model(self, lab, valid):
        P = self.P
        self.road_relaxed = None
        rows = np.flatnonzero(valid.any(1))
        if len(rows) == 0:
            return None, {"road_model": "unavailable"}
        near_row = rows.max()                                       # nearest visible BEV row
        c0, c1 = P.ground_to_px(np.array([0.0, 0.0]), np.array([0.6, -0.6]))[0].astype(int)
        r0 = max(int(near_row - self.px(4.0)), 0)
        patch = lab[r0:near_row, max(c0, 0):c1][valid[r0:near_row, max(c0, 0):c1]]
        if len(patch) < 200:
            return None, {"road_model": "unavailable"}
        med = np.median(patch, 0).astype(float)
        mad = np.median(np.abs(patch - med), 0) * 1.4826 + 2.0
        d = np.abs(lab.astype(np.int16) - med.astype(np.int16))
        road_like = (d[..., 0] < max(28, 3.5 * mad[0])) & (d[..., 1] < max(10, 3.5 * mad[1])) & (d[..., 2] < max(10, 3.5 * mad[2])) & valid
        # relaxed version for the drivable-surface estimate: tolerant to shadows (brightness), strict on colour
        self.road_relaxed = (d[..., 0] < max(50, 5 * mad[0])) & (d[..., 1] < max(8, 3 * mad[1])) & (d[..., 2] < max(8, 3 * mad[2])) & valid
        return road_like, {"road_model_lab": med.round(1).tolist(), "road_like_fraction": round(float(road_like[valid].mean()), 3)}

    # ---- join collinear fragments (dashed lines) ----
    def _merge(self, cands):
        P = self.P
        long = [c for c in cands if _dir_deg(c["px"]) < 30]
        other = [c for c in cands if _dir_deg(c["px"]) >= 30]
        for c in long:                                    # orient far -> near (row ascending)
            if c["px"][0, 1] > c["px"][-1, 1]:
                c["px"] = c["px"][::-1]
        long.sort(key=lambda c: c["px"][0, 1])
        used = [False] * len(long)
        merged = []
        for i, a in enumerate(long):
            if used[i]:
                continue
            used[i] = True
            cur = dict(a)
            changed = True
            while changed:
                changed = False
                for j, b in enumerate(long):
                    if used[j]:
                        continue
                    end, start = cur["px"][-1], b["px"][0]
                    gap = (start[1] - end[1]) * P.res
                    if gap < -0.2 or gap > MERGE_GAP_H:
                        continue
                    da = cur["px"][-1] - cur["px"][max(0, len(cur["px"]) - 3)]
                    db = b["px"][min(2, len(b["px"]) - 1)] - b["px"][0]
                    ang = abs(math.degrees(math.atan2(da[0], da[1] + 1e-9) - math.atan2(db[0], db[1] + 1e-9)))
                    # lateral offset of b's start from cur's extrapolated end
                    d = da / (np.linalg.norm(da) + 1e-9)
                    off = abs(np.cross(d, start - end)) * P.res
                    if ang < MERGE_ANGLE_DEG and off < MERGE_OFFSET_H:
                        cur["px"] = np.vstack([cur["px"], b["px"]])
                        cur["length_h"] += b["length_h"]
                        cur["contrast"] = max(cur["contrast"], b["contrast"])
                        cur["yellow"] = cur["yellow"] or b["yellow"]
                        cur["parts"] += 1
                        used[j] = True
                        changed = True
            merged.append(cur)
        return merged + other

    # ---- road-edge candidates ----
    def _road_edges(self, L, valid, marks, road_like):
        P = self.P
        g = cv2.GaussianBlur(L, (7, 7), 0)
        e = cv2.Canny(g, 30, 90)
        e[~valid] = 0
        lines = cv2.HoughLinesP(e, 1, np.pi / 180, threshold=int(self.px(1.2)),
                                minLineLength=int(self.px(EDGE_MIN_LEN_H)), maxLineGap=int(self.px(0.4)))
        if lines is None:
            return []
        mark_pts = np.vstack([m["px"] for m in marks]) if marks else np.zeros((0, 2))
        best = {}
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            p = np.array([[x1, y1], [x2, y2]], float)
            if _dir_deg(p) > 25 or _is_radial(P, p):
                continue
            if len(mark_pts):
                d = np.abs(np.cross(p[1] - p[0], mark_pts - p[0])) / (np.linalg.norm(p[1] - p[0]) + 1e-9)
                within = (mark_pts[:, 1] >= min(y1, y2) - 5) & (mark_pts[:, 1] <= max(y1, y2) + 5)
                if (within & (d * P.res < 0.3)).sum() >= 2:
                    continue                                      # this edge is a marking border
            # edge strength: mean absolute lateral gradient along the line
            n = int(np.hypot(x2 - x1, y2 - y1))
            xs = np.linspace(x1, x2, n).astype(int); ys = np.linspace(y1, y2, n).astype(int)
            xl, xr = (xs - 3).clip(0, L.shape[1] - 1), (xs + 3).clip(0, L.shape[1] - 1)
            strength = float(np.abs(g[ys, xr].astype(int) - g[ys, xl].astype(int)).mean())
            if strength < 12:
                continue
            Xc, Yc = P.px_to_ground((x1 + x2) / 2, (y1 + y2) / 2)
            side = "left" if Yc > 0 else "right"
            if road_like is not None:                               # road surface must lie on the inner (ego) side
                inner = (xs + (6 if Yc > 0 else -6)).clip(0, L.shape[1] - 1)
                outer = (xs - (6 if Yc > 0 else -6)).clip(0, L.shape[1] - 1)
                if road_like[ys, inner].mean() < 0.6 or road_like[ys, outer].mean() > 0.6:
                    continue
            if len(mark_pts):                                      # a road edge lies outside the markings on its side
                _, my = P.px_to_ground(mark_pts[:, 0], mark_pts[:, 1])
                same = my[np.sign(my) == np.sign(Yc)]
                if len(same) and abs(Yc) < np.abs(same).max() - 0.2:
                    continue
            length_h = n * P.res
            key = abs(Yc)                                          # prefer the outermost well-supported edge
            if side not in best or key > best[side][0]:
                best[side] = (key, {"px": p, "length_h": length_h, "contrast": strength / 12.0,
                                    "yellow": False, "parts": 1, "edge": True})
        return [v[1] for v in best.values()]
