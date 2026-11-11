"""Inverse perspective mapping onto the estimated ground plane (relative units: camera heights).

The remap tables are built ONCE per calibration (they include the lens model), so each frame costs one
cv2.remap.  BEV pixels the camera cannot observe (outside the image, behind the camera, on the bonnet or
on static overlays) are marked invalid and are never filled in.
"""
import numpy as np
import cv2

from .camera_model import CameraModel

BEV_W, BEV_H = 560, 720          # 560/720 gives lateral half-width ~0.39 x forward range (~10 m at the 13.5 h cap)


class BevProjector:
    def __init__(self, cam: CameraModel, roi, static_small=None, hood_row=None):
        self.cam, self.roi = cam, roi
        self.res = (roi["x_max"] - roi["x_min"]) / BEV_H          # ground units per BEV pixel (both axes)
        self.y_max = self.res * BEV_W / 2
        cols = np.arange(BEV_W) + 0.5
        rows = np.arange(BEV_H) + 0.5
        Y = self.y_max - cols * self.res                          # image left = vehicle left (+Y)
        X = roi["x_max"] - rows * self.res                        # image top = far
        GX, GY = np.meshgrid(X, Y, indexing="ij")
        uv, z = cam.project_ground(GX, GY)
        W, H = cam.width, cam.height
        u, v = uv[..., 0], uv[..., 1]
        valid = (z > 1e-3) & np.isfinite(u) & np.isfinite(v)
        u, v = np.nan_to_num(u, nan=-1.0), np.nan_to_num(v, nan=-1.0)
        valid &= (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if hood_row is not None:
            valid &= v < hood_row
        if static_small is not None:
            st = np.asarray(static_small, np.uint8)
            sm = cv2.resize(st, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            ui, vi = u.clip(0, W - 1).astype(int), v.clip(0, H - 1).astype(int)
            valid &= ~sm[vi, ui]
        valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
        self.valid = valid
        self.map_x = np.where(valid, u, -1).astype(np.float32)
        self.map_y = np.where(valid, v, -1).astype(np.float32)
        self.valid_fraction = float(valid.mean())
        # image-space outline of the ROI (for drawing on the original frame)
        edge = []
        for x in np.linspace(roi["x_min"] + 1e-3, roi["x_max"], 40):
            edge.append((x, self.y_max))
        for y in np.linspace(self.y_max, -self.y_max, 20):
            edge.append((roi["x_max"], y))
        for x in np.linspace(roi["x_max"], roi["x_min"] + 1e-3, 40):
            edge.append((x, -self.y_max))
        e = np.array(edge)
        p, zz = cam.project_ground(e[:, 0], e[:, 1])
        # keep only points clearly in front of the camera and not absurdly far outside the image, as
        # separate runs (a point near the camera plane projects to infinity and would draw stray lines)
        ok = (zz > 0.3 * cam.h) & np.isfinite(p).all(1)
        ok &= (np.abs(np.nan_to_num(p[:, 0]) - W / 2) < 3 * W) & (np.abs(np.nan_to_num(p[:, 1]) - H / 2) < 3 * H)
        runs, cur = [], []
        for q, good in zip(p, ok):
            if good:
                cur.append(q)
            elif cur:
                runs.append(np.array(cur)); cur = []
        if cur:
            runs.append(np.array(cur))
        self.roi_outline_runs = [r for r in runs if len(r) >= 2]

    def warp(self, frame):
        bev = cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bev[~self.valid] = 0
        return bev

    def px_to_ground(self, cols, rows):
        cols = np.asarray(cols, float); rows = np.asarray(rows, float)
        return self.roi["x_max"] - (rows + 0.5) * self.res, self.y_max - (cols + 0.5) * self.res

    def ground_to_px(self, X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        return (self.y_max - Y) / self.res - 0.5, (self.roi["x_max"] - X) / self.res - 0.5
