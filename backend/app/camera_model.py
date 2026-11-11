"""Pinhole camera + flat ground plane, expressed in RELATIVE units (camera height h = 1).

Camera frame: x right, y down, z forward (OpenCV).  Ground frame: X forward, Y left, Z up, origin on the
ground directly below the camera.  A ground point (X, Y) is P_cam = X*fwd + Y*left - h*up.
Absolute scale is NOT known: one ground unit = one camera height (whatever that is in metres).
"""
from dataclasses import dataclass, field
import numpy as np
import cv2


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


@dataclass
class CameraModel:
    width: int
    height: int
    f: float
    cx: float
    cy: float
    k1: float = 0.0
    fwd: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    left: np.ndarray = field(default_factory=lambda: np.array([-1.0, 0.0, 0.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, -1.0, 0.0]))
    h: float = 1.0

    @property
    def K(self):
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @property
    def dist(self):
        return np.array([self.k1, 0, 0, 0, 0], float)

    @classmethod
    def from_vanishing(cls, width, height, f, cx, cy, k1, vp_forward, up_dir):
        """Orientation from the road-direction vanishing point and a ground-normal ('up') direction in camera frame."""
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        d = unit(np.linalg.solve(K, [vp_forward[0], vp_forward[1], 1.0]))
        up = unit(up_dir)
        if up[1] > 0:
            up = -up
        fwd = unit(d - np.dot(d, up) * up)
        left = np.cross(up, fwd)
        return cls(width, height, f, cx, cy, k1, fwd, left, up)

    @staticmethod
    def up_from_horizon(f, cx, cy, line):
        """Ground normal from the image horizon line l (a,b,c): n ~ K^T l, oriented upward (negative camera y)."""
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        n = unit(K.T @ np.asarray(line, float))
        return -n if n[1] > 0 else n

    # ---- angles reported for diagnostics (degrees) ----
    def angles(self):
        pitch = np.degrees(np.arctan2(-self.fwd[1], np.hypot(self.fwd[0], self.fwd[2])))  # >0: road dir above optical axis -> camera tilted down
        yaw = np.degrees(np.arctan2(self.fwd[0], self.fwd[2]))
        roll = np.degrees(np.arctan2(-self.left[1], -self.left[0]))
        return {"pitch_deg": float(pitch), "yaw_deg": float(yaw), "roll_deg": float(roll)}

    def horizon_line(self):
        """Image line of the ground plane's vanishing line (undistorted pixel coordinates)."""
        l = np.linalg.inv(self.K).T @ self.up
        return l / np.linalg.norm(l[:2])

    def ground_homography(self):
        """3x3 H mapping ground (X, Y, 1) -> undistorted pixel."""
        return self.K @ np.column_stack([self.fwd, self.left, -self.h * self.up])

    def project_ground(self, X, Y):
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        P = X[..., None] * self.fwd + Y[..., None] * self.left - self.h * self.up
        z = P[..., 2]
        pts = P.reshape(-1, 3)
        uv, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K, self.dist)
        uv = uv.reshape(P.shape[:-1] + (2,))
        if self.k1:
            # the radial polynomial is only meaningful inside the field of view it was fitted on: beyond
            # 1.2x the image-corner radius it folds far-away points back into the image, so mark them invalid
            r2 = (P[..., 0] ** 2 + P[..., 1] ** 2) / np.maximum(z, 1e-9) ** 2
            r2max = 1.44 * (max(self.cx, self.width - self.cx) ** 2 + max(self.cy, self.height - self.cy) ** 2) / self.f ** 2
            uv[(r2 > r2max) | (z <= 0)] = np.nan
        return uv, z

    def pixel_to_ground(self, uv):
        """Undistort pixels and intersect their rays with the ground. Returns (XY, valid)."""
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        n = cv2.undistortPoints(uv, self.K, self.dist).reshape(-1, 2)
        rays = np.column_stack([n, np.ones(len(n))])
        denom = rays @ self.up
        t = -self.h / np.where(np.abs(denom) < 1e-9, -1e-9, denom)
        valid = (denom < -1e-6) & (t > 0)
        P = rays * t[:, None]
        return np.column_stack([P @ self.fwd, P @ self.left]), valid

    def to_dict(self):
        return {"image_size": [self.width, self.height], "focal_px": self.f, "principal_point_px": [self.cx, self.cy],
                "k1": self.k1, "fwd": self.fwd.tolist(), "left": self.left.tolist(), "up": self.up.tolist(),
                "camera_height_units": self.h, **self.angles()}
