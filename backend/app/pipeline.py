"""Per-frame processing, independent of where frames come from (MP4 today, webcam/RTSP later)."""
import base64, time
import numpy as np
import cv2

from .camera_model import CameraModel
from .ipm import BevProjector
from .vector_tracker import TrackedVectorizer
from .render import PanelRenderer

JPEG_QUALITY = 82


def camera_from_calibration(cal):
    c = cal["camera"]
    return CameraModel(c["image_size"][0], c["image_size"][1], c["focal_px"], c["principal_point_px"][0],
                       c["principal_point_px"][1], c["k1"], np.array(c["fwd"]), np.array(c["left"]), np.array(c["up"]))


def jpeg_b64(img, q=JPEG_QUALITY):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode("ascii")


class FramePipeline:
    def __init__(self, calibration):
        self.cal = calibration
        self.bev_enabled = calibration is not None and calibration["status"] != "failed"
        if not self.bev_enabled:                         # no trustworthy road plane: original frames only
            return
        self.cam = camera_from_calibration(calibration)
        roi = calibration["roi_units_h"]
        self.projector = BevProjector(self.cam, roi, calibration.get("static_mask_small"), calibration.get("hood_top_row_px"))
        self.vectorizer = TrackedVectorizer(self.projector, self.cam)
        self.renderer = PanelRenderer(self.projector, self.cam, calibration.get("horizon_row_px"))

    def reset(self):
        """Forget temporal state (call before every new pass over a video)."""
        if self.bev_enabled:
            self.vectorizer.reset()

    def process(self, frame):
        """Returns the three rendered panels for THIS frame plus vector data and timings (ms)."""
        if not self.bev_enabled:
            t0 = time.perf_counter()
            s = 800 / frame.shape[1]
            img = cv2.resize(frame, (800, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
            return ({"original_frame": jpeg_b64(img), "bev_frame": None, "vectorized_bev_frame": None}, [],
                    {"reason": "no validated road-plane transformation"}, {"encode_ms": (time.perf_counter() - t0) * 1e3})
        if frame.shape[1] != self.cam.width or frame.shape[0] != self.cam.height:
            raise ValueError(f"frame size {frame.shape[1]}x{frame.shape[0]} differs from the calibrated {self.cam.width}x{self.cam.height}")
        t0 = time.perf_counter()
        bev = self.projector.warp(frame)
        t1 = time.perf_counter()
        vecs, vinfo = self.vectorizer.update(frame, bev)
        t2 = time.perf_counter()
        p_orig = self.renderer.original(frame)
        p_ipm = self.renderer.ipm(bev)
        p_vec = self.renderer.vector(vecs)
        self.last_panels = (p_orig, p_ipm, p_vec)              # raw panels (used by the optional demo recorder)
        t3 = time.perf_counter()
        out = {"original_frame": jpeg_b64(p_orig), "bev_frame": jpeg_b64(p_ipm), "vectorized_bev_frame": jpeg_b64(p_vec)}
        t4 = time.perf_counter()
        for v in vecs:
            v.pop("px", None)
        return out, vecs, vinfo, {"ipm_ms": (t1 - t0) * 1e3, "vectorize_ms": (t2 - t1) * 1e3,
                                  "render_ms": (t3 - t2) * 1e3, "encode_ms": (t4 - t3) * 1e3}
