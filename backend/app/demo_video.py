"""Optional 1920x1080 demo-video recorder with the same layout as the earlier PSJ_01 final-demo MP4:
brown header bar, A = camera frame with info lines and legend, B = IPM, C = vectorized BEV, footer note.
Panel C here is extracted from the video with traditional CV (not map-derived); the texts say so.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw

from .render import font, PAGE

W, H = 1920, 1080
HEADER = (41, 58, 91)                  # BGR of the demo's brown header
LEGEND = [("road boundary", (44, 160, 44), "line"), ("lane divider", (14, 127, 255), "line"),
          ("pedestrian crossing", (180, 119, 31), "line"), ("ground not observable by the camera", (90, 90, 90), "hatch"),
          ("ROI start (ground below the camera)", (60, 200, 240), "dash"), ("ego vehicle (approx.), arrow = heading", (46, 16, 200), "box")]


class DemoRecorder:
    def __init__(self, path, fps, title, calibration):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.writer = None
        for cc in ("avc1", "mp4v"):
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (W, H))
            if w.isOpened():
                self.writer, self.codec = w, cc
                break
            w.release()
        if self.writer is None:
            raise RuntimeError("no MP4 encoder available for the demo video")
        self.frames = 0
        self.cal = calibration
        self.base = self._base(title)

    def _base(self, title):
        img = Image.new("RGB", (W, H), PAGE[::-1])
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 64], fill=HEADER[::-1])
        d.text((20, 15), title, fill=(255, 255, 255), font=font(26, True))
        d.text((20, 78), "A  |  Original dashcam frame (decoded from the uploaded MP4)", fill=(25, 25, 25), font=font(20, True))
        # legend
        lx, ly = 180, 850
        d.rectangle([lx, ly, lx + 680, ly + 112], outline=(200, 200, 200), fill=(255, 255, 255))
        d.text((lx + 250, ly + 6), "Legend (panels B and C)", fill=(30, 30, 30), font=font(15))
        f = font(15)
        for k, (lab, col, kind) in enumerate(LEGEND):
            x = lx + 12 + (k // 3) * 340
            y = ly + 32 + (k % 3) * 25
            rgb = col[::-1]
            if kind == "line":
                d.line([(x, y + 9), (x + 30, y + 9)], fill=rgb, width=4)
            elif kind == "dash":
                for t in range(0, 30, 8):
                    d.line([(x + t, y + 9), (x + t + 4, y + 9)], fill=rgb, width=2)
            elif kind == "box":
                d.rectangle([x, y + 3, x + 30, y + 15], fill=rgb)
            else:
                d.rectangle([x, y + 3, x + 30, y + 15], fill=(64, 64, 64))
                for t in range(-12, 30, 6):
                    d.line([(x + max(t, 0), y + 15 - max(0, -t)), (x + min(t + 12, 30), y + 3 + max(0, t + 12 - 30))], fill=(160, 160, 160))
            d.text((x + 40, y), lab, fill=(30, 30, 30), font=f)
        d.text((20, 1046), "IPM assumes a flat road (raised objects smear; hatched ground is not observable).  Panel C is extracted from "
                           "this video with traditional computer vision (no map data, no neural network).  Relative scale: 1 h = camera height.",
               fill=(60, 60, 60), font=font(15))
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _title(self, img, x, y, lines, size=17):
        pil = Image.fromarray(cv2.cvtColor(img[y:y + 24 * len(lines) + 4, x:x + 440], cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for k, t in enumerate(lines):
            f = font(size, True)
            d.text(((440 - d.textlength(t, font=f)) / 2, 2 + 22 * k), t, fill=(25, 25, 25), font=f)
        img[y:y + 24 * len(lines) + 4, x:x + 440] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def write(self, frame_index, total, ts, original, ipm_fig, vec_fig, vecs):
        img = self.base.copy()
        a = cv2.resize(original, (1000, int(original.shape[0] * 1000 / original.shape[1])), interpolation=cv2.INTER_AREA)
        img[110:110 + a.shape[0], 20:1020] = a
        bh = 900
        for fig, x in ((ipm_fig, 1030), (vec_fig, 1470)):
            sc = min(436 / fig.shape[1], (bh - 120) / fig.shape[0])   # fit the 440 px slot
            f = cv2.resize(fig, (int(fig.shape[1] * sc), int(fig.shape[0] * sc)), interpolation=cv2.INTER_AREA)
            ox = x + (440 - f.shape[1]) // 2
            img[180:180 + f.shape[0], ox:ox + f.shape[1]] = f
        self._title(img, 1030, 118, ["B  |  IPM bird's-eye view", f"frame {frame_index + 1}/{total}"])
        self._title(img, 1470, 108, ["C  |  Vectorized BEV", "(video-derived, traditional CV)", f"frame {frame_index + 1}/{total}"])
        counts = {}
        for v in vecs:
            key = {"road_boundary": "road boundary", "pedestrian_crossing": "pedestrian crossing"}.get(v["type"], "lane divider")
            counts[key] = counts.get(key, 0) + 1
        info = [f"Video frame {frame_index + 1:06d} / {total}   (t = {ts:.2f} s)",
                f"Automatic calibration: {self.cal.get('status', '?').replace('_', ' ')}, relative scale (h = camera height)",
                "Detected: road boundary {}, lane divider {}, pedestrian crossing: {}".format(
                    counts.get("road boundary", 0), counts.get("lane divider", 0), "present" if counts.get("pedestrian crossing") else "absent")]
        y0 = 110 + a.shape[0] + 14
        pil = Image.fromarray(cv2.cvtColor(img[y0:y0 + 96, 20:1020], cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for k, t in enumerate(info):
            d.text((0, 30 * k), t, fill=(30, 30, 30), font=font(19))
        img[y0:y0 + 96, 20:1020] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        self.writer.write(img)
        self.frames += 1
        return img

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
