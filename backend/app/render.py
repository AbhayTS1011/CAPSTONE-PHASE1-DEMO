"""Rendering of the three synchronized panels in the style of the earlier PSJ_01 final-demo video.

Panels B and C are drawn as figure-style plots: light page, axes with tick labels (camera heights, h),
hatched ground the camera cannot see, an approximate ego-vehicle box with a heading arrow, and the
demo's class colours (road boundary green, lane divider orange, pedestrian crossing blue).
Static parts (axes, ticks, labels, hatching, grid) are drawn ONCE; each frame only pastes the BEV and the
vectors, so rendering stays fast.
"""
import os
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from .ipm import BEV_W, BEV_H

COLORS = {  # BGR, matplotlib tab: colours of the earlier demo
    "road_boundary": (44, 160, 44),
    "lane_divider": (14, 127, 255),
    "lane_divider_yellow": (14, 127, 255),
    "pedestrian_crossing": (180, 119, 31),
    "transverse_marking": (180, 119, 31),
    "lane_marking": (14, 127, 255),
    "lane_marking_yellow": (14, 127, 255),
    "oblique_marking": (14, 127, 255),
    "road_edge_candidate": (44, 160, 44),
}
ORIGINAL_W = 1000
PAGE = (249, 252, 253)                   # BGR of the demo's off-white page
PLOT_BG_C = (17, 17, 17)
HATCH_BASE, HATCH_LINE = (64, 64, 64), (150, 150, 150)
EGO_RED = (46, 16, 200)
ML, MR, MT, MB = 64, 18, 12, 56          # figure margins (px) around the plot area
BEHIND_H = 2.6                           # plot extends this far behind the camera (h) to show the ego box


def font(size, bold=False):
    for name in (("arialbd.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf")):
        for d in ("", r"C:\Windows\Fonts", "/usr/share/fonts/truetype/dejavu", "/Library/Fonts"):
            p = os.path.join(d, name) if d else name
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _hatch(h, w, base, line, step=16):
    img = np.full((h, w, 3), base, np.uint8)
    for k in range(-h, w, step):
        cv2.line(img, (k, h), (k + h, 0), line, 1, cv2.LINE_AA)
    return img


class PanelRenderer:
    def __init__(self, projector, cam, horizon_row=None):
        self.P, self.cam = projector, cam
        P = projector
        self.inv = ~projector.valid
        self.strip = int(round(BEHIND_H / P.res))           # rows below X = 0 (behind the camera)
        self.ph, self.pw = BEV_H + self.strip, BEV_W        # plot area
        self.fh, self.fw = MT + self.ph + MB, ML + self.pw + MR
        self.s = ORIGINAL_W / cam.width
        self.orig_h = int(round(cam.height * self.s))
        hatch = _hatch(self.ph, self.pw, HATCH_BASE, HATCH_LINE)
        self.hatch = hatch
        inv_full = np.ones((self.ph, self.pw), bool)
        inv_full[:BEV_H] = self.inv
        self.inv_full = inv_full
        grid, ticks_x, ticks_y = self._grid()
        self.grid = grid
        base = self._figure_base(ticks_x, ticks_y)
        # panel B: page + axes; IPM pasted per frame
        self.base_b = base.copy()
        # panel C: dark plot, hatched unseen ground, grid, ego -- fully static
        c = np.full((self.ph, self.pw, 3), PLOT_BG_C, np.uint8)
        c[inv_full] = hatch[inv_full]
        c = self._overlay_static(c, grid_alpha=0.30)
        self.plot_c = c
        self.base_c = base.copy()

    # ---------------------------------------------------------------- static layers
    def _row(self, X):
        _, r = self.P.ground_to_px(np.array([X]), np.array([0.0]))
        return float(r[0])

    def _col(self, Y):
        c, _ = self.P.ground_to_px(np.array([0.0]), np.array([Y]))
        return float(c[0])

    def _grid(self):
        P = self.P
        g = np.zeros((self.ph, self.pw), bool)
        step = 2.0 if P.roi["x_max"] < 30 else 5.0
        ticks_x, ticks_y = [], []
        x = 0.0
        while x <= P.roi["x_max"] + 1e-6:
            r = int(round(self._row(x)))
            if 0 <= r < self.ph:
                g[r, :] = True
                ticks_x.append((x, r))
            x += step
        y = 0.0
        while y <= P.y_max + 1e-6:
            for yy in sorted({y, -y}):
                c = int(round(self._col(yy)))
                if 0 <= c < self.pw:
                    g[:, c] = True
                    ticks_y.append((yy, c))
            y += step
        return g, ticks_x, ticks_y

    def _figure_base(self, ticks_x, ticks_y):
        img = Image.new("RGB", (self.fw, self.fh), PAGE[::-1])
        d = ImageDraw.Draw(img)
        f, fl = font(13), font(14)
        x0, y0, x1, y1 = ML, MT, ML + self.pw, MT + self.ph
        for x, r in ticks_x:
            yy = MT + r
            d.line([(x0 - 5, yy), (x0, yy)], fill=(60, 60, 60))
            t = f"{x:g}"
            w = d.textlength(t, font=f)
            d.text((x0 - 8 - w, yy - 8), t, fill=(30, 30, 30), font=f)
        for y, c in ticks_y:
            xx = ML + c
            d.line([(xx, y1), (xx, y1 + 5)], fill=(60, 60, 60))
            t = f"{y:g}".replace("-", "\u2212")
            w = d.textlength(t, font=f)
            d.text((xx - w / 2, y1 + 8), t, fill=(30, 30, 30), font=f)
        lab = "Y lateral (h),  + = left"
        d.text(((x0 + x1) / 2 - d.textlength(lab, font=fl) / 2, y1 + 28), lab, fill=(30, 30, 30), font=fl)
        tmp = Image.new("RGB", (220, 22), PAGE[::-1])
        ImageDraw.Draw(tmp).text((0, 2), "X forward (h)", fill=(30, 30, 30), font=fl)
        tmp = tmp.rotate(90, expand=True)
        img.paste(tmp, (6, (y0 + y1) // 2 - 110))
        d.rectangle([x0 - 1, y0 - 1, x1, y1], outline=(200, 190, 140), width=1)
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def _overlay_static(self, plot, grid_alpha):
        out = plot.copy()
        out[self.grid] = (out[self.grid] * (1 - grid_alpha) + np.array([210, 210, 210]) * grid_alpha).astype(np.uint8)
        r0 = int(round(self._row(0.0)))                         # dashed yellow line: ROI start (below the camera)
        for c in range(0, self.pw, 12):
            cv2.line(out, (c, r0), (min(c + 6, self.pw - 1), r0), (60, 200, 240), 2, cv2.LINE_AA)
        # approximate ego vehicle (box size assumes a typical car; the scale is relative, so it is indicative only)
        cy0, cy1 = self._row(0.4), self._row(-2.3)
        cx0, cx1 = self._col(0.6), self._col(-0.6)
        cv2.rectangle(out, (int(cx0), int(cy0)), (int(cx1), int(cy1)), EGO_RED, -1)
        cv2.rectangle(out, (int(cx0), int(cy0)), (int(cx1), int(cy1)), (245, 245, 245), 1, cv2.LINE_AA)
        cc, ca = int(round((cx0 + cx1) / 2)), int(round(self._row(-1.2)))
        cv2.arrowedLine(out, (cc, ca), (cc, int(self._row(1.6))), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.12)
        cv2.circle(out, (cc, ca), 3, (255, 255, 255), -1, cv2.LINE_AA)
        return out

    def _compose(self, base, plot):
        fig = base.copy()
        fig[MT:MT + self.ph, ML:ML + self.pw] = plot
        return fig

    # ---------------------------------------------------------------- per frame
    def original(self, frame):
        return cv2.resize(frame, (ORIGINAL_W, self.orig_h), interpolation=cv2.INTER_AREA)

    def ipm(self, bev):
        plot = np.empty((self.ph, self.pw, 3), np.uint8)
        plot[:BEV_H] = bev
        plot[self.inv_full] = self.hatch[self.inv_full]
        plot = self._overlay_static(plot, grid_alpha=0.18)
        return self._compose(self.base_b, plot)

    def vector(self, vecs):
        plot = self.plot_c.copy()
        for v in vecs:
            col = COLORS.get(v["type"], (200, 200, 200))
            pts = np.round(v["px"] * 4).astype(np.int32).reshape(-1, 1, 2)   # sub-pixel (shift=2) for smooth curves
            closed = v["type"] == "pedestrian_crossing"
            cv2.polylines(plot, [pts], closed, col, 3, cv2.LINE_AA, shift=2)
        if not vecs:
            f = cv2.FONT_HERSHEY_SIMPLEX
            for k, m in enumerate(["No reliable vector candidates", "detected in this frame."]):
                (tw, _), _ = cv2.getTextSize(m, f, 0.6, 1)
                cv2.putText(plot, m, ((self.pw - tw) // 2, BEV_H // 2 + 26 * k), f, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        return self._compose(self.base_c, plot)
