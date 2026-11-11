"""Video validation, container-metadata inspection and frame sources.

FrameSource is the only thing the processing pipeline needs; Mp4FrameSource implements it for uploaded
files.  A future webcam/RTSP source only has to implement the same three members.
"""
import os, struct, time
from typing import Iterator, Optional, Protocol, Tuple
import cv2
import numpy as np


class VideoError(Exception):
    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint

    def to_dict(self):
        return {"code": self.code, "message": self.message, "hint": self.hint}


# --------------------------------------------------------------------------- MP4 box parsing
_CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"udta", b"edts", b"dinf", b"ilst", b"mvex"}
_CAMERA_KEYS = {b"\xa9mak": "make", b"\xa9mod": "model", b"\xa9xyz": "gps_location", b"\xa9swr": "software",
                b"\xa9day": "date", b"\xa9too": "encoder", b"\xa9nam": "title", b"\xa9cmt": "comment"}


def _boxes(f, start, end, depth, out, path):
    f.seek(start)
    pos = start
    while pos + 8 <= end and len(out["boxes"]) < 4000:
        f.seek(pos)
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        size, typ = struct.unpack(">I4s", hdr)
        hlen = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]; hlen = 16
        elif size == 0:
            size = end - pos
        if size < hlen or pos + size > end + 8:
            out["problems"].append(f"malformed box {typ!r} at {pos}")
            break
        p = path + "/" + typ.decode("latin-1")
        out["boxes"].append(p)
        body = pos + hlen
        if typ == b"mvhd":
            f.seek(body); v = f.read(1)[0]; f.seek(body + (20 if v == 1 else 12))
            ts, dur = (struct.unpack(">IQ", f.read(12)) if v == 1 else struct.unpack(">II", f.read(8)))
            out["movie_timescale"], out["movie_duration_units"] = ts, dur
        elif typ == b"stsd":
            f.seek(body + 8 + 4)
            out.setdefault("sample_entries", []).append(f.read(4).decode("latin-1"))
        elif typ == b"hdlr":
            f.seek(body + 8)
            out.setdefault("handlers", []).append(f.read(4).decode("latin-1"))
        elif typ in _CAMERA_KEYS and depth >= 2:
            f.seek(body)
            raw = f.read(min(size - hlen, 512))
            i = raw.find(b"data")
            txt = raw[i + 12:] if i >= 0 else raw[4:]
            out["tags"][_CAMERA_KEYS[typ]] = txt.decode("utf-8", "replace").strip("\x00 ")
        elif typ == b"meta":
            f.seek(body)
            peek = f.read(8)
            inner = body + (4 if peek[4:8] != b"hdlr" else 0)   # ISO meta has a 4-byte version/flags header
            _boxes(f, inner, pos + size, depth + 1, out, p)
        elif typ in _CONTAINERS:
            _boxes(f, body, pos + size, depth + 1, out, p)
        pos += size


def parse_mp4(path):
    out = {"boxes": [], "tags": {}, "problems": []}
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[4:8] != b"ftyp":
            raise VideoError("not_mp4", "The file is not an MP4/ISO-BMFF container (no 'ftyp' box at the start).",
                             "Export or convert the recording to .mp4 (for example: ffmpeg -i input -c:v libx264 output.mp4).")
        out["brand"] = head[8:12].decode("latin-1")
        _boxes(f, 0, os.path.getsize(path), 0, out, "")
    if not any(b.endswith("/moov") for b in out["boxes"]):
        raise VideoError("corrupted", "The MP4 has no 'moov' (index) box; the file is truncated or corrupted.",
                         "Re-export the video, or re-download it if the copy was interrupted.")
    return out


# --------------------------------------------------------------------------- inspection
def inspect_video(path):
    t0 = time.time()
    meta = parse_mp4(path)
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise VideoError("cannot_open", "OpenCV/FFmpeg could not open the video stream.",
                         "The codec may be unsupported. Convert to H.264: ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4")
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1", "replace").strip("\x00")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, first = cap.read()
    decoded_probe = 0
    while ok and decoded_probe < 5:
        decoded_probe += 1
        ok, _ = cap.read()
    last_ok = False
    if n > 10:
        cap.set(cv2.CAP_PROP_POS_FRAMES, n - 3)
        last_ok, _ = cap.read()
    cap.release()
    if first is None or decoded_probe == 0:
        raise VideoError("cannot_decode", f"The video stream (codec '{fourcc or 'unknown'}') could not be decoded.",
                         "Convert to H.264: ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p output.mp4")
    if fps <= 0 or fps > 1000:
        raise VideoError("bad_fps", f"The container reports an invalid frame rate ({fps}).", "Re-encode the video with a constant frame rate.")
    if w < 160 or h < 120:
        raise VideoError("too_small", f"Resolution {w}x{h} is too small for road analysis.", "Use a video of at least 640x360.")
    if n < 30:
        raise VideoError("too_short", f"Only {n} frames; at least 30 are needed for automatic calibration.", "Use a longer clip.")
    duration = n / fps
    if meta.get("movie_timescale"):
        duration = meta["movie_duration_units"] / meta["movie_timescale"] or duration
    camera_tags = {k: v for k, v in meta["tags"].items() if k in ("make", "model", "gps_location", "software")}
    return {
        "resolution": [w, h], "fps": round(fps, 3), "frame_count": n, "duration_s": round(duration, 3),
        "codec_fourcc": fourcc, "container_brand": meta.get("brand"), "sample_entries": meta.get("sample_entries", []),
        "track_handlers": meta.get("handlers", []), "metadata_tags": meta["tags"], "camera_related_tags": camera_tags,
        "camera_intrinsics_in_metadata": False,
        "metadata_note": ("No camera intrinsics or mounting extrinsics are stored in standard MP4 boxes; "
                          + ("camera make/model tags were found but are reported only, not used to look up parameters."
                             if camera_tags.get("make") or camera_tags.get("model") else "no camera make/model tags were found.")),
        "end_of_stream_decodes": bool(last_ok), "container_problems": meta["problems"],
        "inspection_time_s": round(time.time() - t0, 3), "file_size_bytes": os.path.getsize(path),
    }


# --------------------------------------------------------------------------- frame sources
class FrameSource(Protocol):
    fps: float
    frame_count: Optional[int]

    def frames(self) -> Iterator[Tuple[int, float, np.ndarray]]: ...
    def close(self) -> None: ...


class Mp4FrameSource:
    """Sequential decoder: yields (frame_index, timestamp_s, BGR frame). Never loads the whole file."""

    def __init__(self, path, fps, frame_count):
        self.path, self.fps, self.frame_count = path, fps, frame_count
        self.cap = None

    def frames(self):
        self.cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise VideoError("cannot_open", "Video could not be reopened for processing.")
        i = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else i / self.fps
            yield i, ts, frame
            i += 1
        self.close()

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def read_frames_at(path, indices, burst=1, max_width=None):
    """Random-access helper for calibration sampling: returns {index: [frame, frame+1, ...]}."""
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    out = {}
    try:
        for i in sorted(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            seq = []
            for _ in range(burst):
                ok, fr = cap.read()
                if not ok:
                    break
                if max_width and fr.shape[1] > max_width:
                    s = max_width / fr.shape[1]
                    fr = cv2.resize(fr, (max_width, int(round(fr.shape[0] * s))), interpolation=cv2.INTER_AREA)
                seq.append(fr)
            if seq:
                out[int(i)] = seq
    finally:
        cap.release()
    return out
