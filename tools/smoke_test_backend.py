"""End-to-end backend test over HTTP + WebSocket (the same protocol the browser uses).

  python tools/smoke_test_backend.py <video.mp4> [--server http://127.0.0.1:8000] [--no-pace] [--pause-test] [--out result.json]
Checks: upload/inspection, automatic calibration events, every frame delivered once and in order,
the three panels present in every frame message, pause/resume and stop behaviour.
"""
import argparse, asyncio, json, mimetypes, os, time, urllib.request, uuid
import websockets


def upload(server, path):
    b = uuid.uuid4().hex
    data = open(path, "rb").read()
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{os.path.basename(path)}\"\r\n"
            f"Content-Type: video/mp4\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(server + "/api/videos", data=body, method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


async def run(a):
    t0 = time.time()
    st, res = upload(a.server, a.video)
    print("upload", st, json.dumps(res)[:400])
    if st != 200:
        return {"upload_status": st, "response": res}
    sid = res["session_id"]
    ws_url = a.server.replace("http", "ws") + f"/ws/{sid}"
    out = {"video": os.path.basename(a.video), "inspection": res["inspection"], "stages": [], "frames": [], "states": []}
    async with websockets.connect(ws_url, max_size=50 * 2 ** 20) as ws:
        started = paused_once = False
        pause_t = None
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 300))
            t = m["type"]
            if t == "session_state" and m["state"] == "ready" and not started:
                t = "status"; m = {"state": "ready"}
            if t == "calibration_stage":
                out["stages"].append(m["stage"]); print("  stage:", m["stage"], "-", m["detail"])
            elif t == "calibration_result":
                c = m["calibration"]; out["calibration"] = {k: c[k] for k in ("status", "scale_mode", "warnings", "checks", "parameters", "elapsed_s") if k in c}
                print("  calibration:", c["status"], c.get("elapsed_s"), "s")
            elif t == "status":
                out["states"].append(m["state"])
                if m["state"] == "ready" and not started:
                    started = True
                    await ws.send(json.dumps({"cmd": "start", "pace": not a.no_pace}))
                if m["state"] in ("completed", "stopped", "error"):
                    out["summary"] = m.get("summary"); out["final_state"] = m["state"]; out["error"] = m.get("error")
                    break
                if m["state"] == "paused" and pause_t:
                    await asyncio.sleep(1.0)
                    out["frames_during_pause"] = len([f for f in out["frames"] if f["recv"] > pause_t + 0.3])
                    await ws.send(json.dumps({"cmd": "resume"}))
            elif t == "frame_update":
                keys_ok = all(m.get(k) for k in ("original_frame", "bev_frame", "vectorized_bev_frame"))
                out["frames"].append({"i": m["frame_index"], "ts": m["timestamp_seconds"], "panels": keys_ok,
                                      "n_vec": len(m["vector_candidates"]), "ms": m["metrics"]["processing_time_ms"],
                                      "eff": m["metrics"]["effective_fps"], "pfps": m["metrics"]["processing_fps"],
                                      "mem": m["metrics"]["memory_rss_mb"], "recv": time.time()})
                if a.pause_test and not paused_once and m["frame_index"] == 30:
                    paused_once = True; pause_t = time.time()
                    await ws.send(json.dumps({"cmd": "pause"}))
            elif t == "error":
                out["error"] = m; break
    idx = [f["i"] for f in out["frames"]]
    out["check"] = {
        "frames_received": len(idx), "expected": res["inspection"]["frame_count"],
        "in_order_no_gaps_no_duplicates": idx == list(range(len(idx))),
        "all_three_panels_every_frame": all(f["panels"] for f in out["frames"]),
        "frames_with_vectors": sum(f["n_vec"] > 0 for f in out["frames"]),
        "mean_processing_ms": round(sum(f["ms"] for f in out["frames"]) / max(1, len(idx)), 1),
        "max_memory_rss_mb": max((f["mem"] for f in out["frames"]), default=None),
        "total_wall_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out["check"]), "summary:", out.get("summary"), "pause:", out.get("frames_during_pause"))
    for f in out["frames"]:
        f.pop("recv")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--no-pace", action="store_true"); ap.add_argument("--pause-test", action="store_true"); ap.add_argument("--out")
    a = ap.parse_args()
    r = asyncio.run(run(a))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
