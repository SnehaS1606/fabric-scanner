"""
server.py -- web app for the fabric identification assistant.

    python server.py --checkpoint model_full.pt --data_root "archive"
    then open  http://localhost:5000  in Chrome or Edge.

The browser does the camera, the hold-steady trigger, the flash (screen flash, or the real phone torch),
the microphone and the spoken answers. This server does the classification, Whisper speech-to-text,
the knowledge-base / Moondream2 answers, and keeps the latency log.

Options:
    --mock          interface test only, NO model (returns a fixed fake prediction, clearly labelled)
    --no_vlm        skip Moondream2 (faster start; visual/open questions get a polite 'not available')
    --https         serve over HTTPS so a PHONE can use its camera/mic/torch (needs: pip install cryptography)
"""

import argparse
import csv
import os
import random
import socket
import tempfile
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from pipeline import CLASS_NAMES, MockEngine, describe_result, load_sample_pair

HERE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(HERE, "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024

ENGINE = None
SESSIONS = OrderedDict()
SCAN_COUNT = 0
CAPTURE_DIR = os.path.join(HERE, "captures")
LOG_PATH = os.path.join(HERE, "latency_log.csv")
MIN_FLASH_EFFECT = 2.0       # mean grey-level change below this = the flash barely did anything


def jpg_uri(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    import base64
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def decode_upload(field):
    f = request.files.get(field)
    if f is None:
        raise ValueError(f"missing image '{field}'")
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"'{field}' is not a valid image")
    return img


def run_scan(off_bgr, on_bgr, extra=None):
    """Classify one flash pair, store the session, build the JSON reply."""
    global SCAN_COUNT
    t0 = time.time()
    res = ENGINE.scan(off_bgr, on_bgr)
    probs = res["probs"]
    SCAN_COUNT += 1
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    for name in ("off", "on", "diff"):
        cv2.imwrite(os.path.join(CAPTURE_DIR, f"scan{SCAN_COUNT}_{name}.png"), res[name])

    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"probs": probs, "off": res["off"], "history": []}
    while len(SESSIONS) > 20:
        SESSIONS.popitem(last=False)
    if hasattr(ENGINE, "warm"):
        ENGINE.warm(SESSIONS[sid])

    order = np.argsort(-probs)
    warning = None
    if res["flash_effect"] < MIN_FLASH_EFFECT:
        warning = ("The flash barely changed the picture, so the shine information is weak. "
                   "Hold the fabric closer to the camera and scan again.")
    reply = {
        "sid": sid,
        "top": CLASS_NAMES[int(order[0])],
        "confidence": float(probs[order[0]]),
        "message": describe_result(probs),
        "probs": [{"label": CLASS_NAMES[int(i)], "p": float(probs[i])} for i in order],
        "images": {"off": jpg_uri(res["off"]), "on": jpg_uri(res["on"]), "diff": jpg_uri(res["diff"])},
        "flash_effect": round(res["flash_effect"], 2),
        "warning": warning,
        "scan_seconds": round(time.time() - t0, 2),
        "mock": ENGINE.mock,
    }
    if extra:
        reply.update(extra)
    return reply


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/status")
def status():
    return jsonify({"mock": ENGINE.mock, "vlm": ENGINE.vlm_state,
                    "sample_counts": {c: len(v) for c, v in ENGINE.samples.items()}})


@app.route("/api/scan", methods=["POST"])
def scan():
    try:
        off, on = decode_upload("off"), decode_upload("on")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(run_scan(off, on, {"source": "camera", "flash_mode": request.form.get("flash_mode", "")}))


@app.route("/api/scan_sample", methods=["POST"])
def scan_sample():
    body = request.get_json(silent=True) or {}
    cls = body.get("cls")
    if cls not in CLASS_NAMES:
        cls = random.choice([c for c in CLASS_NAMES if ENGINE.samples.get(c)] or CLASS_NAMES)
    ids = ENGINE.samples.get(cls, [])
    if ENGINE.data_root and ids:
        sid_ = str(body.get("id") or random.choice(ids))
        if sid_ not in ids:
            return jsonify({"error": f"{cls}/{sid_} is not a complete sample"}), 400
        off, on = load_sample_pair(ENGINE.data_root, cls, sid_)
    elif ENGINE.mock:                                   # synthetic pair, interface testing only
        rng = np.random.default_rng(1)
        off = (rng.random((240, 240, 3)) * 90).astype(np.uint8)
        on = np.clip(off.astype(int) + 70, 0, 255).astype(np.uint8)
        sid_ = "synthetic"
    else:
        return jsonify({"error": "No dataset found. Start the server with --data_root <archive folder>."}), 400
    return jsonify(run_scan(off, on, {"source": "dataset", "actual": cls, "sample_id": sid_}))


@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "missing audio"}), 400
    ext = ".webm"
    ctype = (f.mimetype or "").lower()
    if "ogg" in ctype:
        ext = ".ogg"
    elif "mp4" in ctype or "aac" in ctype:
        ext = ".mp4"
    elif "wav" in ctype:
        ext = ".wav"
    fd, path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        f.save(path)
        t0 = time.time()
        text = ENGINE.transcribe(path)
        return jsonify({"text": text, "stt_seconds": round(time.time() - t0, 2)})
    except Exception as e:                              # most often: ffmpeg missing
        return jsonify({"error": f"Speech recognition failed: {e}"}), 500
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    sess = SESSIONS.get(body.get("sid"))
    question = (body.get("question") or "").strip()
    if sess is None:
        return jsonify({"error": "No fabric has been scanned yet. Scan a fabric first."}), 400
    if not question:
        return jsonify({"error": "empty question"}), 400
    t0 = time.time()
    answer, route = ENGINE.answer(sess, question)
    secs = time.time() - t0
    sess["history"].append((question, answer))
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["question", "stt_s", "answer_s", "route"])
        w.writerow([question, body.get("stt_seconds", ""), f"{secs:.2f}", route])
    return jsonify({"answer": answer, "route": route, "answer_seconds": round(secs, 2)})


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "your-pc-ip"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="model_full.pt")
    p.add_argument("--data_root", default="archive")
    p.add_argument("--whisper_model", default="small")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--mock", action="store_true")
    p.add_argument("--no_vlm", action="store_true")
    p.add_argument("--https", action="store_true")
    a = p.parse_args()

    data_root = a.data_root if os.path.isdir(a.data_root) else None
    if data_root is None:
        print(f"(dataset folder '{a.data_root}' not found: the 'scan a dataset sample' button will be unavailable)")

    if a.mock:
        ENGINE = MockEngine(data_root)
        print("*** MOCK MODE: no model is loaded; predictions are fake. For interface testing only. ***")
    else:
        from pipeline import Engine
        ENGINE = Engine(a.checkpoint, data_root, a.whisper_model, load_vlm=not a.no_vlm)

    scheme = "https" if a.https else "http"
    print(f"\nOpen  {scheme}://localhost:{a.port}  in Chrome or Edge on this computer.")
    if a.https:
        print(f"On a phone (same Wi-Fi) open  https://{lan_ip()}:{a.port}  and accept the certificate warning.")
    app.run(host="0.0.0.0" if a.https else "127.0.0.1", port=a.port, threaded=True,
            ssl_context="adhoc" if a.https else None)
