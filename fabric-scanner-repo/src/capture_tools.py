"""
capture_tools.py -- flash-pair capture helpers (no PyTorch needed).

1) Phone flash (EXPERIMENTAL, gives a REAL flash): uses the free Android app "IP Webcam".
   Start its server, note the address it shows (e.g. http://192.168.1.20:8080), keep phone and laptop
   on the same Wi-Fi. This code calls the app's /shot.jpg, /enabletorch and /disabletorch URLs.
2) Webcam screen-flash: a laptop has no flash, so the screen is flashed white as the light source.
"""

import time
import urllib.request

import cv2
import numpy as np


def _get(url, timeout=8):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def phone_frame(base_url):
    data = _get(base_url.rstrip("/") + "/shot.jpg")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("The phone did not return a valid image from /shot.jpg")
    return img


def capture_flash_pair_phone(base_url, settle=0.8):
    """Flash-OFF photo, torch ON, flash-ON photo, torch OFF (torch is always switched off again)."""
    base = base_url.rstrip("/")
    off = phone_frame(base)
    _get(base + "/enabletorch")
    try:
        time.sleep(settle)                 # let the camera exposure settle under the torch
        on = phone_frame(base)
    finally:
        _get(base + "/disabletorch")
    return off, on


def capture_flash_pair_screen(cap):
    """Webcam: ambient frame, then a full-screen white flash of the display, then the lit frame."""
    for _ in range(5):
        cap.read()
    _, off = cap.read()
    cv2.namedWindow("flash", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("flash", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("flash", np.full((720, 1280, 3), 255, np.uint8))
    cv2.waitKey(1)
    t = time.time()
    while time.time() - t < 0.7:
        cap.read()
        cv2.waitKey(1)
    _, on = cap.read()
    cv2.destroyAllWindows()
    return off, on


def flash_effect(off, on):
    """Mean absolute brightness change between the two frames. Near zero = the flash had no effect
    (e.g. camera auto-exposure compensated), so the sheen map would carry no information."""
    a = cv2.cvtColor(off, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(cv2.resize(on, (off.shape[1], off.shape[0])), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean(np.abs(b - a)))
