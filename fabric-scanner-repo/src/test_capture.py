"""
test_capture.py -- 1-minute check of the flash-pair capture, WITHOUT loading any AI model.

    python test_capture.py --camera 0                          # laptop webcam + screen flash
    python test_capture.py --phone_url http://192.168.1.20:8080   # phone with the IP Webcam app

Saves captures/test_flash_off.png, test_flash_on.png, test_sheen_diff.png and prints how strong the
flash effect was. Open the three images: the sheen image should show the fabric's shiny areas.
"""
import argparse
import os
import time

import cv2
import numpy as np

from capture_tools import capture_flash_pair_phone, capture_flash_pair_screen, flash_effect

p = argparse.ArgumentParser()
p.add_argument("--camera", type=int, default=0)
p.add_argument("--phone_url", default=None)
args = p.parse_args()

os.makedirs("captures", exist_ok=True)
print("Hold the fabric about 10-15 cm from the camera (close to the phone camera in phone mode).")
for n in (3, 2, 1):
    print(f"Capturing in {n}...")
    time.sleep(1)

if args.phone_url:
    off, on = capture_flash_pair_phone(args.phone_url)
else:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")
    off, on = capture_flash_pair_screen(cap)
    cap.release()

diff = cv2.absdiff(cv2.resize(on, (off.shape[1], off.shape[0])), off)
cv2.imwrite("captures/test_flash_off.png", off)
cv2.imwrite("captures/test_flash_on.png", on)
cv2.imwrite("captures/test_sheen_diff.png", np.clip(diff.astype(np.int32) * 3, 0, 255).astype(np.uint8))

eff = flash_effect(off, on)
print(f"\nMean brightness change caused by the flash: {eff:.1f}")
if eff < 3:
    print("WARNING: the flash had almost no effect. Move closer, dim the room lights, or the camera's "
          "auto-exposure is cancelling the flash. The sheen map will carry little information.")
else:
    print("OK: the flash produced a measurable difference. Check captures/test_sheen_diff.png.")
