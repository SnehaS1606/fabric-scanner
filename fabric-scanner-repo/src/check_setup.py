"""
check_setup.py -- run this before a demo or a long training run. Prints OK / FAIL for each requirement.
    python check_setup.py
"""
import importlib
import os
import shutil
import sys

DATA_DIR = "archive"
CLASSES = ["Cotton", "Denim", "Silk", "Wool", "Polyester", "Leather"]
fails = 0


def report(ok, text, fix=""):
    global fails
    print(("OK    " if ok else "FAIL  ") + text + ("" if ok else f"   -> {fix}"))
    fails += 0 if ok else 1


for mod, pip in [("torch", "torch"), ("torchvision", "torchvision"), ("cv2", "opencv-python"),
                 ("pywt", "pywavelets"), ("sklearn", "scikit-learn"), ("whisper", "openai-whisper"),
                 ("sounddevice", "sounddevice"), ("scipy", "scipy"), ("pyttsx3", "pyttsx3"),
                 ("transformers", "transformers"), ("PIL", "pillow")]:
    try:
        importlib.import_module(mod)
        report(True, f"python package {mod}")
    except Exception:
        report(False, f"python package {mod}", f"pip install {pip}")

report(shutil.which("ffmpeg") is not None, "ffmpeg on PATH (needed by Whisper)",
       "winget install ffmpeg, then open a NEW terminal")
report(os.path.exists("model_full.pt"), "trained checkpoint model_full.pt in this folder",
       "run: python train_ibug.py --checkpoint model_full.pt")
for c in CLASSES:
    report(os.path.isdir(os.path.join(DATA_DIR, c)), f"dataset folder {c}", f"expected at {DATA_DIR}/{c}")
try:
    import sounddevice as sd
    dev = sd.query_devices(kind="input")
    report(True, f"default microphone: {dev['name']}")
except Exception as e:
    report(False, "default microphone", f"check Windows microphone privacy settings ({type(e).__name__})")

print("\nAll checks passed." if fails == 0 else f"\n{fails} check(s) failed - fix the lines marked FAIL.")
sys.exit(1 if fails else 0)
