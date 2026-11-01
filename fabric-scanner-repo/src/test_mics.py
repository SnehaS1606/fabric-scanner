import numpy as np
import sounddevice as sd

# Candidate input devices from list_devices.py
CANDIDATES = [1, 5, 13, 18]
SECONDS = 3
FS = 16000

for dev in CANDIDATES:
    input(f"\nPress Enter, then SPEAK continuously for {SECONDS}s (device {dev})...")
    try:
        audio = sd.rec(int(SECONDS * FS), samplerate=FS, channels=1,
                       dtype="int16", device=dev)
        sd.wait()
        peak = int(np.abs(audio).max())
        print(f"Device {dev}: peak level = {peak}  ->", "GOOD" if peak > 1000 else "too quiet / silent")
    except Exception as e:
        print(f"Device {dev}: failed ({e})")
