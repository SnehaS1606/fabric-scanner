"""
app.py  --  End-to-end fabric identification assistant

Pipeline:
  capture (hold-steady trigger + flash OFF/ON pair)
    -> alignment + sheen difference map
    -> 3-branch fusion classifier (spatial + DWT frequency + sheen)
    -> class probabilities
    -> multi-turn voice Q&A: Whisper (STT) -> Moondream2 VLM conditioned on
       the class probabilities + conversation history -> spoken answer (TTS)

Run (live webcam):
    python app.py --mode webcam
Run (reliable demo on a dataset sample):
    python app.py --mode dataset --sample_folder "archive/Denim/10"

Install once:
    pip install pyttsx3 sounddevice scipy openai-whisper opencv-python
"""

import argparse
import csv
import os
import re
import time

import cv2
import numpy as np
import sounddevice as sd
import torch
import whisper
import pyttsx3
from PIL import Image
from scipy.io.wavfile import write as write_wav
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_ibug import find_illum_images, align_and_diff, brightness, TARGET_CLASSES
from model import FusionFabricClassifier
from fabric_kb import retrieve_facts, answer_from_kb
from capture_tools import capture_flash_pair_phone

CLASS_NAMES = TARGET_CLASSES
IMG_SIZE = 224
FS = 16000
EXIT_WORDS = ("done", "quit", "exit", "stop", "new fabric", "next fabric")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ------------------------------------------------------------------ TTS
def speak(text):
    print(f"[ASSISTANT] {text}")
    engine = pyttsx3.init()  # re-init each time avoids a known pyttsx3 hang
    engine.say(text)
    engine.runAndWait()


# ------------------------------------------------------------------ models
def load_models(args):
    print("Loading classifier...")
    clf = FusionFabricClassifier(num_classes=len(CLASS_NAMES)).to(device)
    clf.load_state_dict(torch.load(args.checkpoint, map_location=device))
    clf.eval()
    print("Loading Whisper...")
    stt = whisper.load_model("small")
    print("Loading VLM (Moondream2)...")
    vlm = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
    return clf, stt, vlm, tok


# ------------------------------------------------------------------ capture
def center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


def wait_until_steady(cap, hold_seconds=2.0, motion_thresh=4.0):
    """Hold-to-capture trigger: returns once the scene has been still for hold_seconds."""
    prev, steady_since = None, None
    while True:
        ok, frame = cap.read()
        if not ok:
            return False
        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
        status = "Move fabric into view"
        if prev is not None:
            motion = float(np.mean(cv2.absdiff(gray, prev)))
            if motion < motion_thresh:
                steady_since = steady_since or time.time()
                held = time.time() - steady_since
                status = f"Hold steady... {held:.1f}/{hold_seconds:.0f}s"
                if held >= hold_seconds:
                    cv2.destroyAllWindows()
                    return True
            else:
                steady_since = None
        prev = gray
        view = frame.copy()
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Fabric capture (press q to cancel)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return False


def capture_flash_pair(cap):
    """Flash OFF frame, then a full-screen white 'screen flash', then flash ON frame.
    Laptop webcams have no flash, so the screen acts as the light source.
    Hold the fabric ~10-15 cm in front of the webcam."""
    for _ in range(5):
        cap.read()
    _, off = cap.read()

    cv2.namedWindow("flash", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("flash", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("flash", np.full((720, 1280, 3), 255, np.uint8))
    cv2.waitKey(1)
    t = time.time()
    while time.time() - t < 0.7:       # let the camera exposure settle
        cap.read()
        cv2.waitKey(1)
    _, on = cap.read()
    cv2.destroyAllWindows()
    return off, on


def prepare_inputs(off_bgr, on_bgr):
    off = cv2.resize(center_square(off_bgr), (IMG_SIZE, IMG_SIZE))
    on = cv2.resize(center_square(on_bgr), (IMG_SIZE, IMG_SIZE))
    if brightness(off) > brightness(on):        # same convention as training
        off, on = on, off
    diff = align_and_diff(off, on)
    off_t = tf(cv2.cvtColor(off, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    diff_t = tf(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
    return off_t, diff_t, off, on, diff


def load_dataset_pair(sample_folder):
    paths = find_illum_images(sample_folder)
    if paths is None:
        raise SystemExit(
            f"Sample folder not found or missing im_1..im_4: {sample_folder}\n"
            f"Run:  dir \"{os.path.dirname(sample_folder)}\"  and pick a folder number that exists."
        )
    imgs = [cv2.imread(p) for p in paths]
    b = [brightness(i) for i in imgs]
    return imgs[int(np.argmin(b))], imgs[int(np.argmax(b))]


# ------------------------------------------------------------------ classify
def classify(clf, off_t, diff_t):
    with torch.no_grad():
        probs = torch.softmax(clf(off_t, diff_t), dim=1)[0].cpu().numpy()
    return probs


def describe_result(probs):
    order = np.argsort(-probs)
    top, second = order[0], order[1]
    if probs[top] >= 0.6:
        return f"This looks like {CLASS_NAMES[top].lower()}, with {probs[top]*100:.0f} percent confidence."
    return (f"I'm not sure. It could be {CLASS_NAMES[top].lower()} at {probs[top]*100:.0f} percent, "
            f"or {CLASS_NAMES[second].lower()} at {probs[second]*100:.0f} percent.")


# ------------------------------------------------------------------ voice in
def record_question(mic, max_s=8, speech_thresh=1500):
    """Records until ~1s of silence after the person has started speaking."""
    chunk = int(0.2 * FS)
    frames, started, quiet = [], False, 0
    with sd.InputStream(samplerate=FS, channels=1, dtype="int16", device=mic) as stream:
        for _ in range(int(max_s / 0.2)):
            data, _ = stream.read(chunk)
            frames.append(data)
            level = int(np.abs(data).max())
            if level > speech_thresh:
                started, quiet = True, 0
            elif started:
                quiet += 1
                if quiet >= 5:
                    break
    write_wav("question.wav", FS, np.concatenate(frames))
    return started


def transcribe(stt):
    result = stt.transcribe(
        "question.wav", language="en",
        initial_prompt="A short question about a fabric.",
        condition_on_previous_text=False,
        no_speech_threshold=0.5,
    )
    segs = result.get("segments", [])
    if segs and np.mean([s["no_speech_prob"] for s in segs]) > 0.5:
        return ""                      # mostly non-speech: treat as silence
    text = result["text"].strip()
    if len(text.split()) > 25:         # real questions are short; long output = hallucination on noise
        return ""
    return text


# ------------------------------------------------------------------ VLM
def build_prompt(probs, question, history):
    order = np.argsort(-probs)
    top, second = order[0], order[1]
    # Only earlier QUESTIONS are passed (not earlier answers) so the small VLM
    # cannot copy its own previous reply.
    prev = "".join(f"Earlier the shopper asked: {q} " for q, _ in history[-2:])
    facts = retrieve_facts(probs, CLASS_NAMES)
    return (
        f"Facts: {facts} "
        f"The classifier says this fabric is {CLASS_NAMES[top]} ({probs[top]*100:.0f}% confidence). "
        f"{prev}Question: {question} "
        f"Answer the question in one or two sentences using only the facts above."
    )


def generate_answer(vlm, tok, enc, probs, question, history):
    """Routes the question:
       kb      -> structured retrieval from the fabric knowledge base (property questions)
       visual  -> VLM answers directly from the image (colour, pattern, texture...)
       open    -> VLM answers with the retrieved facts as context
    """
    answer, route = answer_from_kb(question, probs, CLASS_NAMES)
    if route in ("visual", "open"):
        prompt = question if route == "visual" else build_prompt(probs, question, history)
        answer = vlm.answer_question(enc, prompt, tok)
        answer = " ".join(re.split(r"(?<=[.!?])\s+", answer.strip())[:2])   # max two sentences
        if "Earlier the shopper asked" in answer or "Facts:" in answer or not answer:
            answer = "Sorry, I didn't understand that. Could you ask it another way?"
    already_states_class = ("classifier" in answer) or ("photo alone" in answer)
    if probs.max() < 0.7 and not already_states_class:
        answer += (f" I'm only {probs.max()*100:.0f} percent sure this is "
                   f"{CLASS_NAMES[int(probs.argmax())].lower()}.")
    return answer, route


# ------------------------------------------------------------------ conversation
def conversation(stt, vlm, tok, probs, off_rgb_pil, mic, log_writer):
    enc = vlm.encode_image(off_rgb_pil)          # encode the image once per capture
    history, silent = [], 0
    speak(describe_result(probs) + " Ask me a question, or say done to scan another fabric.")

    while True:
        time.sleep(1.0)   # let the assistant's voice and room echo die down before recording
        print("\nListening... (speak now)")
        t0 = time.time()
        heard = record_question(mic)
        t_rec = time.time()
        if not heard:
            silent += 1
            if silent >= 2:
                speak("I couldn't hear you. Ending this session.")
                return
            speak("I didn't hear anything. Please try again.")
            continue
        silent = 0

        question = transcribe(stt)
        t_stt = time.time()
        print(f"[YOU] {question}")
        if not question:
            speak("Sorry, I didn't catch that.")
            continue
        if any(w in question.lower() for w in EXIT_WORDS):
            speak("Okay.")
            return

        answer, route = generate_answer(vlm, tok, enc, probs, question, history)
        print(f"[route: {route}]")
        t_vlm = time.time()
        history.append((question, answer))
        speak(answer)
        t_end = time.time()

        row = [question, f"{t_stt - t_rec:.2f}", f"{t_vlm - t_stt:.2f}", f"{t_end - t_vlm:.2f}", route]
        log_writer.writerow(row)
        print(f"Latency  STT {row[1]}s | VLM {row[2]}s | TTS {row[3]}s")


# ------------------------------------------------------------------ main
def main(args):
    if args.mode == "phone" and not args.phone_url:
        raise SystemExit("Phone mode needs --phone_url http://<phone-ip>:8080")
    clf, stt, vlm, tok = load_models(args)
    os.makedirs("captures", exist_ok=True)
    cap = cv2.VideoCapture(args.camera) if args.mode == "webcam" else None

    with open("latency_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["question", "stt_s", "answer_s", "tts_s", "route"])

        scan = 0
        while True:
            t_cap = time.time()
            if args.mode == "webcam":
                speak("Hold the fabric steady in front of the camera.")
                if not wait_until_steady(cap):
                    break
                off_bgr, on_bgr = capture_flash_pair(cap)
            elif args.mode == "phone":
                speak("Hold the fabric close to the phone camera, then press Enter in the terminal.")
                input("Press Enter to capture... ")
                off_bgr, on_bgr = capture_flash_pair_phone(args.phone_url)
            else:
                off_bgr, on_bgr = load_dataset_pair(args.sample_folder)

            off_t, diff_t, off, on, diff = prepare_inputs(off_bgr, on_bgr)
            probs = classify(clf, off_t, diff_t)
            t_clf = time.time()
            print(f"Capture+classify time: {t_clf - t_cap:.2f}s")
            print({c: f"{p*100:.1f}%" for c, p in zip(CLASS_NAMES, probs)})

            scan += 1
            cv2.imwrite(f"captures/scan{scan}_flash_off.png", off)
            cv2.imwrite(f"captures/scan{scan}_flash_on.png", on)
            cv2.imwrite(f"captures/scan{scan}_sheen_diff.png", diff)

            pil = Image.fromarray(cv2.cvtColor(off, cv2.COLOR_BGR2RGB))
            conversation(stt, vlm, tok, probs, pil, args.mic, writer)
            f.flush()

            if input("\nPress Enter to scan another fabric, or type q to quit: ").strip().lower() == "q":
                break

    if cap:
        cap.release()
    print("Session ended. Latency saved to latency_log.csv, images saved in captures/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["webcam", "phone", "dataset"], default="dataset")
    p.add_argument("--phone_url", default=None, help="IP Webcam address, e.g. http://192.168.1.20:8080")
    p.add_argument("--sample_folder", default="archive/Cotton/2")
    p.add_argument("--checkpoint", default="model_full.pt")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--mic", type=int, default=None, help="input device index from list_devices.py")
    main(p.parse_args())
