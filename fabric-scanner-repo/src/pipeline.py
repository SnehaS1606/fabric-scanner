"""
pipeline.py -- inference core used by the web app (server.py).

Engine      : the real thing  (3-branch classifier + Whisper + Moondream2 + knowledge base)
MockEngine  : stand-in used only with `--mock`, so the web interface can be tried without any model.
              It returns a FIXED made-up prediction and is clearly labelled as demo mode in the UI.

No audio hardware libraries are needed here: the browser records the microphone and speaks the answers.
"""

import glob
import os
import re
import threading
import time

import cv2
import numpy as np

from fabric_kb import answer_from_kb, retrieve_facts

CLASS_NAMES = ["Cotton", "Denim", "Silk", "Wool", "Polyester", "Leather"]
IMG_SIZE = 224


# ----------------------------------------------------------------- helpers (no torch needed)
def center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return img[y:y + s, x:x + s]


def gray_mean(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def describe_result(probs):
    order = np.argsort(-probs)
    top, second = order[0], order[1]
    if probs[top] >= 0.6:
        return f"This looks like {CLASS_NAMES[top].lower()}, with {probs[top] * 100:.0f} percent confidence."
    return (f"I'm not sure. It could be {CLASS_NAMES[top].lower()} at {probs[top] * 100:.0f} percent, "
            f"or {CLASS_NAMES[second].lower()} at {probs[second] * 100:.0f} percent.")


def list_samples(data_root):
    """{class: [sample ids]} for dataset samples that have all of im_1..im_4."""
    out = {c: [] for c in CLASS_NAMES}
    if not data_root:
        return out
    for c in CLASS_NAMES:
        d = os.path.join(data_root, c)
        if not os.path.isdir(d):
            continue
        for s in sorted(os.listdir(d)):
            p = os.path.join(d, s)
            if os.path.isdir(p) and all(glob.glob(os.path.join(p, f"im_{i}.*")) for i in range(1, 5)):
                out[c].append(s)
    return out


def load_sample_pair(data_root, cls, sample_id):
    """Darkest and brightest of the four illumination images (same convention as training)."""
    folder = os.path.join(data_root, cls, str(sample_id))
    paths = [glob.glob(os.path.join(folder, f"im_{i}.*"))[0] for i in range(1, 5)]
    imgs = [cv2.imread(p) for p in paths]
    b = [gray_mean(i) for i in imgs]
    return imgs[int(np.argmin(b))], imgs[int(np.argmax(b))]


def build_prompt(probs, question, history):
    order = np.argsort(-probs)
    top = order[0]
    prev = "".join(f"Earlier the shopper asked: {q} " for q, _ in history[-2:])
    facts = retrieve_facts(probs, CLASS_NAMES)
    return (
        f"Facts: {facts} "
        f"The classifier says this fabric is {CLASS_NAMES[top]} ({probs[top] * 100:.0f}% confidence). "
        f"{prev}Question: {question} "
        f"Answer the question in one or two sentences using only the facts above."
    )


def add_uncertainty_note(answer, probs):
    already = ("classifier" in answer) or ("photo alone" in answer)
    if probs.max() < 0.7 and not already:
        answer += (f" I'm only {probs.max() * 100:.0f} percent sure this is "
                   f"{CLASS_NAMES[int(probs.argmax())].lower()}.")
    return answer


# ----------------------------------------------------------------- real engine
class Engine:
    mock = False

    def __init__(self, checkpoint, data_root=None, whisper_model="small", load_vlm=True):
        import torch
        import whisper
        from torchvision import transforms
        from dataset_ibug import align_and_diff
        from model import FusionFabricClassifier

        self.torch = torch
        self.align_and_diff = align_and_diff
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.lock = threading.Lock()        # classifier + Whisper
        self.vlm_lock = threading.Lock()    # Moondream2 only, so it never blocks scanning
        self.data_root = data_root
        self.samples = list_samples(data_root)

        print("Loading classifier...")
        self.clf = FusionFabricClassifier(num_classes=len(CLASS_NAMES)).to(self.device)
        self.clf.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.clf.eval()

        print(f"Loading Whisper ({whisper_model})...")
        self.stt = whisper.load_model(whisper_model)

        self.vlm = self.tok = None
        self.vlm_state = "loading" if load_vlm else "off"
        if load_vlm:
            threading.Thread(target=self._load_vlm, daemon=True).start()

    def _load_vlm(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            print("Loading Moondream2 in the background (the app is usable meanwhile)...")
            self.vlm = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
            self.tok = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
            self.vlm_state = "ready"
            print("Moondream2 ready.")
        except Exception as e:                       # keep the app alive without the VLM
            print("Moondream2 failed to load:", e)
            self.vlm_state = "failed"

    # --- classification
    def scan(self, off_bgr, on_bgr):
        off = cv2.resize(center_square(off_bgr), (IMG_SIZE, IMG_SIZE))
        on = cv2.resize(center_square(on_bgr), (IMG_SIZE, IMG_SIZE))
        if gray_mean(off) > gray_mean(on):           # same convention as training: darker = flash-OFF
            off, on = on, off
        flash_effect = float(np.mean(cv2.absdiff(cv2.cvtColor(off, cv2.COLOR_BGR2GRAY),
                                                 cv2.cvtColor(on, cv2.COLOR_BGR2GRAY))))
        diff = self.align_and_diff(off, on)
        off_t = self.tf(cv2.cvtColor(off, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        diff_t = self.tf(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with self.lock, self.torch.no_grad():
            probs = self.torch.softmax(self.clf(off_t, diff_t), dim=1)[0].cpu().numpy()
        return {"probs": probs, "off": off, "on": on, "diff": diff, "flash_effect": flash_effect}

    def warm(self, sess):
        """Encode the scanned image for the VLM in the background so the first VLM question is faster."""
        if self.vlm_state != "ready":
            return

        def work():
            try:
                with self.vlm_lock:
                    if "enc" not in sess:
                        from PIL import Image
                        sess["enc"] = self.vlm.encode_image(
                            Image.fromarray(cv2.cvtColor(sess["off"], cv2.COLOR_BGR2RGB)))
            except Exception as e:
                print("VLM warm-up failed:", e)
        threading.Thread(target=work, daemon=True).start()

    # --- speech to text
    def transcribe(self, path):
        with self.lock:
            result = self.stt.transcribe(
                path, language="en",
                initial_prompt="A short question about a fabric.",
                condition_on_previous_text=False,
                no_speech_threshold=0.5,
                fp16=False,
            )
        segs = result.get("segments", [])
        if segs and np.mean([s["no_speech_prob"] for s in segs]) > 0.5:
            return ""
        text = result["text"].strip()
        if len(text.split()) > 25:                   # long output on noise = hallucination
            return ""
        return text

    # --- question answering
    def answer(self, sess, question):
        probs = sess["probs"]
        answer, route = answer_from_kb(question, probs, CLASS_NAMES)
        if route in ("visual", "open"):
            if self.vlm_state != "ready":
                answer = ("The vision model is not ready, so I can only answer questions about "
                          "the fabric's properties, such as season, durability and care.")
                route = "vlm_unavailable"
            else:
                with self.vlm_lock:
                    if "enc" not in sess:
                        from PIL import Image
                        sess["enc"] = self.vlm.encode_image(
                            Image.fromarray(cv2.cvtColor(sess["off"], cv2.COLOR_BGR2RGB)))
                    prompt = question if route == "visual" else build_prompt(probs, question, sess["history"])
                    answer = self.vlm.answer_question(sess["enc"], prompt, self.tok)
                answer = " ".join(re.split(r"(?<=[.!?])\s+", answer.strip())[:2])
                if "Earlier the shopper asked" in answer or "Facts:" in answer or not answer:
                    answer = "Sorry, I didn't understand that. Could you ask it another way?"
        return add_uncertainty_note(answer, probs), route


# ----------------------------------------------------------------- mock engine (UI testing only)
class MockEngine:
    mock = True
    vlm_state = "off"

    def __init__(self, data_root=None):
        self.data_root = data_root
        self.samples = list_samples(data_root)

    def scan(self, off_bgr, on_bgr):
        off = cv2.resize(center_square(off_bgr), (IMG_SIZE, IMG_SIZE))
        on = cv2.resize(center_square(on_bgr), (IMG_SIZE, IMG_SIZE))
        if gray_mean(off) > gray_mean(on):
            off, on = on, off
        flash_effect = float(np.mean(cv2.absdiff(cv2.cvtColor(off, cv2.COLOR_BGR2GRAY),
                                                 cv2.cvtColor(on, cv2.COLOR_BGR2GRAY))))
        diff = cv2.absdiff(on, off)
        probs = np.array([0.03, 0.90, 0.01, 0.02, 0.03, 0.01])      # FIXED fake output
        return {"probs": probs, "off": off, "on": on, "diff": diff, "flash_effect": flash_effect}

    def transcribe(self, path):
        return "Is this good for summer?"

    def answer(self, sess, question):
        answer, route = answer_from_kb(question, sess["probs"], CLASS_NAMES)
        if route in ("visual", "open"):
            answer, route = "(demo mode) The vision model is not loaded.", "vlm_unavailable"
        return add_uncertainty_note(answer, sess["probs"]), route
