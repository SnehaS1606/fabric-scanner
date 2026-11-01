"""
full_pipeline_demo.py

This is the REAL end-to-end test:
  1. Records your actual spoken question from the microphone
  2. Transcribes it with Whisper (speech-to-text)
  3. Runs YOUR trained classifier on a real flash-off/flash-on image pair
     to get the actual predicted fabric class + confidence
  4. Feeds the transcribed question + real prediction into the VLM
  5. Prints the VLM's grounded answer

Run:
    python full_pipeline_demo.py
"""

import sounddevice as sd
from scipy.io.wavfile import write as write_wav
import whisper
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_ibug import find_illum_images, align_and_diff, brightness, TARGET_CLASSES
from model import FusionFabricClassifier
from torchvision import transforms

# ---------- CONFIG: change these two paths ----------
SAMPLE_FOLDER = "archive/Cotton/2"   # any real sample folder
CHECKPOINT_PATH = "model_full.pt"                    # your trained classifier
RECORD_SECONDS = 5
# ------------------------------------------------------

IMG_SIZE = 224
CLASS_NAMES = TARGET_CLASSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


INPUT_DEVICE = None  # set to a device number from list_devices.py if the default mic is wrong


def record_question():
    import time
    for n in (3, 2, 1):
        print(f"Get ready... recording in {n}")
        time.sleep(1)
    print(f"\n>>> Speak your question now ({RECORD_SECONDS} seconds)...")
    fs = 16000
    audio = sd.rec(int(RECORD_SECONDS * fs), samplerate=fs, channels=1,
                   dtype='int16', device=INPUT_DEVICE)
    sd.wait()
    write_wav("question.wav", fs, audio)
    peak = int(np.abs(audio).max())
    print(f">>> Done recording. Mic peak level: {peak} (should be well above 1000 when you speak)")
    if peak < 500:
        print(">>> WARNING: almost no sound captured. Wrong mic selected or mic blocked.")


def transcribe_question():
    print("Transcribing with Whisper...")
    stt_model = whisper.load_model("small")
    result = stt_model.transcribe(
        "question.wav",
        language="en",
        initial_prompt="Questions about a fabric material: cotton, silk, wool, denim, polyester, leather, "
                       "suitable for summer or winter clothing.",
    )
    text = result["text"].strip()
    print(f"You asked: \"{text}\"")
    return text


def classify_sample():
    print("Running your trained classifier on the sample...")
    img_paths = find_illum_images(SAMPLE_FOLDER)
    imgs = [cv2.resize(cv2.imread(p), (IMG_SIZE, IMG_SIZE)) for p in img_paths]

    brightnesses = [brightness(im) for im in imgs]
    off_img = imgs[int(np.argmin(brightnesses))]
    on_img = imgs[int(np.argmax(brightnesses))]
    diff_img = align_and_diff(off_img, on_img)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    off_rgb = cv2.cvtColor(off_img, cv2.COLOR_BGR2RGB)
    diff_rgb = cv2.cvtColor(diff_img, cv2.COLOR_BGR2RGB)
    off_tensor = transform(off_rgb).unsqueeze(0).to(device)
    diff_tensor = transform(diff_rgb).unsqueeze(0).to(device)

    model = FusionFabricClassifier(num_classes=len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(off_tensor, diff_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    predicted_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx].item()
    print(f"Classifier predicts: {predicted_class} (confidence {confidence:.1%})")
    return predicted_class, confidence, off_img


def ask_vlm(off_img_bgr, predicted_class, confidence, user_question):
    print("Loading VLM and generating grounded answer...")
    vlm = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")

    image_rgb = cv2.cvtColor(off_img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    encoded_image = vlm.encode_image(pil_image)

    prompt = (
        f"This fabric has been identified as {predicted_class} "
        f"({confidence:.0%} confidence). "
        f"A shopper asks: \"{user_question}\" "
        f"Answer in one or two complete sentences. Refer to the fabric as "
        f"{predicted_class} and give a reason based on its properties."
    )
    answer = vlm.answer_question(encoded_image, prompt, tokenizer)
    return answer


if __name__ == "__main__":
    record_question()
    user_question = transcribe_question()
    if not user_question:
        print("No speech detected. Fix the microphone and run again.")
        raise SystemExit(1)
    predicted_class, confidence, off_img = classify_sample()
    answer = ask_vlm(off_img, predicted_class, confidence, user_question)

    print("\n=== FULL PIPELINE RESULT ===")
    print(f"Predicted class : {predicted_class} ({confidence:.1%} confidence)")
    print(f"Your question   : {user_question}")
    print(f"VLM answer      : {answer}")
