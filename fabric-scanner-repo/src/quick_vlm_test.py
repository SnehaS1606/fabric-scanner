"""
quick_vlm_test.py
Minimal end-to-end test of the VLM-grounded Q&A idea, using ONE sample
image and a hardcoded predicted class (skip the full classifier for
now, just prove the VLM conditioning works).

Run:
    python quick_vlm_test.py
"""

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Step A: load Moondream2 (downloads automatically first run) ---
print("Loading model (first run downloads ~2GB, be patient)...")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")

# --- Step B: load your test image ---
# CHANGE THIS PATH to point at one real sample image from your archive
image_path = "archive/Cotton/2/im_1.jpg"
image = Image.open(image_path)
encoded_image = model.encode_image(image)

# --- Step C: fake classifier output for now (replace later with real prediction) ---
predicted_class = "cotton"
confidence = 0.91

# --- Step D: ask a question, grounded in the classifier's prediction ---
user_question = "Is this fabric suitable for summer clothing?"

prompt = (
    f"The fabric classifier predicts this material is: {predicted_class} "
    f"(confidence {confidence:.0%}). "
    f"User question: {user_question} "
    f"Answer concisely and helpfully, using the classifier's prediction as context."
)

answer = model.answer_question(encoded_image, prompt, tokenizer)

print("\n--- RESULT ---")
print(f"Predicted class: {predicted_class}")
print(f"Question: {user_question}")
print(f"Answer: {answer}")
