import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import openai
import requests

# ----------------------
# Load Model
# ----------------------
model = load_model("best_model.h5")

# Label mapping
ALPHABET = [chr(ord('A') + i) for i in range(26)]
label_to_char = {i: ALPHABET[i] for i in range(26)}

letter_meaning = {
    "A": "Help",
    "B": "Water",
    "C": "Food",
    "D": "Call my family",
    "E": "I am OK",
    "F": "Stop",
    "G": "Yes",
    "H": "No",
    "I": "Thank you",
    "J": "Danger",
    "K": "I need support",
    "L": "Medicine",
    "M": "I am hungry",
    "N": "Bathroom",
    "O": "Emergency",
    "P": "Pick up the phone",
    "Q": "I am scared",
    "R": "Please wait",
    "S": "I am sick",
    "T": "Call doctor",
    "U": "Come here",
    "V": "Go away",
    "W": "Talk to me",
    "X": "I need rest",
    "Y": "Where are you?",
    "Z": "I need assistance"
}

# ----------------------
# Prediction Function
# ----------------------
def predict_letter_meaning(img):
    if img.ndim == 2:
        img = img.reshape(28, 28, 1)

    img = img.astype("float32") / 255.0
    preds = model.predict(img.reshape(1, 28, 28, 1))[0]

    idx = np.argmax(preds)
    letter = label_to_char[idx]
    meaning = letter_meaning.get(letter, "Unknown")

    return letter, meaning, float(preds[idx])

# ----------------------
# Generate Sentence (OpenAI)
# ----------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_sentence(meaning):
    prompt = f"Convert '{meaning}' into a clear, polite English sentence."

    client = openai.OpenAI(api_key=openai.api_key)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text

from murf import Murf
import base64

def murf_tts(text):
    client = Murf(api_key=st.secrets["MURF_API_KEY"])

    response = client.text_to_speech.generate(
        voice_id="en-US-natalie",   # You can change the voice here
        text=text,
        multi_native_locale="en-US"
    )

    # Murf returns BASE64 audio, so decode it:
    audio_bytes = base64.b64decode(response.audio_file)

    # Save mp3
    output_file = "murf_audio.mp3"
    with open(output_file, "wb") as f:
        f.write(audio_bytes)

    return output_file



# ----------------------
# Streamlit UI
# ----------------------
st.title("🤟 Sign Language → Meaning → Sentence → Audio Generator")

uploaded = st.file_uploader("Upload hand sign image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('L')
    st.image(img, caption="Uploaded Image", width=300)

    img_resized = img.resize((28, 28))
    img_array = np.array(img_resized)

    # Prediction
    letter, meaning, prob = predict_letter_meaning(img_array)

    st.subheader("Prediction")
    st.write(f"**Predicted Letter:** {letter}")
    st.write(f"**Meaning:** {meaning}")
    st.write(f"**Confidence:** {prob:.4f}")

    # Sentence
    sentence = generate_sentence(meaning)
    st.subheader("Generated Sentence")
    st.write(sentence)

    # Murf TTS
    audio_path = murf_tts(sentence)

    if audio_path:
        st.audio(audio_path, format="audio/mp3")
