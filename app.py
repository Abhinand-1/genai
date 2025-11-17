import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
from gtts import gTTS
import openai
import pyttsx3
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import openai
import streamlit.components.v1 as components   # <-- ADD THIS

import base64
import os

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
# Generate Sentence using OpenAI API
# ----------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_sentence(meaning):
    prompt = (
        f"You receive a short command: '{meaning}'. "
        "Convert it into a clear, polite English sentence a person might say."
    )

    client = openai.OpenAI(api_key=openai.api_key)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt
    )

    return response.output_text

#----------------------------------
#            text to speech
#-----------------------------------


import requests
import streamlit as st

def murf_tts(text, voice="en-US-wavenet-D", format="mp3"):
    url = "https://api.murf.ai/v1/speech/generate"

    headers = {
        "apikey": st.secrets["MURF_API_KEY"],
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "voice": voice,
        "format": format
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        st.error(f"MURF API Error: {response.text}")
        return None

    # Murf returns binary audio in response.content
    output_file = "murf_audio.mp3"
    with open(output_file, "wb") as f:
        f.write(response.content)

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

    # Generate sentence
    sentence = generate_sentence(meaning)
    st.subheader("Generated Sentence")
    st.write(sentence)

    # Audio
audio_path = murf_tts(sentence)

if audio_path:
    st.audio(audio_path, format="audio/mp3")

 # Browser TTS
components.html(f"""
    <html>
      <body>
        <script>
            var text = "{sentence}";
            var msg = new SpeechSynthesisUtterance(text);
            window.speechSynthesis.speak(msg);
        </script>
      </body>
    </html>
""", height=0, width=0)

