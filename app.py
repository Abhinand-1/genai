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


def text_to_speech(sentence):
    engine = pyttsx3.init()
    engine.save_to_file(sentence, "speech.mp3")
    engine.runAndWait()
    return "speech.mp3"




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

#_________________________
 # Use json.dumps to safely encode the string for JS
    safe_sentence = json.dumps(sentence)

    components.html(
        f"""
        <html>
          <body>
            <script>
              // Speak after a tiny timeout to ensure the component is mounted
              setTimeout(function() {{
                var text = {safe_sentence};
                var msg = new SpeechSynthesisUtterance(text);
                // Optional: set voice/rate/pitch if desired
                // msg.rate = 1.0;
                // msg.pitch = 1.0;
                window.speechSynthesis.speak(msg);
              }}, 100);
            </script>
          </body>
        </html>
        """,
        height=0,
        width=0,
    )
