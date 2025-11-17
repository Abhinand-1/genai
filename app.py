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
    
import streamlit as st
import base64
from murf import Murf, MurfApiError
import os

def murf_tts(sentence_to_speak, voice_id="en-US-matthew", output_filename="murf_audio.mp3"):
    """
    Generates an audio file from text using the Murf API,
    decodes the Base64 audio content, and saves it as an MP3.
    """
    st.write(f"Attempting to generate audio for: **'{sentence_to_speak}'**")
    
    # 1. API Initialization
    try:
        # Assumes MURF_API_KEY is set in st.secrets
        client = Murf(api_key=st.secrets["MURF_API_KEY"])
    except KeyError:
        st.error("MURF_API_KEY not found in Streamlit secrets. Please check your configuration.")
        return None

    # 2. Murf API Call
    try:
        response = client.text_to_speech.generate(
            text=sentence_to_speak,
            voice_id=voice_id,
            format="MP3",
            # CRITICAL: This tells Murf to return the audio data directly as Base64.
            encode_as_base_64=True 
        )
        
    except MurfApiError as e:
        # Handle API errors (e.g., 400 Bad Request, 403 Forbidden/Invalid Key)
        st.error(f"Murf API Error (Status {e.status_code}): {e.body}")
        st.write("Please check your API key, character limit, and voice ID.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Murf API call: {e}")
        return None

    # 3. CRITICAL: Retrieve Base64 Data
    # Use 'encoded_audio' for the Base64 string, not 'audio_file'
    base64_audio_data = getattr(response, 'encoded_audio', None)
    
    if not base64_audio_data:
        st.error("Error: Murf response did not contain the 'encoded_audio' data.")
        st.write(f"Received response object keys: {response.__dict__.keys()}")
        return None

    # 4. Decode and Write the Audio File
    try:
        # Decode the Base64 string into binary bytes
        decoded_audio_bytes = base64.b64decode(base64_audio_data)
        
        # Write the binary bytes to a file in **binary write mode ('wb')**
        with open(output_filename, "wb") as f:
            f.write(decoded_audio_bytes)
            
        st.success(f"✅ Audio generated and saved as `{output_filename}`.")
        return output_filename
        
    except Exception as e:
        st.error(f"Error during Base64 decoding or file writing: {e}")
        return None

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


    audio_path = murf_tts(sentence)
    if audio_path and os.path.exists(audio_path):
        audio_file = open(audio_path, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')


