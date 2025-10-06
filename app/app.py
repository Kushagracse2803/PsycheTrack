import streamlit as st
import joblib
import librosa
import numpy as np
import os
from audiorecorder import audiorecorder
import tempfile

# -----------------------------
# Load Models
# -----------------------------
# Text model & vectorizer
nb_model = joblib.load("notebooks/models/naive_bayes_model.pkl")
tfidf = joblib.load("notebooks/models/tfidf_vectorizer.pkl")

# Audio model
audio_model_path = "models/random_forest_audio_model.pkl"
if os.path.exists(audio_model_path):
    rf_audio = joblib.load(audio_model_path)
else:
    rf_audio = None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🧠 Depression Detection", layout="wide")
st.title("🧠 Multimodal Depression Detection App")
st.write(
    "This app can analyze **Text** 📝 or **Voice (Audio)** 🎙 to detect signs of depression.\n\n"
    "It combines Natural Language Processing (for text) and Audio Signal Processing (for speech)."
)

# Tabs for Text and Audio
tab1, tab2 = st.tabs(["✍️ Text Detection", "🎤 Audio Detection"])

# ---------------------------------------------------
# TAB 1 — TEXT CLASSIFICATION
# ---------------------------------------------------
with tab1:
    st.subheader("📝 Text-based Depression Detection")
    user_input = st.text_area("Type your sentence below:")

    if st.button("Predict (Text)"):
        if user_input.strip() != "":
            input_vec = tfidf.transform([user_input])
            prediction = nb_model.predict(input_vec)[0]

            if prediction == 1:
                st.error("⚠️ The text indicates **Depression**")
            else:
                st.success("✅ The text indicates **Not Depressed**")
        else:
            st.warning("Please enter some text first.")

# ---------------------------------------------------
# TAB 2 — AUDIO CLASSIFICATION
# ---------------------------------------------------
with tab2:
    st.subheader("🎙 Audio-based Depression Detection")

    if rf_audio is None:
        st.warning("⚠️ Audio model not found. Please train and save 'random_forest_audio_model.pkl' first.")
    else:
        st.markdown("### 🔹 Option 1: Upload a `.wav` file")
        uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])

        st.markdown("---")
        st.markdown("### 🔹 Option 2: Record your voice using microphone")
        audio = audiorecorder("🎙 Click to record", "Recording...")

        audio_data = None

        # --- Option 1: Uploaded file ---
        if uploaded_file is not None:
            audio_data = uploaded_file
            st.audio(uploaded_file)
            st.info("Using uploaded audio file for prediction.")

        # --- Option 2: Recorded audio ---
        elif len(audio) > 0:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmpfile.write(audio.tobytes())
                audio_data = tmpfile.name
            st.audio(audio.tobytes())
            st.success("✅ Audio recorded successfully!")

        # --- Prediction Section ---
        if audio_data is not None:
            try:
                # Load and extract MFCCs
                y, sr = librosa.load(audio_data, sr=16000)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)

                # Predict
                prediction_audio = rf_audio.predict(mfcc_mean)[0]

                st.markdown("### 🎯 **Prediction Result:**")
                if prediction_audio == "Depressed":
                    st.error("⚠️ The voice indicates **Depression**")
                else:
                    st.success("✅ The voice indicates **Not Depressed**")

            except Exception as e:
                st.error(f"Error processing audio: {e}")
