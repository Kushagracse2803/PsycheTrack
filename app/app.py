import streamlit as st
import joblib
import librosa
import numpy as np
import os
import tempfile
import cv2
from PIL import Image
import tensorflow as tf

# -----------------------------
# 1. Dynamic Path Setup
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models')

# -----------------------------
# 2. Load Models
# -----------------------------
nb_model_path = os.path.join(models_dir, "naive_bayes_model.pkl")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
audio_model_path = os.path.join(models_dir, "random_forest_audio_model.pkl")
face_model_path = os.path.join(models_dir, "my_model.keras")

# Load Text Models
try:
    nb_model = joblib.load(nb_model_path)
    tfidf = joblib.load(vectorizer_path)
except FileNotFoundError as e:
    st.error(f"Critical Error: Text model file not found at {e.filename}. Please check your folders.")
    st.stop()

# Load Audio Model
if os.path.exists(audio_model_path):
    rf_audio = joblib.load(audio_model_path)
else:
    rf_audio = None

# Load Face Model
if os.path.exists(face_model_path):
    try:
        face_model = tf.keras.models.load_model(face_model_path)
        print("Face model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading face model: {e}")
        face_model = None
else:
    face_model = None

# Emotion Labels (Standard for FER-2013)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="🧠 Depression Detection", layout="wide")
st.title("🧠 Multimodal Depression Detection App")
st.write(
    "This app can analyze **Text** 📝, **Voice (Audio)** 🎙, and **Facial Expressions** 📸 "
    "to detect signs of depression."
)

# Tabs
tab1, tab2, tab3 = st.tabs(["✍️ Text Detection", "🎤 Audio Detection", "📸 Face Detection"])

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
# TAB 2 — AUDIO CLASSIFICATION (FIXED)
# ---------------------------------------------------
with tab2:
    st.subheader("🎙 Audio-based Depression Detection")

    if rf_audio is None:
        st.warning(f"⚠️ Audio model not found at: {audio_model_path}")
    else:
        # Use Native Streamlit Audio Input (Stable)
        audio_buffer = st.audio_input("Record your voice")
        uploaded_file = st.file_uploader("Or upload a .wav file", type=["wav"])

        # Select source
        audio_source = audio_buffer if audio_buffer else uploaded_file

        if audio_source:
            st.audio(audio_source)
            
            if st.button("Analyze Audio"):
                try:
                    # Write to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                        tmpfile.write(audio_source.getvalue())
                        tmp_path = tmpfile.name
                    
                    # Preprocess
                    y, sr = librosa.load(tmp_path, sr=16000)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                    mfcc_mean = np.mean(mfcc, axis=1).reshape(1, -1)
                    
                    # Predict
                    prediction_audio = rf_audio.predict(mfcc_mean)[0]
                    
                    st.markdown("### 🎯 **Prediction Result:**")
                    if str(prediction_audio) in ["Depressed", "1"]:
                        st.error("⚠️ The voice indicates **Depression**")
                    else:
                        st.success("✅ The voice indicates **Not Depressed**")
                        
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error processing audio: {e}")

# ---------------------------------------------------
# TAB 3 — FACE DETECTION
# ---------------------------------------------------
with tab3:
    st.subheader("📸 Facial Expression Analysis")
    st.write("Take a photo to analyze your emotional state.")

    if face_model is None:
        st.warning(f"⚠️ Face model not found at: {face_model_path}. Please run your notebook to generate 'my_model.keras'.")
    else:
        img_file = st.camera_input("Take a snapshot")

        if img_file is not None:
            # 1. Load Image
            image = Image.open(img_file)
            img_array = np.array(image.convert('RGB'))
            
            # 2. Convert to Grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # 3. Detect Face
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Crop and Resize
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                    
                    # Normalize and Reshape
                    roi = roi_gray.astype('float') / 255.0
                    roi = np.expand_dims(roi, axis=0)
                    roi = np.expand_dims(roi, axis=-1)
                    
                    # Predict
                    prediction = face_model.predict(roi)[0]
                    max_index = int(np.argmax(prediction))
                    predicted_emotion = EMOTIONS[max_index]
                    confidence = prediction[max_index]
                    
                    # Display
                    st.image(img_array, caption="Processed Image", width=350)
                    st.markdown(f"### 🎯 **Detected Emotion:** `{predicted_emotion}`")
                    st.progress(float(confidence))
                    
                    # Interpretation
                    if predicted_emotion in ['Sad', 'Fear', 'Neutral']:
                        st.warning(f"⚠️ Detected '{predicted_emotion}'. Persistent low affect can be a sign of depression.")
                    elif predicted_emotion == 'Happy':
                        st.success(f"✅ Detected '{predicted_emotion}'. Looks positive!")
                    else:
                        st.info(f"Detected '{predicted_emotion}'.")
                    
                    break
            else:
                st.warning("⚠️ No face detected. Please ensure your face is clearly visible.")