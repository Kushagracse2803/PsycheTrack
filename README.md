# 🧠 MindScan AI (PsycheTrack)

### **Multimodal Depression Detection System**
*An AI-powered screening tool that analyzes **Text**, **Voice**, and **Facial Expressions** to identify early signs of depression.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 📖 Overview

**MindScan AI** is a "Trimodal" Affective Computing system designed to act as an accessible, privacy-first mental health screener. Unlike traditional tools that rely solely on self-reported questionnaires, MindScan analyzes three distinct biomarkers:

* **Linguistic Cues (Text):** What you say.
* **Acoustic Cues (Audio):** How you say it (prosody/tone).
* **Visual Cues (Face):** Emotional expression (flat affect).

This project was built to bridge the gap between clinical screening and accessible technology, providing users with instant, preliminary insights into their mental well-being.

---

## ✨ Features

### 1. 📝 Text Analysis (NLP)
* **Input:** User types a journal entry or describes their day.
* **Technique:** TF-IDF Vectorization to convert text into numerical format.
* **Model:** **Multinomial Naive Bayes**.
* **Why:** Highly efficient at detecting negative sentiment keywords (e.g., "hopeless", "tired") in sparse text data.

### 2. 🎙️ Audio Analysis (Speech)
* **Input:** User records a 5-second voice note or uploads a `.wav` file.
* **Technique:** Extracts **MFCCs** (Mel-Frequency Cepstral Coefficients) using `Librosa` to capture vocal timbre.
* **Model:** **Random Forest Classifier**.
* **Why:** Effective at detecting "flat affect" (monotone, low-energy speech) associated with depression.

### 3. 📸 Facial Expression Analysis (Vision)
* **Input:** Live webcam feed or image snapshot.
* **Technique:** Face detection via **Haar Cascades** (OpenCV) followed by deep learning analysis.
* **Model:** **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.
* **Why:** Detects persistent "Sad" or "Neutral" expressions which may indicate emotional blunting.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python)
* **Machine Learning:** Scikit-Learn (Naive Bayes, Random Forest)
* **Deep Learning:** TensorFlow / Keras (CNN)
* **Computer Vision:** OpenCV (Haar Cascades)
* **Audio Processing:** Librosa
* **Data Handling:** Pandas, NumPy

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/Kushagracse2803/PsycheTrack.git](https://github.com/Kushagracse2803/PsycheTrack.git)
cd PsycheTrack
 Create a Virtual Environment (Recommended)
python -m venv venv
.\venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

PsycheTrack/
│
├── app/
│   └── app.py                # Main Streamlit Application
│
├── models/                   # Saved ML Models
│   ├── naive_bayes_model.pkl         # NLP Model
│   ├── tfidf_vectorizer.pkl          # Text Vectorizer
│   ├── random_forest_audio_model.pkl # Audio Model
│   └── my_model.keras                # Face Expression CNN
│
├── notebooks/                # Jupyter Notebooks for Training
│   ├── 01_Text_Training.ipynb
│   ├── 02_Audio_Extraction.ipynb
│   └── 03_Face_CNN.ipynb
│
├── data/                     # Raw Data (Not uploaded to GitHub)
│   ├── raw/
│   └── processed/
│
├── requirements.txt          # Python Dependencies
├── .gitignore                # Files to ignore (venv, large models)
└── README.md                 # Project Documentation

⚠️ Disclaimer
This project is a prototype for screening purposes only. It is not a diagnostic tool. If you or someone you know is struggling, please consult a medical professional.

👤 Author
Kushagra Tiwari
