import streamlit as st
import joblib

# Load model and vectorizer
nb_model = joblib.load("notebooks/models/naive_bayes_model.pkl")
tfidf = joblib.load("notebooks/models/tfidf_vectorizer.pkl")

# Streamlit App
st.title("🧠 Depression Detection App")
st.write("Enter a sentence and the model will predict if it's **Depressed** or **Not Depressed**")

# User input
user_input = st.text_area("Type your sentence here:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Transform input
        input_vec = tfidf.transform([user_input])
        prediction = nb_model.predict(input_vec)[0]

        if prediction == 1:
            st.error("⚠️ The text indicates **Depression**")
        else:
            st.success("✅ The text indicates **Not Depressed**")
    else:
        st.warning("Please enter some text first.")
