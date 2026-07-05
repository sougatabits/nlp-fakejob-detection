import streamlit as st
import joblib
import re

st.title("🧠 Fake Job Detection App")

# Dummy simple rule-based demo (replace with model later)
def predict(text):
    text = text.lower()

    if " job guarantee" in text or "whatsapp" in text:
        return "FAKE JOB 🚨"
    return "REAL JOB ✅"

user_input = st.text_area("Enter Job Description")

if st.button("Predict"):
    result = predict(user_input)
    st.write(result)