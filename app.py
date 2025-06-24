import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="IMDb Sentiment Checker", page_icon="🎬", layout="centered")
st.title("🎬 IMDb Sentiment Checker")

# Load sentiment analysis pipeline
@st.cache_resource
def load_pipeline():
    return pipeline("sentiment-analysis")

sentiment_pipe = load_pipeline()

# User input
user_input = st.text_area("Enter your movie review:")

if st.button("Analyze"):
    if user_input.strip() != "":
        result = sentiment_pipe(user_input)[0]
        label = result['label']
        score = result['score']
        st.success(f"**Sentiment:** {label} \n\n**Confidence:** {score:.2f}")
    else:
        st.warning("Please enter a review before analyzing.")
