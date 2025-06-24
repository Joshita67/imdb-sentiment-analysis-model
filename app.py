import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="IMDb Sentiment", layout="centered")
st.title("🎬 IMDb Sentiment Checker")

@st.cache_resource
def load_pipeline():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        framework="pt"
    )

nlp = load_pipeline()

review = st.text_area("Enter your movie review:")
if st.button("Analyze") and review.strip():
    result = nlp(review)[0]
    st.success(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
