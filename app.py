import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="IMDb Sentiment Checker", layout="centered")
st.title("🎬 IMDb Sentiment Checker")

@st.cache_resource
def load_pipeline():
    # Use default lightweight sentiment analysis model
    return pipeline("sentiment-analysis")

nlp = load_pipeline()

review = st.text_area("Enter your movie review:")
if st.button("Analyze") and review.strip():
    result = nlp(review)[0]
    label = result['label']
    score = result['score']
    st.markdown(f"### Sentiment: `{label}`")
    st.markdown(f"**Confidence:** {score:.2f}")
