import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import pandas as pd
from utils import extract_text_from_pdf, summarize_text, extract_keywords, analyze_citations

st.title("Automated Research Paper Analyzer and Summarizer")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)

    st.write("Text extracted from PDF:")
    st.text_area("Extracted Text", text, height=300)

    with st.spinner("Summarizing text..."):
        summary = summarize_text(text)

    st.write("Summary:")
    st.text_area("Summary", summary, height=200)

    with st.spinner("Extracting keywords..."):
        keywords = extract_keywords(text)

    st.write("Keywords:")
    st.write(", ".join(keywords))

    with st.spinner("Analyzing citations..."):
        citations = analyze_citations(text)

    st.write("Citations:")
    st.write(citations)

    # Extra features
    st.write("Extra Features:")
    # Add your extra features here
    # For example, sentiment analysis
    nlp = pipeline("sentiment-analysis")
    sentiment = nlp(summary)
    st.write("Sentiment:")
    st.write(sentiment)
