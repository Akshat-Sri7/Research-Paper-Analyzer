import fitz  # PyMuPDF
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

summarizer = pipeline("summarization")
keyword_extractor = TfidfVectorizer(stop_words=list(stop_words))


def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def summarize_text(text):
    max_chunk_size = 512
    text_chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    summaries = [summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text'] for chunk in text_chunks]
    return " ".join(summaries)


def extract_keywords(text):
    text = re.sub(r'\W+', ' ', text)
    tfidf_matrix = keyword_extractor.fit_transform([text])
    feature_names = keyword_extractor.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    keywords = [feature_names[i] for i in denselist[0].argsort()[-10:]]
    return keywords


def analyze_citations(text):
    # Placeholder function for citation analysis
    citations = re.findall(r'\[(\d+)\]', text)
    return citations
