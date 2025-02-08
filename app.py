from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import nltk
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from transformers import pipeline
from textblob import TextBlob

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load summarization model once at startup (CPU mode)
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

def analyze_sentiment(text):
    """Analyze sentiment polarity of text."""
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0.1:
        sentiment = "Positive"
    elif sentiment_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, round(sentiment_score, 2)

def generate_summary(text):
    """Generate extractive + abstractive summary."""
    # Tokenize sentences
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    
    # Preprocess sentences (remove stopwords)
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        preprocessed_sentences.append(filtered_words)

    # Compute word frequencies
    flat_words = [word for sentence in preprocessed_sentences for word in sentence]
    word_freq = FreqDist(flat_words)

    # Score sentences based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(preprocessed_sentences):
        for word in sentence:
            if word in word_freq:
                sentence_scores[i] = sentence_scores.get(i, 0) + word_freq[word]

    # Extractive summary (top 50% sentences)
    if sentence_scores:
        top_sentences_count = max(1, len(sentences) // 2)
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_sentences_count]
        extractive_summary = ' '.join([sentences[i] for i in sorted(top_sentences)])
    else:
        extractive_summary = text  # If no meaningful summary, return original text

    # Short texts → return extractive summary
    if len(sentences) <= 3:
        return extractive_summary

    # Abstractive summarization with dynamic length
    avg_length = len(extractive_summary.split())
    max_length = min(150, avg_length + 30)
    min_length = max(30, avg_length // 3)

    abstractive_summary = summarizer(extractive_summary, max_length=max_length, min_length=min_length, do_sample=False)

    return abstractive_summary[0]['summary_text'] if len(abstractive_summary[0]['summary_text'].split()) > min_length else extractive_summary

@app.route('/')
def home():
    return render_template('index.html', summary=None, dark_mode=False)

@app.route('/about')
def about():
    return render_template('about.html', dark_mode=False)

@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    text = request.form['input_text']
    summary = generate_summary(text)
    original_sentiment, original_sentiment_score = analyze_sentiment(text)
    summary_sentiment, summary_sentiment_score = analyze_sentiment(summary)

    return render_template('index.html', summary=summary, original_text=text,
                           original_sentiment=original_sentiment, original_sentiment_score=original_sentiment_score,
                           summary_sentiment=summary_sentiment, summary_sentiment_score=summary_sentiment_score)

@app.route('/summarize_file', methods=['POST'])
def summarize_file():
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    summary = generate_summary(text)
    original_sentiment, original_sentiment_score = analyze_sentiment(text)
    summary_sentiment, summary_sentiment_score = analyze_sentiment(summary)

    return render_template('index.html', summary=summary, original_text=text,
                           original_sentiment=original_sentiment, original_sentiment_score=original_sentiment_score,
                           summary_sentiment=summary_sentiment, summary_sentiment_score=summary_sentiment_score)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
