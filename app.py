from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from transformers import pipeline
from textblob import TextBlob
from math import ceil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity
    if sentiment_score > 0.1:
        sentiment = "Positive"
    elif sentiment_score < -0.1:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, sentiment_score

from transformers import pipeline
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import nltk

nltk.download('punkt')
nltk.download('stopwords')

def generate_summary(text):
    # Tokenize sentences
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words("english"))
    
    # Preprocess sentences to remove stopwords
    preprocessed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        preprocessed_sentences.append(filtered_words)

    # Calculate word frequencies
    flat_preprocessed_words = [word for sentence in preprocessed_sentences for word in sentence]
    word_freq = FreqDist(flat_preprocessed_words)

    # Score sentences based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(preprocessed_sentences):
        for word in sentence:
            if word in word_freq:
                if i in sentence_scores:
                    sentence_scores[i] += word_freq[word]
                else:
                    sentence_scores[i] = word_freq[word]

    # Extractive summary: Get top 50% of the sentences based on frequency
    summary_sentences = []
    if sentence_scores:
        sorted_scores = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        top_sentences_count = max(1, len(sentences) // 2)
        top_sentences_indices = [index for index, _ in sorted_scores[:top_sentences_count]]
        top_sentences_indices.sort()
        for index in top_sentences_indices:
            summary_sentences.append(sentences[index])

    # Join sentences to form the extractive summary
    extractive_summary = ' '.join(summary_sentences)

    # If the text is short, return the extractive summary
    if len(sentences) <= 3:
        return extractive_summary

    # Use a pretrained summarizer for abstractive summary
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Dynamically adjust length parameters based on input length
    avg_word_count = len(extractive_summary.split())
    max_summary_length = min(200, avg_word_count + 50)  # Max length for long texts
    min_summary_length = max(50, avg_word_count // 2)  # Min length for summarization

    abstractive_summary = summarizer(
        extractive_summary,
        max_length=max_summary_length,
        min_length=min_summary_length,
        do_sample=False
    )

    # If the abstractive summary is too short, return extractive summary instead
    if len(abstractive_summary[0]['summary_text'].split()) < min_summary_length:
        return extractive_summary

    return abstractive_summary[0]['summary_text']


@app.route('/')
def home():
    return render_template('index.html', summary=None, dark_mode=False)

@app.route('/about')
def about():
    return render_template('about.html',dark_mode=False)

@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    text = request.form['input_text']
    summary = generate_summary(text)
    original_sentiment, original_sentiment_score = analyze_sentiment(text)
    original_sentiment_score = round(original_sentiment_score, 2)
    summary_sentiment, summary_sentiment_score = analyze_sentiment(summary)
    summary_sentiment_score = round(summary_sentiment_score, 2)

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
    original_sentiment_score = round(original_sentiment_score, 2)
    summary_sentiment, summary_sentiment_score = analyze_sentiment(summary)
    summary_sentiment_score = round(summary_sentiment_score, 2)

    return render_template('index.html', summary=summary, original_text=text,
                           original_sentiment=original_sentiment, original_sentiment_score=original_sentiment_score,
                           summary_sentiment=summary_sentiment, summary_sentiment_score=summary_sentiment_score)

if __name__ == '__main__':
    app.run(debug=True)
