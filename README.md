# BrevityAI

# Text Summarizer with Sentiment Analysis Web App

The Text Summarizer is a web application that leverages natural language processing techniques to create concise and coherent summaries from longer pieces of text. It simplifies the process of information extraction, making it easier for users to digest and comprehend voluminous content such as articles, reports, and documents.Additionally, it includes sentiment analysis for the original text.

## Features

- User-friendly web interface for text summarization.
- Automatic summarization of input text.
- Option to upload text files containing articles.
- Sentiment analysis for original and summary text.

## Technologies Used

- Python for backend logic.
- Flask web framework for building the web application.
- NLTK (Natural Language Toolkit) and BART for text processing, extractive and abstractive summarization.
- HTML,CSS for styling the user interface.
- TextBlob for sentiment analysis.

## Usage

1. **Input Text**: Enter the text you want to summarize into the text area on the main page.

2. **Summarize**: Click the "Summarize" button to generate a summary of the input text.

3. **Upload Text File**: Optionally, click the "Upload File" button to fetch content from text files for summarization.

4. **View Summary**: The generated summary will be displayed on the same page along with the original text.

## Installation

To run this application locally, follow these steps:

1. Clone this repository to your local machine.

```bash
git clone (https://github.com/Hahasha4/BrevityAI)
cd BrevityAI
```
2. Install the required Python packages using pip. It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

3. Start the Flask application.

```bash
python app.py
```
## Limitations
- The quality of the summary depends on the complexity and quality of the input text. It may not always capture nuances and context effectively.
- The summarization model used here is based on word frequency and may not handle very technical or domain-specific content optimally.
- Large volumes of text may result in less coherent summaries.
---

