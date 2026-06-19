import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


FINBERT_MODEL_NAME = "ProsusAI/finbert"
MAX_TOKEN_LENGTH = 512
FINBERT_LABELS = ["neutral", "positive", "negative"]

_vader_analyser = SentimentIntensityAnalyzer()
_finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
_finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)


def apply_vader(df):
    """Add a vader_compound column scoring each title's sentiment."""
    df["vader_compound"] = df["title"].apply(
        lambda text: _vader_analyser.polarity_scores(str(text))["compound"]
    )
    return df


def apply_finbert(df):
    """Add a finbert_sentiment column with neutral/positive/negative per title."""
    sentiments = []
    for text in df["title"]:
        tokens = _finbert_tokenizer(
            text, return_tensors = "pt", truncation = True, max_length = MAX_TOKEN_LENGTH
        )
        with torch.no_grad():
            outputs = _finbert_model(**tokens)
            probs = torch.nn.functional.softmax(outputs.logits, dim = 1)
            label_index = torch.argmax(probs).item()
            sentiments.append(FINBERT_LABELS[label_index])
    df["finbert_sentiment"] = sentiments
    return df
