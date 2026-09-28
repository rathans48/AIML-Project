# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:30:05 2025

@author: Rathan S
"""

# analysis.py
import re
import pandas as pd
from transformers import pipeline

# --- Load Models ---
# We load them here (globally) so they only load once when the app starts, not on every run.

# 1. Model for Misinformation (as before)
classifier_misinfo = pipeline("zero-shot-classification", 
                                model="facebook/bart-large-mnli")

# 2. NEW: Model for Toxicity
# This model is trained to detect multiple types of harmful content.
# 'return_all_scores=True' makes it return scores for all labels (toxic, threat, etc.)
classifier_toxic = pipeline("text-classification", 
                              model="martin-ha/toxic-comment-model", 
                              top_k=None)

# --------------------

def clean_text(text):
    """Clean text: strip URLs and HTML tags only; preserve casing/punctuation."""
    if text is None:
        return ""
    # NaN is a float; this guard prevents TypeError in re.sub()
    if isinstance(text, float) and pd.isna(text):
        return ""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>', '', text)     # Remove HTML tags
    return text


def analyze_harmful_content(text):
    """
    Analyzes text with the binary martin-ha/toxic-comment-model
    (labels: 'non-toxic' / 'toxic').
    Returns {'toxic': <score of the 'toxic' label>}.
    """
    # Handle empty or very short strings
    if not text or len(text.strip()) < 5:
        return {'toxic': 0.0}
    
    try:
        # The model has exactly 2 labels ('non-toxic', 'toxic') and
        # returns them as a list with one dict per label, like:
        # [[{'label': 'non-toxic', 'score': 0.99}, {'label': 'toxic', 'score': 0.01}]]
        results = classifier_toxic(text)
        
        # Look up the 'toxic' label's score by name
        if results and isinstance(results, list) and len(results) > 0:
            for label_score_dict in results[0]:
                if label_score_dict['label'] == 'toxic':
                    return {'toxic': float(label_score_dict['score'])}
        
        return {'toxic': 0.0}
        
    except Exception as e:
        print(f"Hugging Face (Toxic) model error: {e}")
        # Return a default "safe" dictionary if analysis fails
        return {'toxic': 0.0}


def detect_misinformation(text):
    """Uses a Hugging Face model to check for misinformation-like content."""
    # Define candidate labels for the model to check against
    candidate_labels = ['credible news', 'misinformation', 'conspiracy theory', 'opinion']
    
    if not text or len(text.split()) < 3:
        return {'top_label': 'N/A', 'top_score': 0}
        
    try:
        result = classifier_misinfo(text, candidate_labels)
        # Return the label with the highest score
        return {
            'top_label': result['labels'][0],
            'top_score': result['scores'][0]
        }
    except Exception as e:
        print(f"Hugging Face (Misinfo) model error: {e}")
        return {'top_label': 'Error', 'top_score': 0}