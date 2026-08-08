# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:30:05 2025

@author: Rathan S
"""

# analysis.py
import re
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
    """A simple function to clean text."""
    text = re.sub(r'http\S+', '', text)  # Remove links
    text = re.sub(r'<.*?>', '', text)     # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Remove special characters
    return text.lower()


def analyze_harmful_content(text):
    """
    Analyzes text using a local Hugging Face model for toxicity, threat, etc.
    REPLACES the old analyze_toxicity function.
    """
    # Handle empty or very short strings
    if not text or len(text.strip()) < 5:
        return {
            'toxic': 0.0,
            'severe_toxic': 0.0,
            'obscene': 0.0,
            'threat': 0.0,
            'insult': 0.0,
            'identity_hate': 0.0
        }
    
    try:
        # The model returns a list containing a list of dictionaries, like:
        # [[{'label': 'toxic', 'score': 0.001}, {'label': 'severe_toxic', 'score': 0.0001}, ...]]
        results = classifier_toxic(text)
        
        # Process the results into a simple, flat dictionary
        scores = {}
        if results and isinstance(results, list) and len(results) > 0:
            for label_score_dict in results[0]:
                scores[label_score_dict['label']] = label_score_dict['score']
        
        return scores
        
    except Exception as e:
        print(f"Hugging Face (Toxic) model error: {e}")
        # Return a default "safe" dictionary if analysis fails
        return {
            'toxic': 0.0,
            'severe_toxic': 0.0,
            'obscene': 0.0,
            'threat': 0.0,
            'insult': 0.0,
            'identity_hate': 0.0
        }


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