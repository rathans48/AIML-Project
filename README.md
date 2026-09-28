# Early Warning System for Hate Speech & Misinformation

**A Human-in-the-Loop (HITL) content moderation application** that classifies toxic commentary and tracks misinformative content across live social platforms, using a zero-shot classifier and a binary toxicity model with a human review layer.

---

## What It Does

1. **Collect** — pulls live text data from YouTube (Google API Client v3), Reddit (PRAW), and news articles (NewsAPI)
2. **Classify** — runs each item through an NLP pipeline built on Hugging Face `Transformers`:
   - A zero-shot classifier (`BART`) to categorize content type (credible news / misinformation / conspiracy theory / opinion) without task-specific training
   - A binary toxicity model (`martin-ha/toxic-comment-model`, labels: non-toxic / toxic) to flag harmful content
3. **Review (Human-in-the-Loop)** — flagged content is surfaced in a Streamlit interface for human review, not auto-actioned
4. **Store & Fine-tune** — human-reviewed decisions are archived in SQLite3. An offline script (`train.py`, toxicity side only) can train a new 6-class toxicity head from human labels; it is not loaded by the app. Content-type (misinformation) classification is zero-shot BART only and is not fine-tuned.

---

## Tech Stack

| Component | Technology |
|---|---|
| NLP Models | Hugging Face `Transformers` (zero-shot BART for content type, binary toxicity classifier: non-toxic / toxic) |
| Interface | Streamlit |
| Data Sources | Google API Client (YouTube v3), PRAW (Reddit), NewsAPI |
| Storage | SQLite3 |
| Fine-tuning | Scikit-Learn (train_test_split in train.py only), Hugging Face `Datasets` |

---

## Repo Structure

```
AIML-Project/
├── data_collection.py   # Pulls live data from YouTube, Reddit, and NewsAPI
├── analysis.py           # NLP pipeline — zero-shot classification + toxicity scoring
├── database.py            # SQLite3 schema and persistence layer for reviewed logs
├── train.py                 # Offline fine-tuning script (toxicity side only): trains a new 6-class head from reviewed data, not loaded by the app
├── server.py                  # Backend service layer
├── app.py                       # Streamlit front-end for human review
└── reviews.db                    # SQLite database of human-reviewed content logs
```

---

## Running Locally

```bash
git clone https://github.com/rathans48/AIML-Project.git
cd AIML-Project
pip install -r requirements.txt   # add a requirements.txt if not already present
Copy config.example.py to config.py and fill in your own API keys before running.
```

Set up API credentials for YouTube Data API v3, Reddit (PRAW), and NewsAPI (see each provider's developer console), then run:

```bash
python data_collection.py   # collect fresh data
streamlit run app.py         # launch the review interface
```

---

## Why Human-in-the-Loop

Fully automated moderation systems risk over- or under-flagging content without context. This project keeps a human reviewer in the loop for every flagged item — the model surfaces candidates, a person makes the final call, and that decision feeds back into fine-tuning. This keeps the system accountable while still using automation to handle the volume of incoming content.

---

## Status

Independent AI/ML project — built and iterated on individually (Mar 2026).
