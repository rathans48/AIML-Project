# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 20:09:08 2025

@author: Rathan S
"""

# train.py
import sqlite3
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# --- 1. CONFIGURATION ---
DB_FILE = "reviews.db"
# The pre-trained model you are starting from
BASE_MODEL = "martin-ha/toxic-comment-model"
# The name of the new, fine-tuned model you will create
NEW_MODEL_NAME = "./my_finetuned_toxic_model"

# Define the labels for THIS model. We will ignore 'misinformation', etc.
TOXICITY_LABELS = [
    "Safe", 
    "Toxic", 
    "Severe Toxic", 
    "Threat", 
    "Insult", 
    "Obscene"
]

# Create the label-to-ID mappings that the model needs
label2id = {label: i for i, label in enumerate(TOXICITY_LABELS)}
id2label = {i: label for i, label in enumerate(TOXICITY_LABELS)}

# --- 2. LOAD AND PREPARE DATA ---
def load_data_from_db():
    print(f"Loading reviewed data from {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    
    # Create a SQL-friendly list of labels to fetch
    # This looks like "('Safe', 'Toxic', 'Insult', ...)"
    labels_to_fetch = tuple(TOXICITY_LABELS)
    
    # SQL query to get only the reviews relevant to THIS model
    query = f"""
    SELECT original_text, human_label 
    FROM reviews 
    WHERE human_label IN {labels_to_fetch}
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Rename columns to what the model expects
    df = df.rename(columns={"original_text": "text", "human_label": "label_name"})
    
    # Map the text labels (e.g., "Safe") to integer IDs (e.g., 0)
    df["label"] = df["label_name"].map(label2id)
    
    # Drop any rows where mapping might have failed
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    
    print(f"Loaded {len(df)} relevant reviews.")
    return df

# --- 3. TOKENIZATION ---
def tokenize_data(batch):
    # Tokenizer will turn text into numbers (token IDs)
    return tokenizer(batch["text"], padding="max_length", truncation=True)

# --- 4. MAIN TRAINING FUNCTION ---
def main():
    # Check for GPU
    if torch.cuda.is_available():
        print("GPU is available. Training will be fast.")
    else:
        print("WARNING: GPU not available. Training will be VERY slow on CPU.")

    # Load the data
    df = load_data_from_db()
    
    if len(df) < 50:
        print("Not enough data to train. Please review at least 50 comments.")
        return

    # Split data: 80% for training, 20% for validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Convert Pandas DataFrames to Hugging Face Dataset objects
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Load the tokenizer for the base model
    print(f"Loading tokenizer for {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Tokenize the datasets
    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_data, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_data, batched=True)

    # Load the base model
    print(f"Loading base model {BASE_MODEL} for fine-tuning...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=len(label2id), # Tell the model how many labels we have
        id2label=id2label,       # Pass our label mappings
        label2id=label2id
    )

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=NEW_MODEL_NAME,      # Where to save the new model
        num_train_epochs=3,             # 3 epochs is a good starting point
        per_device_train_batch_size=8,  # Batch size (lower if you run out of memory)
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",           # Folder to store logs
        logging_steps=10,
        evaluation_strategy="epoch",    # Evaluate at the end of each epoch
        save_strategy="epoch",          # Save at the end of each epoch
        load_best_model_at_end=True,    # Load the best version after training
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
    )

    # Start the training!
    print("--- Starting Fine-Tuning ---")
    trainer.train()
    print("--- Fine-Tuning Complete ---")

    # Save the final, best model
    trainer.save_model(NEW_MODEL_NAME)
    tokenizer.save_pretrained(NEW_MODEL_NAME) # Save the tokenizer too
    print(f"Successfully saved fine-tuned model to {NEW_MODEL_NAME}")

# This makes the script runnable
if __name__ == "__main__":
    main()