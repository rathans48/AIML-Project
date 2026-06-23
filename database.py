# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 19:15:55 2025

@author: Rathan S
"""

# database.py
import sqlite3
import pandas as pd

# Define the database name
DB_FILE = "reviews.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    return sqlite3.connect(DB_FILE)

def init_db():
    """Initializes the database and creates the 'reviews' table if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create the table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_text TEXT,
        source TEXT,
        human_label TEXT,
        reviewer_notes TEXT,
        ai_toxic REAL,
        ai_threat REAL,
        ai_insult REAL,
        ai_obscene REAL,
        ai_content_type TEXT
    )
    """)
    
    # Create a unique index to prevent duplicate entries
    # This stops you from adding the exact same comment from the same source twice
    cursor.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_text_source
    ON reviews (original_text, source)
    """)
    
    conn.commit()
    conn.close()

def add_review(review_data):
    """
    Adds a single reviewed item to the database.
    review_data is a dictionary-like object (e.g., a Pandas row).
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        INSERT INTO reviews (
            original_text, source, human_label, reviewer_notes, 
            ai_toxic, ai_threat, ai_insult, ai_obscene, ai_content_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            review_data['Original Text'],
            review_data['source'],
            review_data['Human_Label'],
            review_data['Reviewer_Notes'],
            review_data['Toxic'],
            review_data['Threat'],
            review_data['Insult'],
            review_data['Obscene'],
            review_data['Content Type']
        ))
        conn.commit()
        return True # Indicates success
    except sqlite3.IntegrityError:
        # This happens if the UNIQUE index fails (duplicate)
        print(f"Skipped duplicate review: {review_data['Original Text'][:30]}...")
        return False # Indicates duplicate
    finally:
        conn.close()

def get_all_reviews():
    """Fetches all rows from the reviews table as a Pandas DataFrame."""
    conn = get_db_connection()
    # pd.read_sql_query is the easiest way to get data into a DataFrame
    df = pd.read_sql_query("SELECT * FROM reviews", conn)
    conn.close()
    return df

def delete_review(review_id):
    """Deletes a review from the database by its unique ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM reviews WHERE id = ?", (review_id,))
    conn.commit()
    conn.close()