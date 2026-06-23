# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:30:32 2025

@author: Rathan S
"""

# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import database

# Import your custom modules
from config import (YOUTUBE_API_KEY) #NEWS_API_KEY, REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, REDDIT_USERNAME, REDDIT_PASSWORD)
from data_collection import get_youtube_comments, get_news_articles, get_reddit_posts
from analysis import clean_text, analyze_harmful_content, detect_misinformation

database.init_db()

st.set_page_config(layout="wide")
st.title("Harmful Content Detection Dashboard")
tab1, tab2 = st.tabs(["Analyze New Data", "View Reviewed Database"])

with tab1:
    # --- 1. Initialize Session State ---
    # This is crucial to keep our data persistent across button clicks
    if 'results_df' not in st.session_state:
        st.session_state['results_df'] = pd.DataFrame()
    
    # --- Sidebar for User Input ---
    st.sidebar.header("Data Collection Controls")
    source_option = st.sidebar.selectbox("Choose a data source", ["YouTube", "Reddit", "NewsAPI"])
    
    # --- NEW: Use an if/elif/else block for the sidebar ---
    if source_option == "YouTube":
        video_id_input = st.sidebar.text_input("Enter YouTube Video ID", "dQw4w9WgXcQ")
    
    elif source_option == "NewsAPI":
        # Add a text box to search for news
        query_input = st.sidebar.text_input("Enter search query (e.g., election)", "election")
    
    else: # Reddit
        # Add a text box to get a subreddit name
        subreddit_input = st.sidebar.text_input("Enter subreddit (e.g., news, politics)", "news")
    
    fetch_button = st.sidebar.button("Fetch and Analyze Data")
        
    # --- Main App Logic ---
    if fetch_button:
        # 2. Data Collection
        with st.spinner("Collecting data..."):
    
            # --- UPDATED LOGIC ---
            if source_option == "YouTube":
                raw_data_df = get_youtube_comments(YOUTUBE_API_KEY, video_id_input)
    
            elif source_option == "NewsAPI":
                raw_data_df = get_news_articles(query_input)
    
            else: # Reddit
                raw_data_df = get_reddit_posts(subreddit_input)
            # --- END OF UPDATED LOGIC ---
    
        if raw_data_df.empty:
            st.warning(f"Failed to collect data from {source_option}. The API might be temporarily down or returned no results. Please try again later.")
            st.session_state['results_df'] = pd.DataFrame() # Clear old results
            st.stop()
        
        st.success(f"Collected {len(raw_data_df)} items from {source_option}!")
    
        # 3. Preprocessing and AI Analysis
        st.info("Analyzing content with local AI models. This may take a moment...")
        progress_bar = st.progress(0)
        results = []
        total_items = len(raw_data_df)
    
        for index, row in raw_data_df.iterrows():
            text_to_analyze = row['text']
            cleaned_text = clean_text(text_to_analyze)
            
            harmful_scores = analyze_harmful_content(cleaned_text)
            misinfo_scores = detect_misinformation(cleaned_text)
            
            results.append({
                'source': row['source'],
                'Original Text': text_to_analyze,
                'Toxic': harmful_scores.get('toxic', 0),
                'Threat': harmful_scores.get('threat', 0),
                'Insult': harmful_scores.get('insult', 0),
                'Obscene': harmful_scores.get('obscene', 0),
                'Content Type': misinfo_scores['top_label'],
                'Confidence': misinfo_scores['top_score'],
                
                # --- NEW COLUMNS FOR HUMAN-IN-THE-LOOP ---
                'Human_Label': 'Not Reviewed', # Default value
                'Reviewer_Notes': ''          # Blank notes field
            })
            
            progress_bar.progress((index + 1) / total_items)
        
        progress_bar.empty()
        st.success("Analysis complete!")
        results_df = pd.DataFrame(results)
        # --- 4. Store results in Session State ---
        st.session_state['results_df'] = pd.DataFrame(results)
    
    # --- Dashboard Display ---
    # This block now runs *outside* the 'if fetch_button' block.
    # It will run as long as there is data in the session state.
    
    if not st.session_state['results_df'].empty:
        results_df = st.session_state['results_df']
        
        st.header("AI Analysis Visualizations")
        
        # Charts & Graphs
        col1, col2 = st.columns(2)
        with col1:
            fig_toxicity = px.histogram(results_df, x="Toxic", title="Distribution of AI Toxicity Scores")
            st.plotly_chart(fig_toxicity, use_container_width=True)
        with col2:
            fig_content_type = px.pie(results_df, names='Content Type', title='Distribution of AI Content Types')
            st.plotly_chart(fig_content_type, use_container_width=True)
    
        # --- 5. NEW: Human-in-the-Loop Review Section ---
        st.header("Human Reviewer Tools")
        st.info("Use the table below to review and correct the AI's labels. Your changes are saved when you click 'Save Reviews'.")
    
        # Configure the editable columns
        column_config = {
            "Human_Label": st.column_config.SelectboxColumn(
                "Human Label",
                help="Correct the AI's label here.",
                options=[
                    "Not Reviewed",
                    "Safe",
                    "Toxic",
                    "Severe Toxic",
                    "Misinformation",
                    "Insult",
                    "Threat"
                ],
                required=True
            ),
            "Reviewer_Notes": st.column_config.TextColumn(
                "Reviewer Notes",
                help="Add any notes here."
            ),
            
            # --- CORRECTED SECTION ---
            
            # Make the AI columns read-only
            "Original Text": st.column_config.TextColumn("Original Text", disabled=True, width="large"),
            
            # ProgressColumn is read-only by default (no 'disabled' argument needed)
            "Toxic": st.column_config.ProgressColumn(
                "AI Toxic", format="%.2f", min_value=0, max_value=1
            ),
            "Threat": st.column_config.ProgressColumn(
                "AI Threat", format="%.2f", min_value=0, max_value=1
            ),
            "Insult": st.column_config.ProgressColumn(
                "AI Insult", format="%.2f", min_value=0, max_value=1
            ),
            # Added the missing columns
            "Obscene": st.column_config.ProgressColumn(
                "AI Obscene", format="%.2f", min_value=0, max_value=1
            ),
            "Confidence": st.column_config.ProgressColumn(
                "AI Confidence", format="%.2f", min_value=0, max_value=1
            ),
            
            "Content Type": st.column_config.TextColumn("AI Content Type", disabled=True),
        }
    
        # Use st.data_editor to create the interactive table
        # We assign it to a new variable `edited_df`
        edited_df = st.data_editor(
            results_df,
            column_config=column_config,
            num_rows="dynamic", # Lets you add/delete rows
            key="data_editor",
            height=400
        )
    
        # A button to "commit" the changes from the editor back to the session state
        if st.button("Save Reviews"):
            st.session_state['results_df'] = edited_df
            st.success("Your reviews have been saved!")
        
        if st.button("Commit Reviews to Database"):
                # Filter for rows that have been reviewed
                reviewed_items = edited_df[edited_df['Human_Label'] != 'Not Reviewed']
                
                if reviewed_items.empty:
                    st.warning("No items have been reviewed! (Please change 'Human_Label' from 'Not Reviewed')")
                else:
                    committed_count = 0
                    duplicate_count = 0
                    
                    with st.spinner("Committing reviews to database..."):
                        # Go row-by-row and add to DB
                        for index, row in reviewed_items.iterrows():
                            if database.add_review(row):
                                committed_count += 1
                            else:
                                duplicate_count += 1
                    
                    st.success(f"Successfully committed {committed_count} new reviews to the database!")
                    if duplicate_count > 0:
                        st.info(f"Skipped {duplicate_count} reviews that were already in the database.")
                    
                    # Clear the committed items from the session state (inbox workflow)
                    st.session_state['results_df'] = edited_df[edited_df['Human_Label'] == 'Not Reviewed']
                    st.rerun()
    
        # 6. Reporting
        st.header("Reporting")
        
        @st.cache_data
        def convert_df_to_csv(df):
            # Convert DataFrame to CSV, ensuring index is not saved
            return df.to_csv(index=False).encode('utf-8')
    
        # IMPORTANT: The download button now uses the data from session state,
        # which includes any saved edits.
        csv_data = convert_df_to_csv(st.session_state['results_df'])
        
        st.download_button(
            label="Download FULL Reviewed Data as CSV",
            data=csv_data,
            file_name='reviewed_analysis_report.csv',
            mime='text/csv',
        )
    
    with tab2:
        st.header("Reviewed Content Database")
        st.info("Here you can view and delete all reviews that have been permanently saved.")
    
        # Fetch all data from the database
        reviews_df = database.get_all_reviews()
    
        if reviews_df.empty:
            st.warning("The database is currently empty. Please review and commit items in the 'Analyze New Data' tab.")
        else:
            # Display the full database
            st.dataframe(reviews_df, use_container_width=True)
        
        # --- Add a simple 'Delete' feature ---
        st.subheader("Delete a Review")
        
        # Get a list of all IDs
        all_ids = reviews_df['id'].tolist()
        
        # Create a selectbox to choose an ID
        id_to_delete = st.selectbox("Select Review ID to delete:", options=all_ids)
        
        if st.button("Delete Selected Review", type="primary"):
            database.delete_review(id_to_delete)
            st.success(f"Deleted review ID: {id_to_delete}")
            st.rerun() # Rerun the app to refresh the dataframe