# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 14:26:17 2025

@author: Rathan S
"""

# data_collection.py
import pandas as pd
import praw
from googleapiclient.discovery import build
from newsapi import NewsApiClient
from config import (
    #YOUTUBE_API_KEY, 
    NEWS_API_KEY, REDDIT_CLIENT_ID, 
    REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT, 
    REDDIT_USERNAME, REDDIT_PASSWORD
)

def get_youtube_comments(api_key, video_id, max_results=50):
    """Fetches comments from a single YouTube video."""
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    
    # Get comments from the video
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=max_results
    )
    response = request.execute()
    
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'source': 'YouTube',
            'text': comment['textDisplay'],
            'author': comment['authorDisplayName']
        })
        
    return pd.DataFrame(comments)

def get_news_articles(query, max_results=50):
    """Fetches news articles from NewsAPI based on a query."""
    try:
        # Initialize the client
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Fetch articles. 'q' is your search query.
        # 'language='en'' ensures we only get English text
        # 'page_size' controls how many results (max 100 on free plan)
        articles_response = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='relevancy',
            page_size=max_results
        )
        
        articles = []
        if articles_response['status'] == 'ok':
            for article in articles_response['articles']:
                # We use 'description' as the 'text' for analysis
                # as 'content' is often truncated.
                articles.append({
                    'source': article['source']['name'],
                    'text': article['description'] or "", # Use description, ensure no None
                    'author': article['author'] or 'N/A'
                })
        
        return pd.DataFrame(articles)
        
    except Exception as e:
        print(f"NewsAPI error: {e}")
        # Return an empty DataFrame on failure
        return pd.DataFrame(columns=['source', 'text', 'author'])

def get_reddit_posts(subreddit_name, limit=25):
    """Fetches 'hot' posts from a specified subreddit."""
    try:
        reddit = praw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
            username=REDDIT_USERNAME,
            password=REDDIT_PASSWORD,
        )
        
        # Get the subreddit
        subreddit = reddit.subreddit(subreddit_name)
        
        posts_data = []
        # Get the 'hot' posts from that subreddit
        for post in subreddit.hot(limit=limit):
            # We will analyze the post title and its selftext (if it's not a link post)
            text_to_analyze = post.title + " " + post.selftext
            
            posts_data.append({
                'source': f'r/{subreddit_name}',
                'text': text_to_analyze,
                'author': post.author.name if post.author else 'N/A'
            })
            
        return pd.DataFrame(posts_data)

    except Exception as e:
        print(f"PRAW (Reddit) error: {e}")
        return pd.DataFrame(columns=['source', 'text', 'author'])