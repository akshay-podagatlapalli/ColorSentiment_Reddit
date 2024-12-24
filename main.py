import re
import praw
import json
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from praw.models import MoreComments
from transformers import pipeline

# GET THE DATA

# this is the API call stuff
reddit = praw.Reddit(
    client_id = "qJsvLNd6Goxgqz4LZ6yEbg",
    client_secret = "KIUx33wL7W434pIy4lru06slhDjAEQ",
    user_agent = "posts_ds" 
)

# selecting a specific subreddit
subreddit = reddit.subreddit("ufc")

# calling the top 10 reddit posts
new_posts = subreddit.new(limit=10)

def remove_urls_specialchar(text, replacement_text=''):
    url_pattern = re.compile(r'https?://\S+|www\.\S+|gif\S+')
    text_without_urls = url_pattern.sub(replacement_text, text)
    reddit_tokens = re.compile(r'\br/\w+|\bu/\w+')
    text_without_tokens = reddit_tokens.sub(replacement_text, text_without_urls)
    special_char = re.compile(r'[^\w\s.,!?]|(?<![\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U0001F1E6-\U0001F1FF])')
    clean_text = special_char.sub(replacement_text, text_without_tokens)
    return clean_text

# printing the titles for the top 10 posts
def get_data(new_posts = new_posts):
    data_dict = {}
    for post in new_posts:
        # Get the title of the post
        post_title = post.title
    
        # fetch all comments
        post.comments.replace_more(limit=None)
    
        # Create a list of all comment bodies for the post
        comments = [comment.body for comment in post.comments.list()]
    
        # Add the list of comments to the dictionary under the post's title
        if post_title in data_dict:
            data_dict[post_title].extend(comments)  # Append if the title already exists
        else:
            data_dict[post_title] = comments  # Add a new entry
    
    return data_dict

reddit_data_dict = get_data()

def init_preprocess(data = reddit_data_dict):
    stop_words = set(stopwords.words('english'))
    clean_data_dict = {}
    clean_comments = []

    for val in data.values():
        for sub_val in val:
            # removing the urls from the comments
            sub_val = remove_urls_specialchar(sub_val)
            
            # removing all the punctuations from the comments
            sub_val = sub_val.translate(str.maketrans('', '',
                                        string.punctuation))
            sub_val = sub_val.lower()
            word_tokens = word_tokenize(sub_val)
            filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
            filtered_sentence = []
            for w in word_tokens:
                if w not in stop_words:
                    filtered_sentence.append(w)
            clean_comments.append(filtered_sentence)
    
    for title in data.keys():
        clean_data_dict[title] = clean_comments

    return clean_data_dict


# with open('output1.txt', 'w') as output_file:
#     output_file.write(json.dumps(init_preprocess()))

# Initialize the sentiment analysis pipeline with the specific model
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="nlptown/bert-base-multilingual-uncased-sentiment")

# Process the preprocessed data
for f_list in init_preprocess().values():
    for sub_list in f_list:
        # Reconstruct the sentence
        sentence = " ".join(sub_list)
        
        # Perform sentiment analysis
        result = sentiment_pipeline(sentence)
        print(f"Sentence: {sentence}")
        print(f"Sentiment: {result}")

# ANALYZE THE DATA

# VISUALIZE THE DATA