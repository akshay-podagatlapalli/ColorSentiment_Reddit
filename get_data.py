# getting all the required modules 
import re
import json
import praw
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# API CALL stuff
reddit = praw.Reddit(
    client_id="qJsvLNd6Goxgqz4LZ6yEbg",
    client_secret="KIUx33wL7W434pIy4lru06slhDjAEQ",
    user_agent="posts_ds"
)

# Add the subreddit as a user input
subreddit = reddit.subreddit("history")

# Add additional variables with different selections
# like "new_posts", "hot_posts", "controversial_posts" etc... as user inputs
# Add the limit as a user input
new_posts = subreddit.new(limit=10)


# this function is used to fetch the data 
# from the specific subreddit and return them 
# in a dictionary
def get_data(new_posts=new_posts):
    data_dict = {}
    for post in new_posts:
        post_title = post.title
        post.comments.replace_more(limit=None)
        comments = [comment.body for comment in post.comments.list()]
        if post_title in data_dict:
            data_dict[post_title].extend(comments)
        else:
            data_dict[post_title] = comments

    return data_dict

# calling the above function
unclean_data_dict = get_data()

# this function does the preprocessing of the text
def clean_text(text, replacement_text=''):
    # first remove URLS
    url_pattern = re.compile(r'https?://\S+|www\.\S+|gif\S+')
    text_without_urls = url_pattern.sub(replacement_text, text)
    # second remove, reddit specific tokens
    reddit_tokens = re.compile(r'\br/\w+|\bu/\w+')
    text_without_tokens = reddit_tokens.sub(replacement_text, text_without_urls)
    # remove all special characters except emojis
    special_char = re.compile(
        r'[^\w\s.,!?]|(?<![\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U0001F1E6-\U0001F1FF])')
    cleaned_text = special_char.sub(replacement_text, text_without_tokens)
    return cleaned_text


def init_preprocess(data=unclean_data_dict):
    stop_words = set(stopwords.words('english'))
    clean_data_dict = {}

    for title, comments in data.items():
        clean_comments = []
        for comment in comments:
            # Remove URLs, special characters, etc.
            comment = clean_text(comment)
            # Remove all punctuation from the comments
            comment = comment.translate(str.maketrans('', '', string.punctuation))
            # convert all words to lower case
            comment = comment.lower()
            # tokenize all the words
            word_tokens = word_tokenize(comment)
            # add the tokenized words into a list
            filtered_sentence = [w for w in word_tokens if w not in stop_words]
            clean_comments.append(filtered_sentence)
        
        # Assign processed comments to the corresponding title
        clean_data_dict[title] = clean_comments

    return clean_data_dict


# Run the preprocessing and 
# and store all the cleaned data inside 
# a dataframe
processed_data = init_preprocess()


# Create the desired structure
comments_dict = {title: [[ " ".join(comment) ] for comment in comments] 
                 for title, comments in processed_data.items()}

sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="nlptown/bert-base-multilingual-uncased-sentiment")


# Initialize empty list to store the results
results = []

# Iterate through each title and its corresponding comments
for title, comments in comments_dict.items():
    for idx, sub_comment in enumerate(comments, start=1):
        result = sentiment_pipeline(sub_comment[0])  # Use the first item in the list (joined comment)
        sentiment_score = result[0]['score']  # Use 'label' (positive/negative) or 'score' for numeric sentiment

        # Append the data to the results list
        results.append([title, f"comment #{idx}", sentiment_score])

# Create a DataFrame from the results list
df = pd.DataFrame(results, columns=["title", "comment_number", "result"])
df.to_csv('output.csv', index=False)

# sia = SentimentIntensityAnalyzer()



# for title, comment in zip(comments_dict.keys(), comments_dict.values()):
#     for sub_comment in comment:
#         # Analyze sentiment using VADER
#         sentiment = sia.polarity_scores(sub_comment[0])  # We access the first (and only) element in the list
#         print(f"Sentiment for '{title}': {sub_comment[0]} -> {sentiment}")