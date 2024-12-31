from flask import Flask, request, render_template
import praw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from transformers import pipeline

app = Flask(__name__)

# PRAW Reddit instance
reddit = praw.Reddit(
    client_id="qJsvLNd6Goxgqz4LZ6yEbg",
    client_secret="KIUx33wL7W434pIy4lru06slhDjAEQ",
    user_agent="posts_ds"
)

# Sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form data
        subreddit_name = request.form.get("subreddit")
        category = request.form.get("category")

        # Fetch posts based on the category
        subreddit = reddit.subreddit(subreddit_name)
        if category == "hot":
            posts = subreddit.hot(limit=10)
        elif category == "new":
            posts = subreddit.new(limit=10)
        elif category == "top":
            posts = subreddit.top(limit=10)
        else:
            posts = subreddit.controversial(limit=10)

        # Collect post titles
        post_titles = [(post.id, post.title) for post in posts]
        return render_template("index.html", titles=post_titles, subreddit=subreddit_name, category=category)

    return render_template("index.html", titles=None)


@app.route("/visualize", methods=["POST"])
def visualize():
    # Get the selected post ID
    subreddit_name = request.form.get("subreddit")
    post_id = request.form.get("post_id")

    # Fetch the post and its comments
    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=None)
    comments = [comment.body for comment in submission.comments.list()]

    # Preprocess comments
    comments_dict = {submission.title: [[comment] for comment in comments]}
    results = []

    for title, comments in comments_dict.items():
        for idx, sub_comment in enumerate(comments, start=1):
            result = sentiment_pipeline(sub_comment)
            sentiment_score = result[0]['score']
            results.append([title, f"comment #{idx}", sentiment_score])

    # Create a DataFrame for scores
    sent_data = pd.DataFrame(results, columns=["title", "comment_number", "result"])

    # Visualize sentiment scores
    scores = sent_data['result'].to_numpy()
    x = np.linspace(0, 1, 500)
    noise = np.random.normal(loc=0.0, scale=5, size=(500, 500))
    base_field = np.interp(np.linspace(0, len(scores) - 1, 500), np.arange(len(scores)), scores)
    Z = base_field + noise
    Z = gaussian_filter(Z, sigma=100)
    Z_norm = (Z - Z.min()) / (Z.max() - Z.min()) * 2 - 1

    cmap = mcolors.LinearSegmentedColormap.from_list("", ['red', 'blue', 'green'])
    colors = cmap((Z_norm + 1) / 2)

    plt.imshow(colors, extent=[1, 0, 1, 0], origin='lower', aspect='auto')
    plt.axis('off')

    # Save the plot
    img_path = "static/sentiment.png"
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return render_template("viz.html", img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)
