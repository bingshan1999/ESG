import csv
import praw
import prawcore
from datetime import datetime, timezone, timedelta
from cred import REDDIT_CLIENT_ID,REDDIT_CLIENT_SECRET,REDDIT_PASSWORD,REDDIT_USER_AGENT,REDDIT_USERNAME
import requests
import time 
import logging

# Replace these values with your own credentials
client_id = REDDIT_CLIENT_ID
client_secret = REDDIT_CLIENT_SECRET
user_agent = REDDIT_USER_AGENT
username = REDDIT_USERNAME
password = REDDIT_PASSWORD

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the Reddit instance
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
    username=username,
    password=password
)

def fetch_posts_for_year(subreddit_name, year):
    subreddit = reddit.subreddit(subreddit_name)
    total_fetched = 0
    retry_attempts = 0

    start_date = datetime(year, 1, 1, tzinfo=timezone.utc)
    end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)

    logging.info(f"Fetching posts from subreddit: {subreddit_name} for the year {year}")

    with open('reddit_posts.csv', 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'score', 'id', 'url', 'num_comments', 'created', 'content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:  # If file is empty, write the header
            writer.writeheader()

        while True:
            try:
                logging.info("Entering try block to fetch posts.")
                for submission in subreddit.top(time_filter='all', limit=None):
                    post_time = datetime.fromtimestamp(submission.created_utc, timezone.utc)
                    if post_time < start_date or post_time >= end_date:
                        continue

                    logging.info(f"Fetched post with ID {submission.id}.")

                    comments = get_post_comments(submission.id)
                    comments_text = "\n\n".join(
                        [f"Comment by {comment['author']} at {datetime.fromtimestamp(comment['created'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00:00')}:\n{comment['body']}\nScore: {comment['score']}"
                         for comment in comments])
                    comments_text = comments_text.replace('\n', ' ')  # Replace newlines with spaces

                    post_data = {
                        'title': submission.title,
                        'score': submission.score,
                        'id': submission.id,
                        'url': submission.url,
                        'num_comments': submission.num_comments,
                        'created': datetime.fromtimestamp(submission.created_utc, timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00:00'),
                        'content': submission.selftext.replace('\n', ' ') + " " + comments_text  # Replace newlines with spaces
                    }

                    writer.writerow(post_data)
                    total_fetched += 1
                    logging.info(f"Fetched and saved {total_fetched} posts so far")
                    time.sleep(1)  # Respect Reddit’s rate limits

                logging.info("Exiting try block after successfully fetching posts.")
                break  # Exit the loop if fetching completes successfully

            except prawcore.exceptions.TooManyRequests as e:
                retry_attempts += 1
                wait_time = min(2 ** retry_attempts, 600)  # Exponential backoff with a cap at 600 seconds (10 minutes)
                wait_minutes = wait_time / 60  # Convert wait time to minutes
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds ({wait_minutes:.2f} minutes)... Attempt #{retry_attempts}")
                time.sleep(wait_time)

            except Exception as e:
                logging.error(f"An error occurred: {e}")
                break

    return total_fetched

def get_post_comments(post_id):
    submission = reddit.submission(id=post_id)
    comments = []
    try:
        submission.comments.replace_more(limit=None)  # Fetch all comments
        for top_level_comment in submission.comments:
            comments.append({
                'author': top_level_comment.author.name if top_level_comment.author else 'Deleted',
                'body': top_level_comment.body.replace('\n', ' '),  # Replace newlines with spaces
                'score': top_level_comment.score,
                'created': top_level_comment.created_utc
            })
    except Exception as e:
        logging.warning(f"An error occurred while fetching comments for post {post_id}: {e}")

    return comments

# Define the subreddit and year range
subreddit_name = 'Bitcoin'  # Replace with the subreddit you want to fetch data from
start_year = 2015
end_year = 2024

# Retrieve posts from the specified subreddit year by year
total_posts = 0
for year in range(start_year, end_year + 1):
    total_fetched = fetch_posts_for_year(subreddit_name, year)
    total_posts += total_fetched
    logging.info(f"Total number of posts fetched and saved for {year}: {total_fetched}")

logging.info(f"Total number of posts downloaded and saved: {total_posts}")
logging.info("Data has been written to reddit_posts.csv")