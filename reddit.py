import requests
from pydantic import BaseModel
import psycopg2
import re
import tensorflow as tf
from transformers import BertTokenizer
from sqlalchemy import create_engine

# ✅ Load the Fine-Tuned BERT Model
model = tf.saved_model.load("fine_tuned_nlptown_bert_model")
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
infer = model.signatures["serving_default"]

# ✅ Database Connection
DB_CONFIG = {
    "dbname": "sentiment_analysis",
    "user": "postgres",
    "password": "Rajsomu123@",
    "host": "localhost",
    "port": "5432"
}
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# ✅ Create Table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS redditReviews (
        id SERIAL PRIMARY KEY,
        review TEXT NOT NULL,
        sentiment VARCHAR(20)
    );
""")
conn.commit()


# ✅ Function to Limit Text Length
def clean_text(text):
    
    ascii_only = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return ascii_only.encode('windows-1252', errors='ignore').decode('windows-1252')

def limit_text_length(text, max_words=20):
    words = text.split()[:max_words]
    return " ".join(words)

# ✅ Function to Predict Sentiment
def predict_sentiment(review):
    inputs = tokenizer(review, return_tensors="tf", padding=True, truncation=True)
    
    outputs = infer(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs.get("token_type_ids", tf.zeros_like(inputs["input_ids"]))
    )
    logits = outputs['logits']
    pred_label = tf.argmax(logits, axis=1).numpy()[0]

    # ✅ Map Prediction to Sentiment Label
    label_map = {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}
    return label_map[pred_label]

# ✅ Fetch Reddit Posts
query = "lollapalooza"
sort_by = "new"
limit = 10
subreddit = "all"
url = f"https://www.reddit.com/r/{subreddit}/search.json?q={query}&sort={sort_by}&limit={limit}&restrict_sr=off"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
if response.status_code != 200:
    print(f"Failed to fetch data: {response.status_code}")
    exit()

data = response.json()
posts = data.get("data", {}).get("children", [])

if not posts:
    print("No posts found.")
else:
    for post in posts:
        post_text = post["data"].get("selftext", "No content available")
        post_text = limit_text_length(post_text)

        # ✅ Predict Sentiment
        sentiment = predict_sentiment(post_text)
        new_post_text = clean_text(post_text)

        # ✅ Store in Database
        cursor.execute("INSERT INTO redditReviews (review, sentiment) VALUES (%s, %s)", (new_post_text, sentiment))
        conn.commit()

        print(f"Stored in DB: {post_text[:50]}... -> Sentiment: {sentiment}")

# ✅ Close DB Connection

cursor.close()
conn.close()
