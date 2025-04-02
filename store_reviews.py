from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from transformers import BertTokenizer
import tensorflow as tf

app = FastAPI()

# ✅ Database Connection
DB_PARAMS = {
    "dbname": "sentiment_analysis",
    "user": "postgres",
    "password": "Rajsomu123@",
    "host": "localhost",
    "port": "5432",
}

def get_db_connection():
    return psycopg2.connect(**DB_PARAMS, cursor_factory=RealDictCursor)

# ✅ Load Fine-Tuned BERT Model & Tokenizer
MODEL_PATH = "fine_tuned_nlptown_bert_model"
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]  # Accessing the model's inference function

# ✅ Create Table
def create_table():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            id SERIAL PRIMARY KEY,
            review_text TEXT NOT NULL,
            sentiment VARCHAR(20)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

create_table()

# ✅ Request Model
class Review(BaseModel):
    review_text: str

# ✅ Function to Predict Sentiment
def predict_sentiment(review_text: str) -> str:
    inputs = tokenizer(review_text, return_tensors="tf", padding=True, truncation=True)
    
    # ✅ Pass input as a dictionary
    outputs = infer(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs.get("token_type_ids", tf.zeros_like(inputs["input_ids"]))
    )
    
    logits = outputs["logits"]
    pred_label = tf.argmax(logits, axis=1).numpy()[0]
    
    label_map = {
        0: 'negative',
        1: 'negative',
        2: 'neutral',
        3: 'positive',
        4: 'positive'
    }
    return label_map[pred_label]

# ✅ API to Store Reviews with Predicted Sentiment
@app.post("/add_review/")
def add_review(review: Review):
    try:
        sentiment = predict_sentiment(review.review_text)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO reviews (review_text, sentiment) VALUES (%s, %s) RETURNING id", 
                    (review.review_text, sentiment))
        review_id = cur.fetchone()["id"]
        conn.commit()
        cur.close()
        conn.close()
        return {"message": "Review added successfully", "review_id": review_id, "sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ✅ API to Fetch All Reviews
@app.get("/get_reviews/")
def get_reviews():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM reviews")
        reviews = cur.fetchall()
        cur.close()
        conn.close()
        return {"reviews": reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
