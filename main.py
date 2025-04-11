from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from transformers import BertTokenizer
import numpy as np

app = FastAPI()

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your React frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
model = tf.saved_model.load("fine_tuned_nlptown_bert_model_ver2")
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Label mapping
label_map = {
    0: 'negative',
    1: 'negative',
    2: 'neutral',
    3: 'positive',
    4: 'positive'
}

class TextRequest(BaseModel):
    text: str

@app.post("/play")
async def predict_sentiment(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty input text")
    
    try:
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="tf",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Run inference
        infer = model.signatures["serving_default"]
        outputs = infer(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            token_type_ids=tf.zeros_like(inputs['input_ids'])
        )
        
        # Process output
        logits = outputs['logits'].numpy()
        pred_label = np.argmax(logits, axis=1)[0]
        
        return {"sentiment": label_map[pred_label]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Test client setup (optional)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
