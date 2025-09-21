# FastAPI → a Python framework to build APIs quickly.
# Pydantic (BaseModel) → used to define the structure of input data (like JSON payloads).
# joblib → used to load the trained SVM classifier (svm_model.pkl).
# SentenceTransformer → loads the saved BERT encoder (bert_encoder/).

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer

# Load models
# BERT = text → embeddings
# SVM = embeddings → prediction
svm_model = joblib.load("svm_model.pkl")
bert_model = SentenceTransformer("bert_encoder")

# This creates the API app
app = FastAPI()

# Allow requests from frontend
origins = [
    "http://localhost:3000",     # React/Next.js dev server
    "https://dreamcanvas-murex.vercel.app/"  # Your deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # Which domains can talk to your API
    allow_credentials=True,
    allow_methods=["*"],          # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],          # Allow all headers (e.g., Content-Type, Authorization)
)

# Defines the format of input data expected from the frontend.
# here api expects json object from the frontend - like 
# {
#   "description": "Some job posting text here..."
# }

class JobPost(BaseModel):
    description: str

# Receive request
# User sends a POST request to /predict with JSON containing the job description.
# Example request body:
# { "description": "Work from home, earn $5000 weekly, no experience needed..." }

@app.post("/predict")
def predict(post: JobPost):
    # Encode
    # Converts the job description into a numerical vector (embedding) using BERT.
    # This vector is what the SVM understands.
    
    embedding = bert_model.encode([post.description])
    
    # Predict
    # Feeds the embedding into the trained SVM.
    # Returns 1 if fraudulent (fake job), 0 if real.
    
    pred = svm_model.predict(embedding)[0]
    # Send Response
    # return {"prediction": int(pred)}
    # The API responds with a JSON object like:
    # { "prediction": 1 }
    return {"prediction": int(pred)}  # 1 = Fake, 0 = Real
