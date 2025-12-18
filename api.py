# FastAPI → a Python framework to build APIs quickly.
# Pydantic (BaseModel) → used to define the structure of input data (like JSON payloads).
# joblib → used to load the trained SVM classifier (svm_model.pkl).
# SentenceTransformer → loads the saved BERT encoder (bert_encoder/).

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
import os
import requests


# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SVM_MODEL_PATH = os.path.join(BASE_DIR, "svm_model.pkl")

# Public GitHub Release URL (your model)
SVM_MODEL_URL = (
    "https://github.com/PradnyaKulkarni2005/"
    "JobPostingDetection/releases/download/v1.0/svm_model.pkl"
)

# Download SVM model if not present

if not os.path.exists(SVM_MODEL_PATH):
    print(" Downloading SVM model...")
    response = requests.get(SVM_MODEL_URL, timeout=60)
    response.raise_for_status()  # fail fast if download breaks

    with open(SVM_MODEL_PATH, "wb") as f:
        f.write(response.content)

    print("✅ SVM model downloaded")

# Load models

svm_model = joblib.load(SVM_MODEL_PATH)

# Auto-download public BERT model
bert_model = SentenceTransformer(
    "paraphrase-MiniLM-L3-v2",
    device="cpu"
)




# This creates the API app
app = FastAPI()

# Allow requests from frontend
origins = [
    "http://localhost:3000",     # React/Next.js dev server
    "https://dreamcanvas-murex.vercel.app"  # Your deployed frontend
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
@app.get("/")
def health():
    return {"status": "API running"}

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
    
    # Distance from decision boundary
    score = svm_model.decision_function(embedding)

    # Convert to pseudo-confidence (0–1 range)
    confidence = float(abs(score[0]) / (abs(score[0]) + 1))

    return {
        "prediction": int(pred),
        "confidence": confidence
    } # 1 = Fake, 0 = Real


