
# 🕵️‍♀️ Job Posting Detection API

An **AI-powered backend service** that detects whether a job posting is **Real or Fake** using **NLP + Machine Learning**.
The system combines **BERT-based sentence embeddings** with a **Support Vector Machine (SVM)** classifier and exposes predictions via a **FastAPI REST API**.

This service is designed to be **production-ready**, **Dockerized**, and **deployed on Render**, and is integrated into the **DreamCanvas** platform.

---

## 🚀 Features

* 🤖 **NLP-powered classification** of job postings
* 🧠 **Sentence Transformers (BERT)** for semantic embeddings
* 📊 **SVM (LinearSVC)** for fake vs real prediction
* 🌐 **FastAPI REST API** with JSON responses
* 🐳 **Dockerized** for cloud deployment
* ☁️ **Deployed on Render** (Free tier compatible)
* 🔗 **Model artifact handling via GitHub Releases**

---

## 🧠 Model Architecture

1. **Input**: Job description text
2. **Embedding**:

   * Model: `paraphrase-MiniLM-L3-v2` (Sentence Transformers)
3. **Classifier**:

   * Model: `LinearSVC` (Scikit-learn)
4. **Output**:

   * `prediction`: `0` → Real, `1` → Fake
   * `confidence`: Pseudo-confidence based on distance from decision boundary

> ⚠️ Note: Confidence is not a true probability (LinearSVC does not support `predict_proba`).

---

## 🛠️ Tech Stack

| Layer         | Technology                   |
| ------------- | ---------------------------- |
| API           | FastAPI                      |
| ML            | Scikit-learn (LinearSVC)     |
| NLP           | Sentence-Transformers (BERT) |
| Runtime       | Python 3.10                  |
| Container     | Docker                       |
| Deployment    | Render                       |
| Model Storage | GitHub Releases              |

---

## 📂 Project Structure

```
JobPostingDetection/
├── api.py               # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── .gitignore           # Ignored files (models, cache, etc.)
└── README.md            # Project documentation
```

---

## 📦 Model Files

* `svm_model.pkl`

  * **Not committed to GitHub**
  * Hosted via **GitHub Releases**
  * Downloaded automatically at runtime

* BERT encoder

  * Loaded dynamically from Hugging Face
  * No large model files stored in the repository

---

## ⚙️ Setup & Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/PradnyaKulkarni2005/JobPostingDetection.git
cd JobPostingDetection
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

---

### 4️⃣ Test health endpoint

Open in browser:

```
http://localhost:8000/
```

Response:

```json
{"status":"API running"}
```

---

## 🔍 API Usage

### **POST** `/predict`

#### Request Body

```json
{
  "description": "Frontend Developer role with React.js experience..."
}
```

#### Response

```json
{
  "prediction": 0,
  "confidence": 0.84
}
```

* `prediction = 0` → Real job
* `prediction = 1` → Fake job

---

## 🌐 Live Deployment

**Base URL:**

```
https://job-posting-detection-api.onrender.com
```

### Health Check

```
GET /
```

### Prediction

```
POST /predict
```

> ⚠️ On Render Free tier, the first request may take longer due to cold start.

---

## 🐳 Docker Support

### Build Image

```bash
docker build -t job-posting-detection .
```

### Run Container

```bash
docker run -p 8000:8000 job-posting-detection
```

---

## 🔐 CORS Configuration

The API allows requests from:

* `http://localhost:3000`
* `https://dreamcanvas-murex.vercel.app`

This enables seamless integration with the **DreamCanvas frontend**.

---

## 🔮 Future Improvements

* Calibrated probabilities (`SVC(probability=True)`)
* Explanation of predictions (LLM-based)
* Batch job post analysis
* Persistent logging & analytics
* Rate limiting and auth
* Improved dataset & retraining

---

## 👩‍💻 Author

**Pradnya Kulkarni**

* GitHub: [@PradnyaKulkarni2005](https://github.com/PradnyaKulkarni2005)
* Project: **DreamCanvas – AI Career Coach**

---

## 📄 License

This project is open-source and feel free to contribute.

---


