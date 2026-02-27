# AI Resume Job Matcher

An AI-powered resume-to-job matching engine built using:

- TF-IDF Vectorization
- Cosine Similarity
- Hybrid Scoring Model (70% cosine + 30% keyword overlap)
- Explainable AI output (matched keywords)

---

## 🚀 Features

- Calculates match percentage between resume and job description
- Uses hybrid weighted scoring
- Returns matched keywords for transparency
- REST API endpoint using Flask

---

## 🛠 Tech Stack

- Python
- Flask
- Scikit-learn
- NumPy

---

## 📌 API Endpoint

POST `/match`

### Request Body (JSON)

```json
{
  "resume_text": "Python Flask SQL",
  "job_description": "Looking for Python developer with Flask"
}
