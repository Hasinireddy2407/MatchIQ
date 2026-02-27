from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


def keyword_overlap_score(resume_text, job_description):
    resume_words = set(resume_text.split())
    job_words = set(job_description.split())

    common_words = resume_words.intersection(job_words)

    if len(job_words) == 0:
        return 0, []

    score = len(common_words) / len(job_words)

    return score, list(common_words)


def calculate_match(resume_text, job_description):

    resume_text = preprocess(resume_text)
    job_description = preprocess(job_description)

    documents = [resume_text, job_description]

    # TF-IDF Similarity
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Keyword Overlap
    keyword_score, matched_keywords = keyword_overlap_score(resume_text, job_description)

    # Hybrid Score (weighted)
    final_score = (0.7 * cosine_sim) + (0.3 * keyword_score)

    match_percentage = round(final_score * 100, 2)

    return {
        "match_percentage": match_percentage,
        "cosine_similarity": round(cosine_sim * 100, 2),
        "keyword_score": round(keyword_score * 100, 2),
        "matched_keywords": matched_keywords[:10]   # limit to top 10
    }