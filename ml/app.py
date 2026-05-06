from flask import Flask, render_template, request
import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from pdfminer.high_level import extract_text
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")
tfidf = joblib.load("vectorizer.pkl")

# Skill list (simple dictionary)
skills_list = [
    "python", "java", "c++", "sql", "django",
    "flask", "machine learning", "html", "css",
    "javascript", "selenium", "testing"
]

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    if 'resume' not in request.files:
        return "No file uploaded"

    file = request.files['resume']
    file.save("uploaded_resume.pdf")

    # Extract text
    resume_text = extract_text("uploaded_resume.pdf")

    # Clean text
    cleaned = clean_text(resume_text)
    resume_tfidf = tfidf.transform([cleaned])

    # Prediction
    prediction = model.predict(resume_tfidf)[0]
    proba = model.predict_proba(resume_tfidf)
    confidence = round(np.max(proba) * 100, 2)

    # Skill Extraction
    found_skills = []
    for skill in skills_list:
        if skill in resume_text.lower():
            found_skills.append(skill)

    # Word Count
    word_count = len(resume_text.split())

    # Resume Length Suggestion
    if word_count < 150:
        suggestion = "Resume is too short. Add more details."
    elif word_count > 800:
        suggestion = "Resume is too long. Try to make it concise."
    else:
        suggestion = "Resume length looks good."

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           skills=found_skills,
                           word_count=word_count,
                           suggestion=suggestion)

if __name__ == "__main__":
    app.run(debug=True)