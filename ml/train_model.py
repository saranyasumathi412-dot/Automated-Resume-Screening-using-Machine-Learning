import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')

data = pd.read_csv("Resume.csv")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

data['cleaned'] = data['Resume'].apply(clean_text)

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['cleaned'])
y = data['Category']

model = MultinomialNB()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "vectorizer.pkl")

print("Model Saved Successfully ✅")
