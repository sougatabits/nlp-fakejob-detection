from src.data_cleaning import load_data, clean_data
from src.data_preprocessing import preprocess
from src.feature_engineering import tfidf_features
from src.train_model import train_model


# 📌 Load data
indian_jobs_data = load_data("data/synthetic_indian_jobs.csv")

# 📌 Clean data
indian_jobs_data = clean_data(indian_jobs_data)

# 📌 Combine text
indian_jobs_data['text'] = (
    indian_jobs_data['title'].astype(str) + " " +
    indian_jobs_data['label'].astype(str)
)

# 📌 Preprocess text
indian_jobs_data['clean_text'] = indian_jobs_data['text'].apply(preprocess)

# 📌 Feature extraction (TF-IDF + vectorizer)
X, vectorizer = tfidf_features(indian_jobs_data['clean_text'])
y = indian_jobs_data['label']

# 📌 Train model (IMPORTANT: pass vectorizer)
model = train_model(X, y, vectorizer)