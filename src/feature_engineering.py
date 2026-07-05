from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

def tfidf_features(text_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer


def train_word2vec(sentences):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=2)
    return model


def word2vec_features(model, sentences):
    def vectorize(text):
        words = text.split()
        vecs = [model.wv[w] for w in words if w in model.wv]

        if len(vecs) == 0:
            return np.zeros(100)

        return np.mean(vecs, axis=0)

    return np.array([vectorize(t) for t in sentences])