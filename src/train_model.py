import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def train_model(X, y, vectorizer):

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, preds))

    # Save directory
    os.makedirs("models", exist_ok=True)

    # Save model
    joblib.dump(model, "models/tfidf_model.pkl")

    # Save vectorizer
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("\n✅ Model and vectorizer saved successfully in 'models/' folder")

    return model