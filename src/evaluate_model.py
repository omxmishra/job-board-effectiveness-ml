import joblib
import pandas as pd
import os

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data_preprocessing import (
    load_data,
    drop_leakage_columns,
    split_features_target,
    encode_features
)


def evaluate(data_path):
    # Load artifacts
    model = joblib.load("models/logistic_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    columns = joblib.load("models/columns.pkl")

    # Load and preprocess data
    df = load_data(data_path)
    df = drop_leakage_columns(df)

    X, y = split_features_target(df)
    X_encoded = encode_features(X, reference_columns=columns)

    # Scale
    X_scaled = scaler.transform(X_encoded)

    # Predict probabilities
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Apply threshold (IMPORTANT — same as training decision)
    threshold = 0.45
    y_pred = (y_prob >= threshold).astype(int)

    # Metrics
    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))


if __name__ == "__main__":
    evaluate("data/raw/job_search_platform_efficacy_100k.csv")