import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import (
    load_data,
    drop_leakage_columns,
    split_features_target,
    encode_features
)


def train_pipeline(data_path):
    # Load data
    df = load_data(data_path)

    # Clean data
    df = drop_leakage_columns(df)

    # Split
    X, y = split_features_target(df)

    # Encode
    X_encoded = encode_features(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/logistic_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    # Save feature columns for inference
    joblib.dump(X_encoded.columns.tolist(), "models/columns.pkl")


if __name__ == "__main__":
    train_pipeline("data/raw/job_search_platform_efficacy_100k.csv")