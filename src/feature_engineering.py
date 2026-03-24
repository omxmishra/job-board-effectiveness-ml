import os
import pandas as pd
import joblib


# -------------------- Load Artifacts --------------------
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(BASE_DIR, "..", "models", "logistic_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")
    columns_path = os.path.join(BASE_DIR, "..", "models", "columns.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    columns = joblib.load(columns_path)

    return model, scaler, columns


# -------------------- Prediction Pipeline --------------------
def predict(input_df):
    model, scaler, columns = load_artifacts()

    # Encode input
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align columns with training data
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale input
    input_scaled = scaler.transform(input_encoded)

    # Predict probability
    prob = model.predict_proba(input_scaled)[:, 1]

    # Apply threshold
    threshold = 0.45
    prediction = (prob >= threshold).astype(int)

    return prediction, prob