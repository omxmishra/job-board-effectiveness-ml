import pandas as pd
import joblib


def load_artifacts():
    model = joblib.load("../models/logistic_model.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    columns = joblib.load("../models/columns.pkl")
    return model, scaler, columns


def predict(input_df):
    model, scaler, columns = load_artifacts()

    # Encode
    input_encoded = pd.get_dummies(input_df, drop_first=True)

    # Align columns
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Scale
    input_scaled = scaler.transform(input_encoded)

    # Predict probability
    prob = model.predict_proba(input_scaled)[:, 1]

    # Apply threshold
    prediction = (prob >= 0.45).astype(int)

    return prediction, prob