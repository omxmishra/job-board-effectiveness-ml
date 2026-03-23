import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    return df


def drop_leakage_columns(df):
    drop_cols = [
        "Student_ID",
        "First_Round_Interviews",
        "Second_Round_Interviews",
        "Time_to_Offer_Days",
        "Offer_Salary",
        "Company_Size_Offered",
        "Role_Relevance",
        "Accepted_Offer"
    ]
    return df.drop(columns=drop_cols)


def split_features_target(df):
    X = df.drop("Offer_Received", axis=1)
    y = df["Offer_Received"]
    return X, y


def encode_features(X, reference_columns=None):
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Align columns for inference
    if reference_columns is not None:
        X_encoded = X_encoded.reindex(columns=reference_columns, fill_value=0)
    
    return X_encoded