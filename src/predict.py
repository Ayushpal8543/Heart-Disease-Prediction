import joblib
import pandas as pd

from src.recommendation import get_recommendation


# Load model and scaler
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict_patient(data):
    """
    Predict heart disease risk for a patient.
    """

    df = pd.DataFrame([data])

    # Scale input
    scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled)[0]

    probability = model.predict_proba(scaled)[0][1]

    result = get_recommendation(
        prediction,
        probability
    )

    return prediction, probability, result


if __name__ == "__main__":

    sample = {

        "male":1,
        "age":55,
        "education":2,
        "currentSmoker":1,
        "cigsPerDay":20,
        "BPMeds":0,
        "prevalentStroke":0,
        "prevalentHyp":1,
        "diabetes":0,
        "totChol":240,
        "sysBP":150,
        "diaBP":95,
        "BMI":30,
        "heartRate":80,
        "glucose":110

    }

    prediction, probability, report = predict_patient(sample)

    print(report)