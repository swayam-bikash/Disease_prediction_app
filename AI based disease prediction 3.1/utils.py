import numpy as np
import pandas as pd


def create_input_vector(selected_symptoms, all_symptoms):
    input_vector = [0] * len(all_symptoms)
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            input_vector[all_symptoms.index(symptom)] = 1
    return input_vector


def predict_disease(selected_symptoms, model, all_symptoms):
    input_vector = create_input_vector(selected_symptoms, all_symptoms)
    # use DataFrame so feature names match what the model was trained with
    input_df = pd.DataFrame([input_vector], columns=all_symptoms)

    disease = model.predict(input_df)[0]

    # top 3 predictions with confidence
    top3 = []
    try:
        probs = model.predict_proba(input_df)[0]
        classes = model.classes_
        top_indices = np.argsort(probs)[::-1][:3]
        top3 = [
            {"disease": classes[i], "confidence": round(probs[i] * 100, 2)}
            for i in top_indices
        ]
        confidence = top3[0]["confidence"]
    except:
        confidence = 0

    important_symptoms = [s.replace("_", " ").title() for s in selected_symptoms[:5]]

    return disease, confidence, important_symptoms, top3
