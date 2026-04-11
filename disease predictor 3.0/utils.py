import numpy as np


def create_input_vector(selected_symptoms, all_symptoms):
    input_vector = [0] * len(all_symptoms)

    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            index = all_symptoms.index(symptom)
            input_vector[index] = 1

    return input_vector


def predict_disease(selected_symptoms, model, all_symptoms):

    # convert symptoms to model input
    input_vector = create_input_vector(selected_symptoms, all_symptoms)
    input_array = np.array(input_vector).reshape(1, -1)

    # predict disease
    disease = model.predict(input_array)[0]

    # get confidence if model supports probability
    try:
        probs = model.predict_proba(input_array)
        confidence = max(probs[0]) * 100
        confidence = round(confidence, 2)
    except:
        confidence = 0

    # simple explanation (show selected symptoms nicely)
    important_symptoms = []
    for s in selected_symptoms[:5]:
        clean_name = s.replace("_", " ").title()
        important_symptoms.append(clean_name)

    return disease, confidence, important_symptoms