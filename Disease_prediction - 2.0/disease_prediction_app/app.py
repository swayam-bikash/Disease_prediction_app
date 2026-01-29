from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and helpers
model = pickle.load(open("disease_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))
symptoms = pickle.load(open("symptom_list.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html", symptoms=symptoms)

@app.route("/predict", methods=["POST"])
def predict():
    selected_symptoms = request.form.getlist("symptoms")

    input_data = np.zeros(len(symptoms))

    for symptom in selected_symptoms:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            input_data[index] = 1

    input_data = input_data.reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max()

    disease = le.inverse_transform([prediction])[0]
    confidence = round(probability * 100, 2)

    return render_template(
        "result.html",
        disease=disease,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
