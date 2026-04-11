from flask import Flask, render_template, request, redirect, session
import pandas as pd
import pickle
import json
import os
import sqlite3

from utils import predict_disease

app = Flask(__name__)
app.secret_key = "your_fixed_secret_key_here_do_not_change"

# make session last longer (important fix)
app.permanent_session_lifetime = 86400  # 24 hours


# ---------------------------
# DATABASE SETUP
# ---------------------------
def init_db():

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            password TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            disease TEXT,
            confidence REAL,
            symptoms TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ---------------------------
# LOAD MODEL
# ---------------------------
model = None

if os.path.exists("models/model.pkl"):
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    print("model not found, run model.py first")


# ---------------------------
# LOAD DATA
# ---------------------------
print("loading dataset...")

data = pd.read_csv("data/Training.csv")

# safety (handle NaN again)
data = data.fillna(0)

all_symptoms = list(data.columns[:-1])


# ---------------------------
# MOTIVATION MESSAGES
# ---------------------------
motivation_map = {
    "(vertigo) Paroymsal  Positional Vertigo": "Vertigo can feel disorienting, but it's very manageable. Simple repositioning exercises often bring quick relief. Take it slow and steady. 🌀",
    "AIDS": "Living with HIV/AIDS today is very different from the past. With proper treatment, people live long, healthy lives. You are not alone — support and care are available. 💙",
    "Acne": "Acne is incredibly common and very treatable. With the right skincare routine and guidance, clear skin is absolutely achievable. Be kind to yourself. 🌸",
    "Alcoholic hepatitis": "Your liver has a remarkable ability to heal when given the chance. Taking this step to understand your health is already a sign of strength. Recovery is possible. 💪",
    "Allergy": "Allergies are manageable with the right treatment plan. Identifying your triggers is the first step, and you're already on it. Relief is closer than you think. 🌼",
    "Arthritis": "Millions of people live active, fulfilling lives with arthritis. The right treatment and gentle movement can make a huge difference. Keep going. 🦾",
    "Bronchial Asthma": "Asthma is very controllable with proper medication and care. Many athletes and active people have asthma and thrive. You can too. 🌬️",
    "Cervical spondylosis": "Neck pain from cervical spondylosis improves greatly with physiotherapy and lifestyle adjustments. You're on the right path by seeking help early. 🧘",
    "Chicken pox": "Chicken pox is temporary and your body will recover fully. Rest, stay hydrated, and avoid scratching — you'll be back to normal soon. 🌟",
    "Chronic cholestasis": "With proper medical care and dietary adjustments, cholestasis can be well managed. Your body is resilient — trust the process. 💛",
    "Common Cold": "The common cold is your immune system doing its job. Rest up, stay warm, drink fluids, and you'll be feeling better in no time. 🍵",
    "Dengue": "Dengue recovery requires rest and hydration, and most people recover fully within a week or two. You're taking the right steps by acting early. 💧",
    "Diabetes ": "Diabetes is one of the most manageable chronic conditions. With the right diet, exercise, and medication, you can live a completely normal life. 🥗",
    "Dimorphic hemmorhoids(piles)": "Piles are more common than you think and very treatable. Dietary changes and medical treatment bring significant relief. You're not alone in this. 🌿",
    "Drug Reaction": "Drug reactions can be alarming, but stopping the trigger and getting medical attention leads to quick recovery in most cases. Stay calm and act promptly. 🏥",
    "Fungal infection": "Fungal infections respond well to antifungal treatment. With proper hygiene and medication, you'll clear this up quickly. 🍃",
    "GERD": "GERD is very common and highly manageable with diet changes and medication. Small lifestyle adjustments can bring major relief. You've got this. 🥦",
    "Gastroenteritis": "Gastroenteritis usually clears up on its own within a few days. Stay hydrated, rest well, and your body will bounce back. 💪",
    "Heart attack": "Seeking help early is the most important thing you could have done. Modern cardiac care is incredibly advanced — recovery is very possible. Stay strong. ❤️",
    "Hepatitis B": "Hepatitis B is manageable with antiviral therapy. Many people with Hepatitis B live long, healthy lives with proper care. You are not alone. 💙",
    "Hepatitis C": "Hepatitis C is now curable in most cases with modern antiviral treatments. This is a very hopeful time — treatment works. 🌟",
    "Hepatitis D": "With proper medical supervision, Hepatitis D can be managed effectively. Early action makes a real difference. Stay positive. 💛",
    "Hepatitis E": "Hepatitis E usually resolves on its own with rest and hydration. Most people recover fully within a few weeks. Take it easy and you'll be fine. 🌿",
    "Hypertension ": "High blood pressure is one of the most controllable conditions. Lifestyle changes and medication can bring it to a healthy range. Small steps, big results. 🧘",
    "Hyperthyroidism": "Hyperthyroidism is very treatable. With the right medication or therapy, your thyroid levels can be balanced and you'll feel like yourself again. ⚡",
    "Hypoglycemia": "Low blood sugar is manageable with the right diet and monitoring. Once you understand your triggers, you can stay ahead of it easily. 🍎",
    "Hypothyroidism": "Hypothyroidism is easily managed with daily medication. Most people feel completely normal once their levels are balanced. You'll get there. 🌤️",
    "Impetigo": "Impetigo heals quickly with antibiotic treatment. Keep the area clean and follow your doctor's advice — you'll clear it up fast. 🌸",
    "Jaundice": "Jaundice is a symptom your body uses to signal it needs help. With proper treatment and rest, most people recover fully. Your body is working hard for you. 💛",
    "Malaria": "Malaria is very treatable when caught early. With the right antimalarial medication, most people recover completely. You're doing the right thing. 🌿",
    "Migraine": "Migraines are challenging but very manageable with the right treatment plan. Many people find great relief through medication and lifestyle changes. Hang in there. 🧠",
    "Osteoarthristis": "Osteoarthritis doesn't have to slow you down. Physical therapy, medication, and staying active can keep you moving comfortably. Keep going. 🦵",
    "Paralysis (brain hemorrhage)": "Recovery from brain hemorrhage takes time, but the brain has an incredible ability to heal and adapt. Every small step forward matters. Stay strong. 🧠💪",
    "Peptic ulcer diseae": "Peptic ulcers heal well with the right medication and dietary care. Avoiding triggers and following treatment brings lasting relief. You're on the right track. 🥗",
    "Pneumonia": "Pneumonia responds well to treatment when addressed early. Rest, medication, and fluids are your best allies right now. You'll breathe easy again soon. 🌬️",
    "Psoriasis": "Psoriasis is a manageable condition and many people live comfortably with it. Modern treatments have come a long way — relief is very achievable. 🌸",
    "Tuberculosis": "TB is curable with a full course of antibiotics. Millions of people recover from TB every year. Stay consistent with your treatment and you will too. 💪",
    "Typhoid": "Typhoid responds well to antibiotics and most people recover fully. Rest, hydration, and completing your medication course are key. You'll be well soon. 🌟",
    "Urinary tract infection": "UTIs are among the most common and easily treated infections. A short course of antibiotics usually clears it up completely. Relief is on the way. 💧",
    "Varicose veins": "Varicose veins are very common and manageable. Compression, elevation, and medical treatment can significantly reduce discomfort. You're in good hands. 🦵",
    "hepatitis A": "Hepatitis A usually resolves on its own with rest and good nutrition. Most people recover fully within a few weeks. Take care of yourself and you'll be fine. 🌿",
}

default_motivation = "🌿 Don't panic. This is an AI-based prediction to help guide you, not a final diagnosis. Many conditions are very treatable when caught early. Stay calm and consult a qualified doctor. You've got this. 💪"

# ---------------------------
# LOAD DISEASE INFO
# ---------------------------
if os.path.exists("data/disease_info.json"):
    with open("data/disease_info.json", "r") as f:
        disease_info = json.load(f)
else:
    disease_info = {}


# ---------------------------
# HOME
# ---------------------------
@app.route("/")
def home():

    # check if logged in
    if "user" not in session:
        return redirect("/login")

    return render_template("index.html", symptoms=all_symptoms, username=session["user"])


# ---------------------------
# LOGIN
# ---------------------------
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (email, password)
        )

        user = cur.fetchone()

        conn.close()

        if user:
            session["user"] = email
            session.permanent = True
            return redirect("/")
        else:
            return render_template("login.html", error="Invalid email or password")

    return render_template("login.html")


# ---------------------------
# SIGNUP
# ---------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "GET":
        return render_template("signup.html")

    email = request.form.get("email")
    password = request.form.get("password")

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    existing_user = cur.fetchone()

    if existing_user:
        conn.close()
        return render_template("signup.html", error="An account with this email already exists")

    cur.execute(
        "INSERT INTO users (email, password) VALUES (?, ?)",
        (email, password)
    )

    conn.commit()
    conn.close()

    return redirect("/login")


# ---------------------------
# LOGOUT
# ---------------------------
@app.route("/logout")
def logout():

    session.pop("user", None)
    return redirect("/login")


# ---------------------------
# PREDICT
# ---------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect("/login")

    selected_symptoms = request.form.getlist("symptoms")

    if not selected_symptoms:
        return "Please select at least one symptom"

    disease, confidence, important_symptoms = predict_disease(
        selected_symptoms,
        model,
        all_symptoms
    )

    info = disease_info.get(disease, {
        "description": "No description available",
        "causes": "Not available",
        "precautions": "Consult doctor",
        "medicines": "Consult doctor"
    })

    # save to history
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (email, disease, confidence, symptoms) VALUES (?, ?, ?, ?)",
        (session["user"], disease, confidence, ", ".join(selected_symptoms))
    )
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        disease=disease,
        confidence=confidence,
        symptoms=selected_symptoms,
        important_symptoms=important_symptoms,
        info=info,
        motivation=motivation_map.get(disease, default_motivation),
        username=session["user"]
    )


# ---------------------------
# HISTORY
# ---------------------------
@app.route("/history")
def history():

    if "user" not in session:
        return redirect("/login")

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT disease, confidence, symptoms, created_at FROM predictions WHERE email=? ORDER BY created_at DESC",
        (session["user"],)
    )
    records = cur.fetchall()
    conn.close()

    return render_template("history.html", records=records, username=session["user"])


# ---------------------------
# CHAT
# ---------------------------
@app.route("/chat", methods=["POST"])
def chat():
    from flask import jsonify

    if "user" not in session:
        return jsonify({"reply": "Please log in to use the chat assistant."})

    message = request.form.get("message", "").lower()

    # match symptoms mentioned in the message
    found_symptoms = []
    for symptom in all_symptoms:
        keyword = symptom.replace("_", " ").lower()
        if keyword in message:
            found_symptoms.append(symptom)

    if not found_symptoms:
        return jsonify({
            "reply": "I couldn't detect any known symptoms in your message. Try describing them more clearly, e.g. 'I have fever, headache and nausea'."
        })

    # run prediction
    disease, confidence, _ = predict_disease(found_symptoms, model, all_symptoms)

    symptom_labels = ", ".join([s.replace("_", " ") for s in found_symptoms])
    reply = (
        f"Based on the symptoms I detected ({symptom_labels}), "
        f"the most likely condition is {disease} "
        f"(confidence: {confidence}%). "
        f"Please consult a doctor for a proper diagnosis."
    )

    return jsonify({"reply": reply})


# ---------------------------
# DASHBOARD
# ---------------------------
@app.route("/dashboard")
def dashboard():

    if "user" not in session:
        return redirect("/login")

    total_symptoms = len(all_symptoms)
    total_diseases = len(data["prognosis"].unique())

    return render_template(
        "dashboard.html",
        total_symptoms=total_symptoms,
        total_diseases=total_diseases
    )


# ---------------------------
# RUN
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)