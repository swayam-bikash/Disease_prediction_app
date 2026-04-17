from flask import Flask, render_template, request, redirect, session, jsonify, abort, Response
from datetime import timedelta, datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import pickle
import json
import os
import sqlite3
import requests
import uuid
import csv
import io
import sys
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

from utils import predict_disease

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"

import subprocess, threading

def _start_ollama():
    ollama_exe = os.environ.get(
        "OLLAMA_EXE",
        r"C:\Users\swayam bikash\AppData\Local\Programs\Ollama\ollama.exe"
    )
    if os.path.exists(ollama_exe):
        subprocess.Popen([ollama_exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

threading.Thread(target=_start_ollama, daemon=True).start()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fallback_dev_key_change_in_production")
app.permanent_session_lifetime = timedelta(days=1)
app.config["ADMIN_EMAILS"] = set(filter(None, os.environ.get("ADMIN_EMAILS", "").split(",")))
app.config["GOOGLE_MAPS_API_KEY"] = os.environ.get("GOOGLE_MAPS_API_KEY", "")


@app.context_processor
def inject_admin():
    return {"is_admin": session.get("user") in app.config["ADMIN_EMAILS"]}


# ---------------------------
# DATABASE SETUP
# ---------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE,
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


def migrate_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    # Add users.created_at
    try:
        cur.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
        conn.commit()
    except Exception:
        pass

    # Add predictions.bookmarked
    try:
        cur.execute("ALTER TABLE predictions ADD COLUMN bookmarked INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass

    # Add predictions.severity_data
    try:
        cur.execute("ALTER TABLE predictions ADD COLUMN severity_data TEXT")
        conn.commit()
    except Exception:
        pass

    # Add predictions.share_token
    try:
        cur.execute("ALTER TABLE predictions ADD COLUMN share_token TEXT")
        conn.commit()
    except Exception:
        pass

    # Create chat_messages table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            role TEXT,
            message TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()
migrate_db()


# ---------------------------
# SEED ADMIN ACCOUNT
# ---------------------------
def seed_admin():
    admin_email = os.environ.get("ADMIN_EMAIL", "")
    admin_password = os.environ.get("ADMIN_PASSWORD", "")
    if not admin_email or not admin_password:
        print("ADMIN_EMAIL or ADMIN_PASSWORD not set in .env — skipping admin seed.")
        return
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE email=?", (admin_email,))
    if not cur.fetchone():
        cur.execute("INSERT INTO users (email, password) VALUES (?, ?)",
                    (admin_email, generate_password_hash(admin_password)))
        conn.commit()
        print(f"Admin account created: {admin_email}")
    conn.close()

seed_admin()


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
data = pd.read_csv("data/Training.csv")
data = data.fillna(0)
all_symptoms = [c for c in data.columns if c != "prognosis" and not c.startswith("Unnamed")]


# ---------------------------
# MOTIVATION MESSAGES
# ---------------------------
motivation_map = {
    "Common Cold": "Rest well, stay hydrated, and you'll recover soon. 💪",
    "Diabetes ": "Manageable with lifestyle and medication. 🥗",
    "Heart attack": "Immediate medical care is critical. ❤️"
}
default_motivation = "🌿 This is an AI-based prediction. Consult a doctor."


# ---------------------------
# LOAD DISEASE INFO
# ---------------------------
if os.path.exists("data/disease_info.json"):
    with open("data/disease_info.json", "r") as f:
        disease_info = json.load(f)
else:
    disease_info = {}


# ---------------------------
# FOLLOW-UP MAP (Task 22)
# ---------------------------
FOLLOWUP_MAP = {
    "fever": "How long have you had the fever, and have you measured your temperature?",
    "headache": "Is the headache constant or intermittent, and where exactly does it hurt?",
    "cough": "Is the cough dry or productive? How long have you had it?",
    "chest_pain": "Is the chest pain sharp or dull? Does it radiate to your arm or jaw?",
    "chest pain": "Is the chest pain sharp or dull? Does it radiate to your arm or jaw?",
    "fatigue": "How long have you been feeling fatigued? Is it affecting your daily activities?",
    "nausea": "Is the nausea accompanied by vomiting? Does it occur after eating?",
    "vomiting": "How many times have you vomited? Is there any blood in the vomit?",
    "dizziness": "Is the dizziness constant or does it come in episodes? Do you feel like the room is spinning?",
    "rash": "Where is the rash located? Is it itchy, painful, or spreading?",
    "shortness_of_breath": "Does the shortness of breath occur at rest or only with activity?",
    "back_pain": "Is the back pain sharp or dull? Does it radiate down your legs?",
    "stomach_pain": "Where exactly is the stomach pain? Is it constant or comes and goes?",
    "sore_throat": "How long have you had the sore throat? Is it painful to swallow?",
}


def detect_symptoms(message, all_symptoms):
    """Return list of matched symptom keywords from the message."""
    normalized = message.lower().replace("_", " ").strip()
    matched = []
    for symptom in all_symptoms:
        symptom_readable = symptom.replace("_", " ").lower()
        if symptom_readable in normalized:
            matched.append(symptom)
    return matched


# ---------------------------
# ADMIN DECORATOR (Task 24)
# ---------------------------
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user" not in session:
            return redirect("/login")
        if session["user"] not in app.config["ADMIN_EMAILS"]:
            return "403 Forbidden: Admin access required.", 403
        return f(*args, **kwargs)
    return decorated


# ---------------------------
# ROUTES
# ---------------------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect("/login")
    return render_template("index.html", username=session["user"])


@app.route("/symptoms")
def symptoms():
    if "user" not in session:
        return redirect("/login")
    return render_template("symptoms.html", symptoms=all_symptoms, username=session["user"])


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()
        if user and check_password_hash(user[2], password):
            session["user"] = email
            session.permanent = True
            return redirect("/intro")
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "GET":
        return render_template("signup.html")
    email = request.form.get("email")
    password = request.form.get("password")
    hashed = generate_password_hash(password)
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=?", (email,))
    if cur.fetchone():
        conn.close()
        return render_template("signup.html", error="Account already exists")
    cur.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed))
    conn.commit()
    conn.close()
    return redirect("/login")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return redirect("/login")
    selected_symptoms = request.form.getlist("symptoms")
    if not selected_symptoms:
        return "Please select at least one symptom"

    disease, confidence, important_symptoms, top3 = predict_disease(selected_symptoms, model, all_symptoms)

    info = disease_info.get(disease, {
        "description": "No description available",
        "causes": "Not available",
        "precautions": "Consult doctor",
        "medicines": "Consult doctor"
    })

    # Build severity dict from hidden form inputs (severity_<symptom>)
    severity_dict = {}
    for symptom in selected_symptoms:
        val = request.form.get(f"severity_{symptom}", "").strip().lower()
        if val in ("mild", "moderate", "severe"):
            severity_dict[symptom] = val
    severity_json = json.dumps(severity_dict) if severity_dict else None

    share_token = str(uuid.uuid4())

    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (email, disease, confidence, symptoms, severity_data, share_token) VALUES (?, ?, ?, ?, ?, ?)",
        (session["user"], disease, confidence, ", ".join(selected_symptoms), severity_json, share_token)
    )
    prediction_id = cur.lastrowid
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        disease=disease,
        confidence=confidence,
        symptoms=selected_symptoms,
        important_symptoms=important_symptoms,
        top3=top3,
        info=info,
        motivation=motivation_map.get(disease, default_motivation),
        username=session["user"],
        severity_data=severity_dict,
        prediction_id=prediction_id,
        bookmarked=0,
        share_token=share_token,
        maps_api_key=app.config["GOOGLE_MAPS_API_KEY"],
        disease_info_json=json.dumps(disease_info)
    )


@app.route("/history/delete/<int:record_id>", methods=["POST"])
def delete_history(record_id):
    if "user" not in session:
        return redirect("/login")
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE id=? AND email=?", (record_id, session["user"]))
    conn.commit()
    conn.close()
    return redirect("/history")


@app.route("/history/delete_all", methods=["POST"])
def delete_all_history():
    if "user" not in session:
        return redirect("/login")
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE email=?", (session["user"],))
    conn.commit()
    conn.close()
    return redirect("/history")


@app.route("/history_intro")
def history_intro():
    if "user" not in session:
        return redirect("/login")
    return render_template("history_intro.html")


@app.route("/history")
def history():
    if "user" not in session:
        return redirect("/login")
    filter_param = request.args.get("filter", "")
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    if filter_param == "bookmarked":
        cur.execute(
            "SELECT id, disease, confidence, symptoms, created_at, bookmarked FROM predictions WHERE email=? AND bookmarked=1 ORDER BY created_at DESC",
            (session["user"],)
        )
    else:
        cur.execute(
            "SELECT id, disease, confidence, symptoms, created_at, bookmarked FROM predictions WHERE email=? ORDER BY created_at DESC",
            (session["user"],)
        )
    records = cur.fetchall()

    # Chart data: predictions grouped by date for the last 30 days
    cur.execute(
        """SELECT DATE(created_at) as day, COUNT(*) as cnt
           FROM predictions
           WHERE email=? AND created_at >= DATE('now', '-30 days')
           GROUP BY day
           ORDER BY day ASC""",
        (session["user"],)
    )
    chart_data = [{"date": row[0], "count": row[1]} for row in cur.fetchall()]

    conn.close()
    return render_template("history.html", records=records, username=session["user"], filter=filter_param, chart_data=chart_data)


@app.route("/bookmark/<int:prediction_id>", methods=["POST"])
def bookmark(prediction_id):
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT bookmarked FROM predictions WHERE id=? AND email=?", (prediction_id, session["user"]))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Not found"}), 404
    new_val = 0 if row[0] else 1
    cur.execute("UPDATE predictions SET bookmarked=? WHERE id=? AND email=?", (new_val, prediction_id, session["user"]))
    conn.commit()
    conn.close()
    return jsonify({"bookmarked": new_val})


@app.route("/share/<token>")
def share_result(token):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT disease, confidence, symptoms FROM predictions WHERE share_token=?", (token,))
    row = cur.fetchone()
    conn.close()
    if not row:
        abort(404)
    disease, confidence, symptoms_str = row
    symptoms = [s.strip() for s in symptoms_str.split(",") if s.strip()]
    info = disease_info.get(disease, {
        "description": "No description available",
        "causes": "Not available",
        "precautions": "Consult doctor",
        "medicines": "Consult doctor"
    })
    return render_template(
        "share_result.html",
        disease=disease,
        confidence=confidence,
        symptoms=symptoms,
        info=info
    )


@app.route("/chat", methods=["POST"])
def chat():
    if "user" not in session:
        return jsonify({"reply": "Please log in to use the chat assistant."})

    message = request.form.get("message", "").strip()
    if not message:
        return jsonify({"reply": "Please enter a message."})

    # Task 22.4: optional context from previous bot message
    context = request.form.get("context", "").strip()

    normalized = message.lower().replace("_", " ").strip()

    # greeting handler
    greetings = {"hi","hello","hey","hii","helo","howdy","sup","yo","greetings","good morning","good afternoon","good evening","good night"}
    if normalized in greetings or any(normalized.startswith(g) for g in greetings):
        reply = "👋 Hello! I'm Dr. Atlas, your AI medical assistant. How are you feeling today? Describe your symptoms and I'll help you out! 🩺"
        # persist greeting exchange
        try:
            conn = sqlite3.connect("users.db")
            cur = conn.cursor()
            cur.execute("INSERT INTO chat_messages (email, role, message) VALUES (?, ?, ?)", (session["user"], "user", message))
            cur.execute("INSERT INTO chat_messages (email, role, message) VALUES (?, ?, ?)", (session["user"], "assistant", reply))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[chat] DB insert failed: {e}", file=sys.stderr)
        return jsonify({"reply": reply})

    # basic english word check — must contain at least one common word
    common_words = {
        "i","my","have","has","feel","feeling","pain","ache","hurt","hurts",
        "fever","cold","cough","headache","stomach","chest","back","throat",
        "nausea","vomit","dizzy","tired","weak","rash","swelling","bleeding",
        "breath","breathing","heart","blood","sugar","bp","pressure","allergy",
        "infection","disease","medicine","tablet","doctor","symptom","sick",
        "ill","unwell","help","what","why","how","is","are","can","should",
        "the","a","an","and","or","with","since","for","days","hours","week",
        "diabetes","flu","cold","covid","cancer","injury","wound","burn",
        "itching","burning","loss","weight","appetite","sleep","stress"
    }

    words = set(normalized.replace(",","").replace(".","").split())
    if not words.intersection(common_words):
        return jsonify({"reply": "I didn't understand that. Please describe your symptoms or health concern in plain English. 😊"})

    # ML prediction from message keywords
    ml_result = ""
    try:
        matched = [s for s in all_symptoms if s.replace("_", " ") in normalized or s in normalized]
        if matched:
            ml_disease, ml_conf, _, ml_top3 = predict_disease(matched, model, all_symptoms)
            top3_text = "\n".join([f"- {t['disease']} ({t['confidence']}%)" for t in ml_top3])
            ml_result = f"\n\n🤖 ML Prediction based on detected symptoms:\n{top3_text}"
    except Exception:
        ml_result = ""

    # Build prompt with optional context (Task 22.4)
    prompt_prefix = f'User previously mentioned: {context}\n\n' if context else ""
    prompt = f"""{prompt_prefix}You are MediBot, a medical assistant. The user says: "{message}"

List the most likely medical conditions, what they should do, and when to see a doctor. Be specific and concise. No formatting, just plain sentences."""

    # Single AI call
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 250, "temperature": 0.4, "top_p": 0.9}
        }, timeout=60)

        raw = response.json().get("response", "").strip()
        if not raw:
            raw = "I couldn't generate a response. Please try rephrasing."

        # structure the raw response into sections on the backend
        sentences = [s.strip() for s in raw.replace('\n', ' ').split('.') if s.strip()]
        third = max(1, len(sentences) // 3)

        conditions = '. '.join(sentences[:third]) + '.'
        what_to_do = '. '.join(sentences[third:third*2]) + '.'
        see_doctor = '. '.join(sentences[third*2:]) + '.'

        reply = f"🔍 Possible Conditions:\n{conditions}\n\n💊 What To Do:\n{what_to_do}\n\n🏥 See a Doctor If:\n{see_doctor}"
        reply += ml_result

    except requests.exceptions.Timeout:
        reply = "⏳ AI is taking too long. Please try again." + ml_result
    except Exception:
        reply = f"⚠️ Ollama is not running. Start it with: ollama run {OLLAMA_MODEL}"

    # Task 22.3: append follow-up question if symptoms detected
    detected = detect_symptoms(message, all_symptoms)
    if detected:
        # find the most relevant follow-up from FOLLOWUP_MAP
        followup = None
        for sym in detected:
            key = sym.replace("_", " ").lower()
            if key in FOLLOWUP_MAP:
                followup = FOLLOWUP_MAP[key]
                break
            if sym.lower() in FOLLOWUP_MAP:
                followup = FOLLOWUP_MAP[sym.lower()]
                break
        if followup:
            reply += f"\n\n❓ Follow-up: {followup}"

    # Task 21.1: persist both messages in chat_messages
    try:
        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("INSERT INTO chat_messages (email, role, message) VALUES (?, ?, ?)", (session["user"], "user", message))
        cur.execute("INSERT INTO chat_messages (email, role, message) VALUES (?, ?, ?)", (session["user"], "assistant", reply))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[chat] DB insert failed: {e}", file=sys.stderr)

    return jsonify({"reply": reply})


@app.route("/intro")
def intro():
    if "user" not in session:
        return redirect("/login")
    return render_template("intro.html")


@app.route("/intro2")
def intro2():
    if "user" not in session:
        return redirect("/login")
    return render_template("intro2.html")


@app.route("/contact")
def contact():
    if "user" not in session:
        return redirect("/login")
    return render_template("contact.html")


@app.route("/send_feedback", methods=["POST"])
def send_feedback():
    name = request.form.get("name", "").strip()
    message = request.form.get("message", "").strip()
    if not name or not message:
        return jsonify({"success": False, "error": "Name and message are required."})

    entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {name}: {message}\n"

    with open("feedback.txt", "a", encoding="utf-8") as f:
        f.write(entry)

    return jsonify({"success": True})


@app.route("/profile", methods=["GET", "POST"])
def profile():
    if "user" not in session:
        return redirect("/login")
    email = session["user"]

    if request.method == "POST":
        current_pw = request.form.get("current_password", "")
        new_pw = request.form.get("new_password", "")
        confirm_pw = request.form.get("confirm_password", "")

        if len(new_pw) < 8:
            return render_template("profile.html", **_profile_data(email), error="New password must be at least 8 characters.")
        if new_pw != confirm_pw:
            return render_template("profile.html", **_profile_data(email), error="New passwords do not match.")

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE email=?", (email,))
        row = cur.fetchone()
        if not row or not check_password_hash(row[0], current_pw):
            conn.close()
            return render_template("profile.html", **_profile_data(email), error="Current password is incorrect.")
        cur.execute("UPDATE users SET password=? WHERE email=?", (generate_password_hash(new_pw), email))
        conn.commit()
        conn.close()
        return render_template("profile.html", **_profile_data(email), success="Password updated successfully.")

    return render_template("profile.html", **_profile_data(email))


def _profile_data(email):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    # created_at may not exist on older DBs — fall back gracefully
    try:
        cur.execute("SELECT email, created_at FROM users WHERE email=?", (email,))
        user_row = cur.fetchone()
        created_at = user_row[1] if user_row else "N/A"
    except Exception:
        cur.execute("SELECT email FROM users WHERE email=?", (email,))
        user_row = cur.fetchone()
        created_at = "N/A"
    cur.execute("SELECT COUNT(*) FROM predictions WHERE email=?", (email,))
    total_predictions = cur.fetchone()[0]
    try:
        cur.execute("SELECT COUNT(*) FROM predictions WHERE email=? AND bookmarked=1", (email,))
        bookmarked_predictions = cur.fetchone()[0]
    except Exception:
        bookmarked_predictions = 0
    conn.close()
    return {
        "username": email,
        "email": user_row[0] if user_row else email,
        "created_at": created_at,
        "total_predictions": total_predictions,
        "bookmarked_predictions": bookmarked_predictions,
    }


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/login")

    # real prediction stats from DB
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT disease, COUNT(*) as cnt FROM predictions GROUP BY disease ORDER BY cnt DESC LIMIT 5")
    top_diseases = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM predictions WHERE email=?", (session["user"],))
    user_total = cur.fetchone()[0]

    # Top symptoms for current user
    cur.execute("SELECT symptoms FROM predictions WHERE email=?", (session["user"],))
    symptom_rows = cur.fetchall()
    conn.close()

    symptom_counts = {}
    for (symptoms_str,) in symptom_rows:
        if symptoms_str:
            for s in symptoms_str.split(", "):
                s = s.strip()
                if s:
                    symptom_counts[s] = symptom_counts.get(s, 0) + 1
    top_symptoms = sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    return render_template(
        "dashboard.html",
        total_symptoms=len(all_symptoms),
        total_diseases=len(data["prognosis"].unique()),
        top_diseases=top_diseases,
        user_total=user_total,
        top_symptoms=top_symptoms
    )


# ---------------------------
# TASK 20: WEEKLY SUMMARY
# ---------------------------
def _weekly_summary_data(email):
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute(
        """SELECT disease, symptoms, bookmarked FROM predictions
           WHERE email=? AND created_at >= DATE('now', '-7 days')""",
        (email,)
    )
    rows = cur.fetchall()
    conn.close()

    total_count = len(rows)
    if total_count == 0:
        return {
            "total_count": 0,
            "most_frequent_disease": None,
            "most_common_symptom": None,
            "bookmarked_count": 0,
        }

    # most frequent disease
    disease_counts = {}
    for row in rows:
        d = row[0]
        disease_counts[d] = disease_counts.get(d, 0) + 1
    most_frequent_disease = max(disease_counts, key=disease_counts.get)

    # most common symptom
    symptom_counts = {}
    for row in rows:
        if row[1]:
            for s in row[1].split(", "):
                s = s.strip()
                if s:
                    symptom_counts[s] = symptom_counts.get(s, 0) + 1
    most_common_symptom = max(symptom_counts, key=symptom_counts.get) if symptom_counts else None

    # bookmarked this week
    bookmarked_count = sum(1 for row in rows if row[2])

    return {
        "total_count": total_count,
        "most_frequent_disease": most_frequent_disease,
        "most_common_symptom": most_common_symptom,
        "bookmarked_count": bookmarked_count,
    }


@app.route("/weekly_summary")
def weekly_summary():
    if "user" not in session:
        return redirect("/login")
    data_dict = _weekly_summary_data(session["user"])
    return render_template("weekly_summary.html", username=session["user"], **data_dict)


@app.route("/api/weekly_summary")
def api_weekly_summary():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify(_weekly_summary_data(session["user"]))


# ---------------------------
# TASK 21: CHAT HISTORY
# ---------------------------
@app.route("/api/chat_history")
def api_chat_history():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    limit = request.args.get("limit", 20, type=int)
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT role, message, created_at FROM chat_messages WHERE email=? ORDER BY created_at DESC LIMIT ?",
        (session["user"], limit)
    )
    rows = cur.fetchall()
    conn.close()
    messages = [{"role": r[0], "message": r[1], "created_at": r[2]} for r in rows]
    return jsonify({"messages": messages})


@app.route("/chat_history")
def chat_history():
    if "user" not in session:
        return redirect("/login")
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute(
        "SELECT role, message, created_at FROM chat_messages WHERE email=? ORDER BY created_at ASC",
        (session["user"],)
    )
    rows = cur.fetchall()
    conn.close()
    messages = [{"role": r[0], "message": r[1], "created_at": r[2]} for r in rows]
    return render_template("chat_history.html", username=session["user"], messages=messages)


@app.route("/chat_history/clear", methods=["POST"])
def chat_history_clear():
    if "user" not in session:
        return redirect("/login")
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("DELETE FROM chat_messages WHERE email=?", (session["user"],))
    conn.commit()
    conn.close()
    return redirect("/chat_history")


# ---------------------------
# TASK 24: ADMIN PANEL
# ---------------------------
@app.route("/admin")
@admin_required
def admin():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()

    # All users with prediction count
    cur.execute("""
        SELECT u.email, u.created_at, COUNT(p.id) as pred_count
        FROM users u
        LEFT JOIN predictions p ON u.email = p.email
        GROUP BY u.email
        ORDER BY u.created_at DESC
    """)
    users_list = cur.fetchall()

    # Aggregate stats
    cur.execute("SELECT COUNT(*) FROM users")
    total_users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cur.fetchone()[0]

    cur.execute("SELECT disease, COUNT(*) as cnt FROM predictions GROUP BY disease ORDER BY cnt DESC LIMIT 5")
    top_diseases = cur.fetchall()

    conn.close()

    disease_names = sorted(disease_info.keys())

    return render_template(
        "admin.html",
        username=session["user"],
        users_list=users_list,
        total_users=total_users,
        total_predictions=total_predictions,
        top_diseases=top_diseases,
        disease_names=disease_names,
    )


@app.route("/admin/export_csv")
@admin_required
def admin_export_csv():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("SELECT email, disease, confidence, symptoms, created_at FROM predictions ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["email", "disease", "confidence", "symptoms", "date"])
    for row in rows:
        writer.writerow(row)

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions_export.csv"}
    )


@app.route("/admin/disease/<name>", methods=["GET", "POST"])
@admin_required
def admin_disease_edit(name):
    if request.method == "POST":
        updated = {
            "description": request.form.get("description", ""),
            "causes": request.form.get("causes", ""),
            "precautions": request.form.get("precautions", ""),
            "medicines": request.form.get("medicines", ""),
        }
        disease_info[name] = updated
        with open("data/disease_info.json", "w") as f:
            json.dump(disease_info, f, indent=2)
        return redirect("/admin")

    info = disease_info.get(name, {
        "description": "",
        "causes": "",
        "precautions": "",
        "medicines": "",
    })
    return render_template("admin_disease_edit.html", username=session["user"], disease_name=name, info=info)


if __name__ == "__main__":
    app.run(debug=True)
