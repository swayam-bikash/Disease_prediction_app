================================================================
        AI DISEASE PREDICTION SYSTEM — MediBot AI Clinic
================================================================

PROJECT OVERVIEW
----------------
An AI-powered full-stack web application that predicts diseases
based on user-selected symptoms using machine learning models,
combined with an AI chatbot (Dr. Atlas) powered by Ollama (tinyllama).

Users select symptoms, rate their severity, and receive a
prescription-style result with diagnosis, confidence score,
top 3 differential diagnoses, disease info, precautions,
suggested medicines, and nearby hospital finder.

TECH STACK
----------
Backend  : Python 3, Flask
ML       : scikit-learn (Random Forest, Naive Bayes, Gradient Boosting)
AI Chat  : Ollama — tinyllama (local LLM, runs offline)
Database : SQLite (users.db — auto-created)
Frontend : Jinja2 templates, vanilla JavaScript, Chart.js
PDF      : jsPDF + html2canvas (client-side)
Auth     : Werkzeug password hashing + Flask sessions
Config   : python-dotenv (.env file)

KEY FEATURES
------------
Authentication
  - Signup / Login with bcrypt-hashed passwords
  - 1-day persistent session
  - Password change from profile page
  - Admin role via ADMIN_EMAILS env variable

Symptom Selection
  - Full symptom grid with checkboxes
  - Fuzzy search autocomplete (Levenshtein distance)
  - Keyboard navigation (arrow keys + enter)
  - Per-symptom severity sliders (Mild / Moderate / Severe)
  - Built-in BMI calculator widget

Disease Prediction
  - Binary input vector → ML model inference
  - Top 3 differential diagnoses with confidence bars
  - Severity badges on result card
  - Disease comparison panel (side-by-side diff)
  - Animated confidence bar (color-coded: green/amber/red)

Result Page
  - Prescription-style card (MediBot AI Clinic)
  - Disease description, causes, precautions, medicines
  - Motivational message per disease
  - Download as PDF (jsPDF)
  - Save as image (html2canvas)
  - Print-ready layout
  - Bookmark prediction (toggle ⭐)
  - Shareable link via UUID token (no login required to view)
  - Nearby hospital finder (Google Maps embed or fallback link)

AI Chatbot — Dr. Atlas
  - Floating chat widget on symptoms page
  - Powered by Ollama (tinyllama, runs locally)
  - Greeting detection, English word validation
  - ML symptom extraction from chat message
  - Structured reply: Possible Conditions / What To Do / See a Doctor If
  - Follow-up questions based on detected symptoms
  - Conversation context passed between messages
  - Chat history persisted in SQLite
  - Maximize / minimize chat window
  - History loaded on chat open

Prediction History
  - Full history table per user
  - Bookmark filter (⭐ Bookmarked Only)
  - Line chart — predictions over last 30 days (Chart.js)
  - Delete single record or clear all

Weekly Health Summary
  - Predictions this week
  - Most frequent disease this week
  - Most common symptom this week
  - Bookmarked count this week
  - JSON API endpoint: /api/weekly_summary

Dashboard
  - Total symptoms, diseases, models trained, user predictions
  - Model accuracy comparison bar chart (RF / NB / GB)
  - Most predicted diseases doughnut chart (global)
  - Most common symptoms horizontal bar chart (per user)

Admin Panel (admin-only)
  - Platform-wide stats: total users, total predictions, top diseases
  - Full user list with registration date and prediction count
  - Export all predictions as CSV
  - Disease info editor (edit description, causes, precautions, medicines)
  - Changes saved live to data/disease_info.json

Other
  - Dark / light theme toggle (persisted in localStorage)
  - Animated intro videos on login and before symptoms page
  - Contact page with feedback modal (saved to feedback.txt)
  - Shareable read-only prescription page (public, no login needed)
  - Chat history page with clear option

SETUP INSTRUCTIONS
------------------
1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO

2. Create a virtual environment and install dependencies:
   python -m venv venv
   venv\Scripts\activate        (Windows)
   source venv/bin/activate     (Mac/Linux)
   pip install -r requirements.txt

3. Set up environment variables:
   Copy .env.example to .env and fill in your values:
     SECRET_KEY       — long random string (run: python -c "import secrets; print(secrets.token_hex(32))")
     ADMIN_EMAIL      — your admin email
     ADMIN_PASSWORD   — your admin password
     ADMIN_EMAILS     — comma-separated admin emails
     GOOGLE_MAPS_API_KEY — optional, for hospital finder embed

4. Train the ML model:
   python model.py
   (This creates models/model.pkl)

5. Install and start Ollama (for AI chat):
   Download from https://ollama.com
   Run: ollama pull tinyllama

6. Run the app:
   python app.py
   Open http://localhost:5000

PROJECT STRUCTURE
-----------------
app.py                      — Main Flask application (all routes)
model.py                    — ML model training script
utils.py                    — Prediction utility (input vector + inference)
requirements.txt            — Python dependencies
.env                        — Secret config (NOT committed to git)
.env.example                — Template for .env
.gitignore                  — Excludes .env, users.db, model.pkl, etc.

data/
  Training.csv              — Symptom-disease training dataset
  Testing.csv               — Symptom-disease testing dataset
  disease_info.json         — Disease metadata (editable via admin panel)

models/
  model.pkl                 — Trained ML model (generated by model.py)

static/
  style.css                 — Main stylesheet (dark/light theme)
  a1.mp4                    — Post-login intro video
  a2.mp4                    — Pre-symptoms transition video
  a3.mp4                    — (Available asset)
  admin.jpg                 — Admin panel background
  history.png               — History page background
  main.jpg / login.jpg      — Home / login backgrounds
  symptoms.jpg / chat.png   — Symptoms / chat backgrounds
  contact.jpg               — Contact page background

templates/
  index.html                — Home / landing page
  intro.html                — Post-login intro video page
  intro2.html               — Pre-symptoms transition video page
  symptoms.html             — Symptom selection + Dr. Atlas chatbot
  result.html               — Prescription result page
  share_result.html         — Public shareable result (no login)
  history.html              — Prediction history table + chart
  history_intro.html        — History intro transition page
  dashboard.html            — Analytics dashboard
  weekly_summary.html       — Weekly health summary
  chat_history.html         — Full chat history page
  profile.html              — User profile + password change
  login.html                — Login page
  signup.html               — Signup page
  contact.html              — Contact / feedback page
  admin.html                — Admin panel
  admin_disease_edit.html   — Disease info editor

ENVIRONMENT VARIABLES
---------------------
SECRET_KEY          — Flask session secret key (required)
ADMIN_EMAIL         — Admin account email for seeding (required)
ADMIN_PASSWORD      — Admin account password for seeding (required)
ADMIN_EMAILS        — Comma-separated list of admin emails (required)
GOOGLE_MAPS_API_KEY — Google Maps Embed API key (optional)
OLLAMA_EXE          — Custom path to ollama.exe (optional, Windows only)

NOTES
-----
- users.db, feedback.txt, and models/model.pkl are auto-generated
  at runtime and are excluded from git via .gitignore
- The AI chat requires Ollama running locally on port 11434
- The app auto-attempts to start Ollama on Windows if OLLAMA_EXE is set
- This project is for educational purposes only — not a substitute
  for professional medical advice

AUTHOR
------
Swayam Bikash
AI + ML Final Year Project — MediBot AI Clinic
================================================================
