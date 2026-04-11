AI Disease Prediction System
---------------------------

This project is a web-based application that predicts possible diseases based on
user-selected symptoms using machine learning.

----------------------------------------
HOW TO RUN THE PROJECT
----------------------------------------

1. Install Python (if not installed)

2. Install required libraries:
   Open terminal in project folder and run:
   pip install -r requirements.txt

3. Train the model:
   python model.py

   This will create a file:
   models/model.pkl

4. Run the application:
   python app.py

5. Open browser and go to:
   http://127.0.0.1:5000/

----------------------------------------
FEATURES
----------------------------------------

- User login and signup with separate pages
- Select symptoms and predict disease
- Shows prediction confidence with a progress bar
- Motivational message tailored to each disease
- Displays disease info:
    - Description
    - Causes
    - Precautions
    - Medicines
- Prediction history saved per user
- AI disclaimer warning on result page
- Chatbot to enter symptoms in text form
- Clean, modern UI with hero images

----------------------------------------
SYMPTOM EXAMPLES
----------------------------------------

Below are example symptom combinations and the diseases they may predict:

1. Common Cold
   Symptoms: continuous_sneezing, chills, fatigue, cough, high_fever,
             headache, swelled_lymph_nodes, malaise, phlegm,
             throat_irritation, redness_of_eyes, sinus_pressure,
             runny_nose, congestion, chest_pain, loss_of_smell, muscle_pain

2. Malaria
   Symptoms: chills, vomiting, high_fever, sweating, headache,
             nausea, diarrhoea, muscle_pain

3. Diabetes
   Symptoms: fatigue, weight_loss, restlessness, lethargy,
             irregular_sugar_level, blurred_and_distorted_vision,
             obesity, excessive_hunger, increased_appetite, polyuria

4. Typhoid
   Symptoms: chills, vomiting, fatigue, high_fever, headache,
             nausea, constipation, abdominal_pain, diarrhoea,
             toxic_look_(typhos), belly_pain

5. Migraine
   Symptoms: acidity, indigestion, headache, blurred_and_distorted_vision,
             excessive_hunger, stiff_neck, depression, irritability,
             visual_disturbances

6. Pneumonia
   Symptoms: chills, fatigue, cough, high_fever, breathlessness,
             sweating, malaise, phlegm, chest_pain, fast_heart_rate,
             rusty_sputum

7. Urinary Tract Infection
   Symptoms: burning_micturition, bladder_discomfort, foul_smell_of_urine,
             continuous_feel_of_urine, internal_itching

8. Hypertension
   Symptoms: headache, chest_pain, dizziness, loss_of_balance,
             lack_of_concentration

9. Dengue
   Symptoms: skin_rash, chills, joint_pain, vomiting, fatigue,
             high_fever, headache, nausea, loss_of_appetite,
             pain_behind_the_eyes, back_pain, malaise, muscle_pain,
             red_spots_over_body

10. Fungal Infection
    Symptoms: itching, skin_rash, nodal_skin_eruptions,
              dischromic_patches

11. Allergy
    Symptoms: continuous_sneezing, shivering, chills, watering_from_eyes

12. GERD
    Symptoms: stomach_pain, acidity, ulcers_on_tongue, vomiting,
              cough, chest_pain

13. Chicken Pox
    Symptoms: itching, skin_rash, fatigue, lethargy, high_fever,
              headache, loss_of_appetite, mild_fever, swelled_lymph_nodes,
              malaise, red_spots_over_body, phlegm, throat_irritation,
              patches_in_throat

14. Arthritis
    Symptoms: muscle_weakness, stiff_neck, swelling_joints,
              movement_stiffness, painful_walking

15. Heart Attack
    Symptoms: vomiting, breathlessness, sweating, chest_pain

----------------------------------------
PROJECT STRUCTURE
----------------------------------------

app.py              -> main backend (Flask)
model.py            -> trains ML models
utils.py            -> prediction logic
requirements.txt    -> dependencies

data/
    Training.csv        -> training dataset
    Testing.csv         -> testing dataset
    disease_info.json   -> disease descriptions, causes, precautions, medicines

models/
    model.pkl           -> trained ML model

templates/
    index.html          -> symptom selection page
    result.html         -> prediction result page
    login.html          -> login page
    signup.html         -> create account page
    history.html        -> prediction history page
    dashboard.html      -> dashboard

static/
    style.css           -> all styles

----------------------------------------
IMPORTANT NOTES
----------------------------------------

- Always run model.py before app.py
- If model.pkl is missing, prediction will not work
- disease_info.json is used to show extra details about each disease
- Prediction history is stored in users.db (SQLite)

----------------------------------------
FOR EDUCATIONAL USE ONLY
----------------------------------------

This system is for learning and educational purposes only.
It should NOT be used as a replacement for professional medical diagnosis.
Always consult a certified doctor before taking any medication.

----------------------------------------
