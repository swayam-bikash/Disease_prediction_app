import pandas as pd
import pickle
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


print("loading dataset...")

# load training data
data = pd.read_csv("data/Training.csv")

# handle missing values
data = data.fillna(0)

# split into features and label — drop prognosis and any unnamed trailing columns
X = data.drop(columns=["prognosis"] + [c for c in data.columns if c.startswith("Unnamed")])
y = data["prognosis"]

print("splitting data into train and test...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


print("\ndefining models...\n")

# defining models one by one (more readable)
rf_model = RandomForestClassifier(n_estimators=100)
nb_model = GaussianNB()
gb_model = GradientBoostingClassifier(n_estimators=100)

# store in dictionary
models = {
    "Random Forest": rf_model,
    "Naive Bayes": nb_model,
    "Gradient Boosting": gb_model
}


trained_models = {}
results = {}


print("training models...\n")

# training each model
for name in models:

    print("training:", name)

    current_model = models[name]

    current_model.fit(X_train, y_train)

    trained_models[name] = current_model


print("\nevaluating models...\n")

# evaluating models
for name in trained_models:

    model = trained_models[name]

    y_pred = model.predict(X_test)

    # calculating metrics separately (more human style)
    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        average='weighted',
        zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    # storing results
    results[name] = {
        "model": model,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm
    }

    print("model:", name)
    print("accuracy:", round(acc, 4))
    print("precision:", round(prec, 4))
    print("recall:", round(rec, 4))
    print("confusion matrix:\n", cm)
    print("-" * 50)


print("\nselecting best model based on accuracy...\n")

best_model = None
best_model_name = ""
best_accuracy = 0

# compare all models
for name in results:

    current_accuracy = results[name]["accuracy"]

    print("checking:", name, "accuracy:", round(current_accuracy, 4))

    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        best_model = results[name]["model"]
        best_model_name = name


print("\nbest model selected:", best_model_name)
print("best accuracy:", round(best_accuracy, 4))


print("\nsaving model...\n")

# make sure folder exists
if not os.path.exists("models"):
    os.makedirs("models")

# save model
with open("models/model.pkl", "wb") as f:
    pickle.dump(best_model, f)


print("model saved successfully at models/model.pkl")
print("\ntraining process completed 👍")
