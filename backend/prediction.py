import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# =========================
# MODEL TRAINING
# =========================
def build_disease_prediction_system(csv_path):
    print(f"\n--- Loading Data from {csv_path} ---")

    df = pd.read_csv(csv_path)

    # Separate features & target
    X = df.drop("disease", axis=1)
    y = df["disease"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # -------------------------
    # Decision Tree
    # -------------------------
    dt_model = DecisionTreeClassifier(
        max_depth=25,
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
    print(f"Decision Tree Accuracy : {dt_acc*100:.2f}%")

    # -------------------------
    # Random Forest
    # -------------------------
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=30,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
    print(f"Random Forest Accuracy : {rf_acc*100:.2f}%")

    # -------------------------
    # Auto-select best model
    # -------------------------
    if rf_acc >= dt_acc:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = dt_model
        best_name = "Decision Tree"

    print(f"\nBest Model Selected: {best_name}")

    print("\n--- Classification Report ---")
    print(classification_report(
        y_test,
        best_model.predict(X_test),
        target_names=label_encoder.classes_
    ))

    return best_model, label_encoder, X.columns


# =========================
# TOP-N PREDICTION
# =========================
def predict_top_diseases(model, encoder, feature_names, symptoms_present, top_n=3):

    if model is None or encoder is None:
        raise ValueError("Model or encoder not loaded")

    feature_index = {f: i for i, f in enumerate(feature_names)}
    input_vector = np.zeros(len(feature_names))

    for symptom in symptoms_present:
        if symptom in feature_index:
            input_vector[feature_index[symptom]] = 2
        else:
            print(f"Warning: Symptom '{symptom}' not recognized")

    probs = model.predict_proba([input_vector])[0]
    top_indices = np.argsort(probs)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        disease = encoder.inverse_transform([idx])[0]
        score = probs[idx]
        results.append((disease, score))

    return results
