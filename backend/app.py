from flask import Flask, redirect, request, jsonify, render_template, session, url_for
from flask_cors import CORS
from prediction import build_disease_prediction_system, predict_top_diseases
import csv
import hashlib
import os
from datetime import datetime
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

USER_FIELDS = ["username", "password", "name", "email", "phone", "location", "image"]


def data_path(filename):
    return os.path.join(BASE_DIR, filename)


def ensure_csv(filename, fieldnames):
    path = data_path(filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    return path


# --------------------------------------------------
# LOAD ML MODEL ONCE
# --------------------------------------------------
try:
    model, encoder, features = build_disease_prediction_system(data_path("training_data.csv"))
    print("ML model loaded successfully")
except Exception as exc:
    print(f"Error loading ML model: {exc}")
    model, encoder, features = None, None, None


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def normalize(username):
    return (username or "").strip().lower()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_user_profile(username):
    path = data_path("user_profiles.csv")
    if not os.path.exists(path):
        return None

    username = normalize(username)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if normalize(row.get("username")) == username:
                return row
    return None


def load_history(username):
    path = data_path("history_search.csv")
    if not os.path.exists(path):
        return []

    username = normalize(username)
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)
                if normalize(row.get("username")) == username]


def load_saved_diseases(username):
    path = data_path("saved_diseases.csv")
    if not os.path.exists(path):
        return []

    username = normalize(username)
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)
                if normalize(row.get("username")) == username]


# --------------------------------------------------
# UI ROUTES
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect("/")
    history = load_history(session["user"])
    return render_template("dashboard.html", recent_history=history[-3:])


@app.route("/history")
def history_page():
    if "user" not in session:
        return redirect("/")
    return render_template("history.html", history=list(reversed(load_history(session["user"]))))


@app.route("/profile")
def profile():
    if "user" not in session:
        return redirect("/")
    username = normalize(session["user"])
    user = load_user_profile(username)
    if not user:
        return "Profile not found", 404
    return render_template(
        "profile.html",
        user=user,
        history=load_history(username),
        saved_diseases=load_saved_diseases(username),
    )


@app.route("/settings")
def settings():
    if "user" not in session:
        return redirect("/")
    return render_template("settings.html", user=load_user_profile(session["user"]))


@app.route("/settings/update", methods=["POST"])
def update_settings():
    if "user" not in session:
        return redirect("/")

    username = normalize(session["user"])
    path = ensure_csv("user_profiles.csv", USER_FIELDS)
    name = request.form.get("name", "")
    email = request.form.get("email", "")
    phone = request.form.get("phone", "")
    location = request.form.get("location", "")
    image_path = None

    if "image" in request.files:
        file = request.files["image"]
        if file and allowed_file(file.filename):
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            filename = secure_filename(f"{username}_{file.filename}")
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            image_path = "/static/uploads/" + filename

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if normalize(row.get("username")) == username:
            row.update({"name": name, "email": email, "phone": phone, "location": location})
            if image_path:
                row["image"] = image_path

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=USER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return redirect("/profile")


@app.route("/help", methods=["GET", "POST"])
def help_page():
    if "user" not in session:
        return redirect("/")

    if request.method == "POST":
        path = data_path("help.csv")
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["name", "email", "subject", "created_at"])
            writer.writerow([
                request.form.get("name", ""),
                request.form.get("email", ""),
                request.form.get("subject", ""),
                datetime.now().isoformat(timespec="seconds"),
            ])
        return redirect("/help")

    return render_template("help.html")


@app.route("/contact", methods=["POST"])
def contact():
    path = data_path("contact_requests.csv")
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["name", "email", "message", "created_at"])
        writer.writerow([
            request.form.get("name", ""),
            request.form.get("email", ""),
            request.form.get("message", "").replace("\n", " ").replace("\r", " "),
            datetime.now().isoformat(timespec="seconds"),
        ])
    return redirect("/help")


@app.route("/about")
def about():
    if "user" not in session:
        return redirect("/")
    return render_template("about.html")


# --------------------------------------------------
# AUTH ROUTES
# --------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = normalize(request.form.get("username"))
        if not username:
            return "Username is required", 400
        if load_user_profile(username):
            return "Username already exists", 409

        path = ensure_csv("user_profiles.csv", USER_FIELDS)
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=USER_FIELDS).writerow({
                "username": username,
                "password": hash_password(request.form.get("password", "")),
                "name": request.form.get("name", ""),
                "email": request.form.get("email", ""),
                "phone": request.form.get("phone", ""),
                "location": "",
                "image": "",
            })
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("index.html")

    data = request.get_json(silent=True) or request.form
    username = normalize(data.get("username", ""))
    password = data.get("password", "")

    user = load_user_profile(username)
    if user and user.get("password") == hash_password(password):
        session["user"] = username
        return jsonify({"message": "Login successful"})

    return jsonify({"error": "Invalid credentials"}), 401


@app.route("/logout", methods=["POST"])
def logout():
    session.pop("user", None)
    return jsonify({"message": "Logged out"})


# --------------------------------------------------
# DATA ROUTES
# --------------------------------------------------
@app.route("/symptoms")
def get_symptoms():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"symptoms": list(features) if features is not None else []})


@app.route("/predict", methods=["POST"])
def predict():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    if model is None or encoder is None or features is None:
        return jsonify({"error": "ML model is unavailable"}), 503

    data = request.get_json(silent=True) or {}
    symptoms = data.get("symptoms", [])
    raw = predict_top_diseases(model, encoder, features, symptoms, top_n=3)
    return jsonify({"results": [{"name": d, "score": round(p * 100, 2)} for d, p in raw]})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=int(os.environ.get("PORT", 5000)))
