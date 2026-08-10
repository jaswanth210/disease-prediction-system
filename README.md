# Disease Prediction System Using Machine Learning

A Flask-based machine learning application that accepts user-selected symptoms and predicts the most likely diseases using classification models.

## Features

- Symptom-based disease prediction
- Random Forest and Decision Tree model comparison
- Automatic selection of the better-performing model
- Top-3 prediction results with probability scores
- Flask web interface
- User registration and login
- Prediction history and saved-disease pages
- Profile and settings pages

## Technology Stack

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML/CSS/JavaScript

## Project Structure

```text
backend/
├── app.py
├── prediction.py
├── training_data.csv
├── requirements.txt
├── .env.example
├── templates/
└── static/
```

## Run Locally

1. Open a terminal in the `backend` folder.
2. Create a virtual environment:

```bash
python -m venv venv
```

3. Activate it.

Windows:

```bash
venv\\Scripts\\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Set a secret key. You can copy `.env.example` to `.env` and set `SECRET_KEY`.

6. Start the application:

```bash
python app.py
```

7. Open `http://localhost:5000` in your browser.

## Important Note

This project is an academic machine-learning demonstration and should not be used as a medical diagnosis or as a replacement for a qualified healthcare professional.

## Author

K. Jaswanth Venkat — MCA 2026

## Deployment

The repository includes `vercel.json` and `api/index.py` for Vercel deployment.

For a production application, replace CSV-based user/runtime storage with a persistent database because serverless filesystems are not designed for permanent application data.
