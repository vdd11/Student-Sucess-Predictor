from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_success_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "app_features.joblib"
STATS_PATH = BASE_DIR / "models" / "dashboard_stats.joblib"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.csv"


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file missing. Run python train_model.py first.")
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    stats = joblib.load(STATS_PATH) if STATS_PATH.exists() else {}
    metrics = pd.read_csv(METRICS_PATH).to_dict("records") if METRICS_PATH.exists() else []
    return model, features, stats, metrics


@app.route("/")
def home():
    _, _, stats, _ = load_artifacts()
    return render_template("index.html", stats=stats)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model, features, _, _ = load_artifacts()
    result = None
    if request.method == "POST":
        form = request.form
        input_data = {
            "age": int(form.get("age")),
            "studytime": int(form.get("studytime")),
            "failures": int(form.get("failures")),
            "absences": int(form.get("absences")),
            "G1": int(form.get("G1")),
            "G2": int(form.get("G2")),
            "schoolsup": form.get("schoolsup"),
            "famsup": form.get("famsup"),
            "paid": form.get("paid"),
            "internet": form.get("internet"),
            "higher": form.get("higher"),
            "romantic": form.get("romantic"),
        }
        X_new = pd.DataFrame([input_data], columns=features)
        prediction = int(model.predict(X_new)[0])
        probability = float(model.predict_proba(X_new)[0][1])
        result = {
            "label": "Likely to Pass" if prediction == 1 else "At Risk of Not Passing",
            "probability": round(probability * 100, 1),
            "risk": round((1 - probability) * 100, 1),
            "prediction": prediction,
        }
    return render_template("predict.html", result=result)


@app.route("/dashboard")
def dashboard():
    _, _, stats, metrics = load_artifacts()
    return render_template("dashboard.html", stats=stats, metrics=metrics)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
