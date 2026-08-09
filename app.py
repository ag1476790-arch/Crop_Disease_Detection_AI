from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense as KerasDense
from tensorflow.keras.utils import get_custom_objects
import tensorflow.keras as tf_keras
import keras
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import os
import sqlite3
import datetime


class DenseCompat(KerasDense):
    @classmethod
    def from_config(cls, config):
        config.pop("quantization_config", None)
        return super().from_config(config)

# Register DenseCompat globally so legacy model configs resolve properly.
get_custom_objects().update({
    "Dense": DenseCompat,
})
try:
    tf_keras.layers.Dense = DenseCompat
except Exception:
    pass

# Patch keras.layers.Dense.from_config to ignore quantization_config on legacy models.
try:
    orig_dense_from_config = keras.layers.Dense.from_config

    @classmethod
    def _patched_dense_from_config(cls, config):
        config.pop("quantization_config", None)
        return orig_dense_from_config.__func__(cls, config)

    keras.layers.Dense.from_config = _patched_dense_from_config
except Exception:
    pass

app = Flask(__name__, template_folder="template")

DB_PATH = "reports.db"
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------
# Database helpers
# --------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def query_db(query, args=(), one=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(query, args)
    rows = cursor.fetchall()
    conn.close()
    return rows[0] if one and rows else rows


def save_report(filename, predicted_class, confidence):
    created_at = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO reports (filename, predicted_class, confidence, created_at) VALUES (?, ?, ?, ?)",
        (filename, predicted_class, confidence, created_at),
    )
    conn.commit()
    conn.close()


def get_reports():
    rows = query_db(
        "SELECT id, filename, predicted_class, confidence, created_at FROM reports ORDER BY id DESC"
    )
    return [dict(row) for row in rows]


def get_report(report_id):
    row = query_db(
        "SELECT id, filename, predicted_class, confidence, created_at FROM reports WHERE id = ?", (report_id,), one=True
    )
    return dict(row) if row else None


# Initialize the database when the app starts
init_db()


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "crop_disease_model (1).keras"

# Compatibility helper for older saved models with quantization_config on Dense layers
custom_objects = {
    "Dense": DenseCompat
}

model = load_model(MODEL_PATH, custom_objects=custom_objects)


# --------------------------------------------------
# Class names - 38 PlantVillage classes
# --------------------------------------------------

class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def clean_disease_name(name):
    """
    Converts PlantVillage class name into
    a more readable name.
    """

    name = name.replace("___", " - ")
    name = name.replace("_", " ")

    return name


def get_treatment_instructions(disease_name):
    """Return treatment guidance for the detected disease."""

    normalized = disease_name.lower()

    if "healthy" in normalized:
        return "Plant appears healthy. Maintain good watering, nutrition, and sanitation."
    if "powdery mildew" in normalized:
        return "Remove infected leaves, improve air circulation, and apply a sulfur-based fungicide."
    if "early blight" in normalized or "late blight" in normalized:
        return "Remove affected foliage, avoid overhead watering, and treat with an appropriate fungicide."
    if "leaf spot" in normalized:
        return "Remove infected leaves, improve airflow, and apply a fungicide if needed."
    if "black rot" in normalized:
        return "Prune out infected areas, destroy affected material, and use a fungicide spray."
    if "rust" in normalized:
        return "Remove nearby alternate hosts, improve spacing, and apply a rust-specific fungicide."
    if "bacterial spot" in normalized:
        return "Remove infected tissue, sanitize tools, and spray with copper-based bactericide."
    if "virus" in normalized:
        return "Remove infected plants, destroy them, and control insect vectors like whiteflies."
    if "spider mites" in normalized or "mite" in normalized:
        return "Rinse leaves with water, introduce natural predators, or apply insecticidal soap."
    if "citrus greening" in normalized or "huanglongbing" in normalized:
        return "Remove infected trees and control citrus psyllid vectors with insecticide."

    return "Inspect the plant, remove infected tissue, and treat with the appropriate crop-specific spray or cultural control."


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/history")
def history():
    reports = get_reports()
    return render_template("history.html", reports=reports)


@app.route("/report/<int:report_id>")
def report(report_id):
    report = get_report(report_id)
    if not report:
        return render_template("history.html", reports=get_reports(), error="Report not found.")

    treatment = get_treatment_instructions(report["predicted_class"])
    return render_template("report.html", report=report, treatment=treatment)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect(url_for("home"))

    if "file" not in request.files:
        return render_template(
            "index.html",
            error="Please upload an image."
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "index.html",
            error="Please select an image."
        )

    try:
        filename = secure_filename(file.filename)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{timestamp}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        # Open uploaded image from saved file
        image = Image.open(save_path).convert("RGB")

        # Resize to model input size
        image = image.resize((224, 224))

        # Convert image to NumPy array
        image_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # IMPORTANT:
        # Do NOT divide by 255 here because
        # the trained model already contains:
        #
        # Rescaling(1./255)

        # Make prediction
        predictions = model.predict(image_array, verbose=0)

        # Get predicted class
        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        # Confidence
        confidence = float(predictions[0][predicted_index]) * 100

        # Human-readable name
        disease_name = clean_disease_name(predicted_class)

        save_report(filename, disease_name, confidence)
        treatment = get_treatment_instructions(disease_name)

        return render_template(
            "index.html",
            prediction=disease_name,
            confidence=round(confidence, 2),
            treatment=treatment,
            report_saved=True
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"Error processing image: {str(e)}"
        )


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":
    import os
    port=int(os.environ.get("PORT",5000))
    app.run( host="0.0.0.0", port=port)