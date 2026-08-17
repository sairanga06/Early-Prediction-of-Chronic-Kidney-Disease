from flask import Flask, render_template, request, jsonify
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)


# =========================================================
# GLOBAL VARIABLES
# =========================================================

feature_encoders = {}
numeric_fill_values = {}
target_encoder = None
training_columns = []

model = None
model_accuracy = None
dataset_name = None


MISSING_VALUES = [
    "?",
    "",
    "nan",
    "NaN",
    "None",
    "null",
    "NULL"
]


# =========================================================
# COLUMN NAME MAPPING
# Same mapping used in the Tkinter application
# =========================================================

COLUMN_MAPPING = {
    "blood_pressure": "bp",
    "specific_gravity": "sg",
    "albumin": "al",
    "sugar": "su",
    "red_blood_cells": "rbc",
    "pus_cell": "pc",
    "pus_cell_clumps": "pcc",
    "bacteria": "ba",
    "blood glucose random": "bgr",
    "blood_urea": "bu",
    "serum_creatinine": "sc",
    "sodium": "sod",
    "potassium": "pot",
    "hemoglobin": "hemo",
    "packed_cell_volume": "pcv",
    "white_blood_cell_count": "wc",
    "red_blood_cell_count": "rc",
    "hypertension": "htn",
    "diabetesmellitus": "dm",
    "coronary_artery_disease": "cad",
    "appetite": "appet",
    "pedal_edema": "pe",
    "anemia": "ane",
    "class": "classification"
}


# =========================================================
# CLEAN MISSING VALUES
# =========================================================

def clean_missing_values(data):
    data = data.copy()

    for column in data.columns:
        data[column] = data[column].replace(
            MISSING_VALUES,
            pd.NA
        )

        if data[column].dtype == "object":
            data[column] = (
                data[column]
                .astype("string")
                .str.strip()
            )

    return data


# =========================================================
# FIND CKD DATASET
# =========================================================

def find_dataset():
    project_root = Path(__file__).resolve().parent.parent
    dataset_folder = project_root / "Dataset"

    if not dataset_folder.exists():
        raise FileNotFoundError(
            f"Dataset folder was not found:\n{dataset_folder}"
        )

    csv_files = list(dataset_folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV dataset was found inside the Dataset folder."
        )

    # Prefer CKD.csv if it exists
    for file in csv_files:
        if file.name.lower() == "ckd.csv":
            return file

    # Otherwise use the first CSV file
    return csv_files[0]


# =========================================================
# TRAIN MODEL
# Uses the same preprocessing logic as Tkinter application
# =========================================================

def train_model():
    global feature_encoders
    global numeric_fill_values
    global target_encoder
    global training_columns
    global model
    global model_accuracy
    global dataset_name

    dataset_path = find_dataset()
    dataset_name = dataset_path.name

    print("\n==========================================")
    print("CKD FLASK APPLICATION")
    print("==========================================")
    print(f"Dataset: {dataset_name}")

    # -----------------------------------------------------
    # LOAD DATASET
    # -----------------------------------------------------

    df = pd.read_csv(dataset_path)

    if df.empty:
        raise ValueError("The CKD dataset is empty.")

    print(f"Dataset size: {len(df)} rows")
    print(f"Number of columns: {len(df.columns)}")

    # -----------------------------------------------------
    # RENAME COLUMNS
    # -----------------------------------------------------

    data = df.copy()

    data.rename(
        columns=COLUMN_MAPPING,
        inplace=True
    )

    # -----------------------------------------------------
    # CLEAN MISSING VALUES
    # -----------------------------------------------------

    data = clean_missing_values(data)

    # -----------------------------------------------------
    # REMOVE ID COLUMN
    # -----------------------------------------------------

    if "id" in data.columns:
        data.drop(
            columns=["id"],
            inplace=True
        )

    if len(data.columns) < 2:
        raise ValueError(
            "Dataset must contain features and a target column."
        )

    # -----------------------------------------------------
    # TARGET = LAST COLUMN
    # Same logic as existing Tkinter application
    # -----------------------------------------------------

    target_column = data.columns[-1]

    X = data.drop(
        columns=[target_column]
    ).copy()

    y = data[target_column].copy()

    y = (
        y.astype("string")
        .str.strip()
    )

    # -----------------------------------------------------
    # SAVE TRAINING COLUMN ORDER
    # -----------------------------------------------------

    training_columns = list(X.columns)

    feature_encoders = {}
    numeric_fill_values = {}

    # -----------------------------------------------------
    # PREPROCESS FEATURES
    # -----------------------------------------------------

    for column in X.columns:

        numeric_values = pd.to_numeric(
            X[column],
            errors="coerce"
        )

        non_missing = X[column].notna().sum()

        if non_missing > 0:
            numeric_ratio = (
                numeric_values.notna().sum()
                / non_missing
            )
        else:
            numeric_ratio = 0

        # ---------------------------------------------
        # NUMERIC COLUMN
        # ---------------------------------------------

        if numeric_ratio >= 0.8:

            X[column] = numeric_values

            median_value = X[column].median()

            if pd.isna(median_value):
                median_value = 0.0

            numeric_fill_values[column] = float(
                median_value
            )

            X[column] = X[column].fillna(
                median_value
            )

        # ---------------------------------------------
        # CATEGORICAL COLUMN
        # ---------------------------------------------

        else:

            mode_values = X[column].mode(
                dropna=True
            )

            if mode_values.empty:
                fill_value = "unknown"
            else:
                fill_value = str(
                    mode_values.iloc[0]
                ).strip()

            X[column] = X[column].fillna(
                fill_value
            )

            X[column] = (
                X[column]
                .astype(str)
                .str.strip()
            )

            encoder = LabelEncoder()

            X[column] = encoder.fit_transform(
                X[column]
            )

            feature_encoders[column] = encoder

    # -----------------------------------------------------
    # FINAL FEATURE CHECK
    # -----------------------------------------------------

    X = X.fillna(0)

    if X.isna().sum().sum() != 0:
        raise ValueError(
            "Missing values still exist in feature data."
        )

    # -----------------------------------------------------
    # ENCODE TARGET
    # -----------------------------------------------------

    y = y.fillna(
        y.mode().iloc[0]
        if not y.mode().empty
        else "unknown"
    )

    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(
        y.astype(str).str.strip()
    )

    if len(target_encoder.classes_) < 2:
        raise ValueError(
            "The target column must contain at least two classes."
        )

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # Same settings as Tkinter application
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=21,
        stratify=y
    )

    # -----------------------------------------------------
    # LOGISTIC REGRESSION
    # Same model as Tkinter application
    # -----------------------------------------------------

    model = LogisticRegression(
        solver="liblinear",
        max_iter=2000
    )

    model.fit(
        X_train,
        y_train
    )

    # -----------------------------------------------------
    # MODEL ACCURACY
    # -----------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    model_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # -----------------------------------------------------
    # DISPLAY INFORMATION IN TERMINAL
    # -----------------------------------------------------

    print("------------------------------------------")
    print(f"Training size : {len(X_train)}")
    print(f"Testing size  : {len(X_test)}")
    print(f"Features      : {len(training_columns)}")
    print(
        f"Accuracy      : {model_accuracy * 100:.2f}%"
    )
    print(
        f"Target classes: {list(target_encoder.classes_)}"
    )
    print("------------------------------------------")
    print("Model trained successfully.")
    print("==========================================\n")


# =========================================================
# PREPARE USER INPUT FOR PREDICTION
# =========================================================

def prepare_prediction_data(input_data):

    if model is None:
        raise ValueError(
            "The model has not been trained."
        )

    if not training_columns:
        raise ValueError(
            "Training columns are not available."
        )

    data = input_data.copy()

    # -----------------------------------------------------
    # SAME COLUMN MAPPING
    # -----------------------------------------------------

    data.rename(
        columns=COLUMN_MAPPING,
        inplace=True
    )

    # -----------------------------------------------------
    # CLEAN VALUES
    # -----------------------------------------------------

    data = clean_missing_values(data)

    # -----------------------------------------------------
    # REMOVE ID
    # -----------------------------------------------------

    if "id" in data.columns:
        data.drop(
            columns=["id"],
            inplace=True
        )

    # -----------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # -----------------------------------------------------

    missing_columns = [
        column
        for column in training_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------
    # KEEP SAME COLUMN ORDER AS TRAINING
    # -----------------------------------------------------

    data = data[
        training_columns
    ].copy()

    # -----------------------------------------------------
    # APPLY SAME ENCODING / NUMERIC PROCESSING
    # -----------------------------------------------------

    for column in training_columns:

        # ---------------------------------------------
        # CATEGORICAL COLUMN
        # ---------------------------------------------

        if column in feature_encoders:

            encoder = feature_encoders[column]

            values = (
                data[column]
                .fillna(
                    encoder.classes_[0]
                )
                .astype(str)
                .str.strip()
            )

            unknown_values = sorted(
                set(values)
                - set(encoder.classes_)
            )

            if unknown_values:
                raise ValueError(
                    f"Unknown value(s) in '{column}': "
                    f"{unknown_values}"
                )

            data[column] = encoder.transform(
                values
            )

        # ---------------------------------------------
        # NUMERIC COLUMN
        # ---------------------------------------------

        else:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            fill_value = numeric_fill_values.get(
                column,
                0.0
            )

            data[column] = data[column].fillna(
                fill_value
            )

    # -----------------------------------------------------
    # FINAL CHECK
    # -----------------------------------------------------

    data = data.fillna(0)

    if data.isna().sum().sum() != 0:
        raise ValueError(
            "Prediction data still contains missing values."
        )

    return data


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        accuracy=(
            f"{model_accuracy * 100:.2f}%"
            if model_accuracy is not None
            else "N/A"
        ),
        dataset=dataset_name
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "success",
        "message": "CKD Flask application is running.",
        "model_loaded": model is not None,
        "dataset": dataset_name,
        "accuracy": (
            round(model_accuracy * 100, 2)
            if model_accuracy is not None
            else None
        )
    })


# =========================================================
# PREDICTION API
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -------------------------------------------------
        # CHECK REQUEST
        # -------------------------------------------------

        if not request.is_json:
            return jsonify({
                "status": "error",
                "message": "Request must contain JSON data."
            }), 400

        input_json = request.get_json()

        if not input_json:
            return jsonify({
                "status": "error",
                "message": "No prediction data was received."
            }), 400

        # -------------------------------------------------
        # CREATE DATAFRAME
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [input_json]
        )

        # -------------------------------------------------
        # PREPARE DATA
        # -------------------------------------------------

        prediction_data = prepare_prediction_data(
            input_data
        )

        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        prediction = model.predict(
            prediction_data
        )

        prediction_number = int(
            prediction[0]
        )

        # -------------------------------------------------
        # CONVERT BACK TO ORIGINAL CLASS
        # -------------------------------------------------

        result = target_encoder.inverse_transform(
            [prediction_number]
        )[0]

        return jsonify({
            "status": "success",
            "prediction": str(result),
            "accuracy": (
                round(model_accuracy * 100, 2)
                if model_accuracy is not None
                else None
            )
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    try:

        # Train the model before starting Flask
        train_model()

        print("Starting Flask server...")
        print("Open: http://127.0.0.1:5000")
        print("Health check: http://127.0.0.1:5000/health")

        app.run(
            debug=True
        )

    except Exception as e:

        print("\n==========================================")
        print("APPLICATION STARTUP ERROR")
        print("==========================================")
        print(e)
        print("==========================================\n")