from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# =========================================================
# GLOBAL VARIABLES
# =========================================================

filename = None
df = None

X_train = None
X_test = None
y_train = None
y_test = None

model = None

knn_acc = None
nb_acc = None
lr_accuracy = None

feature_encoders = {}
numeric_fill_values = {}
target_encoder = None
training_columns = []

MISSING_VALUES = ["?", "", "nan", "NaN", "None", "null", "NULL"]


# =========================================================
# COLUMN NAME MAPPING
# Supports CKD.csv and modified_dataset/test1.csv
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
# HELPER: CLEAN VALUES
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
# UPLOAD DATASET
# =========================================================

def upload():
    global filename, df

    filename = filedialog.askopenfilename(
        title="Select Dataset",
        filetypes=[("CSV files", "*.csv")]
    )

    if not filename:
        return

    try:
        df = pd.read_csv(filename)

        if df.empty:
            messagebox.showerror(
                "Error",
                "The selected dataset is empty."
            )
            return

        pathlabel.config(text=filename)

        text.delete("1.0", END)

        text.insert(
            END,
            "Dataset loaded successfully\n\n"
        )

        text.insert(
            END,
            f"Dataset Size: {len(df)} rows\n"
        )

        text.insert(
            END,
            f"Number of Columns: {len(df.columns)}\n\n"
        )

        text.insert(
            END,
            "Columns:\n"
        )

        for column in df.columns:
            text.insert(
                END,
                f"- {column}\n"
            )

    except Exception as e:
        messagebox.showerror(
            "Dataset Error",
            f"Unable to load dataset.\n\n{e}"
        )


# =========================================================
# SPLIT DATASET + PREPROCESSING
# =========================================================

def splitdataset():
    global df
    global X_train, X_test, y_train, y_test
    global feature_encoders, numeric_fill_values
    global target_encoder, training_columns

    if df is None:
        messagebox.showwarning(
            "Warning",
            "Please upload a dataset first."
        )
        return

    try:
        data = df.copy()

        data.rename(
            columns=COLUMN_MAPPING,
            inplace=True
        )

        data = clean_missing_values(data)

        if "id" in data.columns:
            data.drop(
                columns=["id"],
                inplace=True
            )

        if len(data.columns) < 2:
            raise ValueError(
                "Dataset must contain features and a target column."
            )

        target_column = data.columns[-1]

        X = data.drop(
            columns=[target_column]
        ).copy()

        y = data[target_column].copy()

        y = (
            y.astype("string")
            .str.strip()
        )

        training_columns = list(X.columns)

        feature_encoders = {}
        numeric_fill_values = {}

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

        X = X.fillna(0)

        if X.isna().sum().sum() != 0:
            raise ValueError(
                "Missing values still exist in the feature data."
            )

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

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=21,
            stratify=y
        )

        X_test.to_csv(
            "test1.csv",
            index=False
        )

        text.delete("1.0", END)

        text.insert(
            END,
            "Dataset split successfully\n\n"
        )

        text.insert(
            END,
            f"Training Size: {len(X_train)}\n"
        )

        text.insert(
            END,
            f"Test Size: {len(X_test)}\n\n"
        )

        text.insert(
            END,
            "Features used:\n"
        )

        for column in training_columns:
            text.insert(
                END,
                f"- {column}\n"
            )

        text.insert(
            END,
            "\nMissing values handled successfully.\n"
        )

        text.insert(
            END,
            f"Target classes: {list(target_encoder.classes_)}\n"
        )

    except Exception as e:
        messagebox.showerror(
            "Split Error",
            f"Unable to split dataset.\n\n{e}"
        )


# =========================================================
# CHECK DATASET
# =========================================================

def check_dataset():
    if X_train is None:
        messagebox.showwarning(
            "Warning",
            "Please upload and split the dataset first."
        )
        return False

    return True


# =========================================================
# KNN
# =========================================================

def knn():
    global knn_acc

    if not check_dataset():
        return

    try:
        knn_model = KNeighborsClassifier(
            n_neighbors=5
        )

        knn_model.fit(
            X_train,
            y_train
        )

        y_pred = knn_model.predict(
            X_test
        )

        knn_acc = accuracy_score(
            y_test,
            y_pred
        )

        text.insert(
            END,
            f"\nKNN Accuracy: {knn_acc * 100:.2f}%\n"
        )

    except Exception as e:
        messagebox.showerror(
            "KNN Error",
            str(e)
        )


# =========================================================
# NAIVE BAYES
# =========================================================

def naive_bayes():
    global nb_acc

    if not check_dataset():
        return

    try:
        nb_model = GaussianNB()

        nb_model.fit(
            X_train,
            y_train
        )

        y_pred = nb_model.predict(
            X_test
        )

        nb_acc = accuracy_score(
            y_test,
            y_pred
        )

        text.insert(
            END,
            f"\nNaive Bayes Accuracy: {nb_acc * 100:.2f}%\n"
        )

    except Exception as e:
        messagebox.showerror(
            "Naive Bayes Error",
            str(e)
        )


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

def logistic_regression():
    global model, lr_accuracy

    if not check_dataset():
        return

    try:
        model = LogisticRegression(
            solver="liblinear",
            max_iter=2000
        )

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

        lr_accuracy = accuracy_score(
            y_test,
            y_pred
        )

        text.insert(
            END,
            f"\nLogistic Regression Accuracy: "
            f"{lr_accuracy * 100:.2f}%\n"
        )

    except Exception as e:
        messagebox.showerror(
            "Logistic Regression Error",
            str(e)
        )


# =========================================================
# PLOT RESULTS
# =========================================================

def plot_bar_graph():

    if not check_dataset():
        return

    if (
        knn_acc is None
        or nb_acc is None
        or lr_accuracy is None
    ):
        messagebox.showwarning(
            "Warning",
            "Please run KNN, Naive Bayes, and "
            "Logistic Regression first."
        )
        return

    algorithms = [
        "KNN",
        "Naive Bayes",
        "Logistic Regression"
    ]

    accuracies = [
        knn_acc * 100,
        nb_acc * 100,
        lr_accuracy * 100
    ]

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        algorithms,
        accuracies,
        color=[
            "#3498db",
            "#e67e22",
            "#2ecc71"
        ]
    )

    plt.xlabel("Algorithms")
    plt.ylabel("Accuracy (%)")

    plt.title(
        "Accuracy of ML Algorithms"
    )

    plt.ylim(0, 100)

    for index, accuracy in enumerate(accuracies):
        plt.text(
            index,
            accuracy + 1,
            f"{accuracy:.2f}%",
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout()
    plt.show()


# =========================================================
# PREPARE PREDICTION DATA
# =========================================================

def prepare_prediction_data(input_data):

    if not training_columns:
        raise ValueError(
            "Please split the dataset before prediction."
        )

    data = input_data.copy()

    data.rename(
        columns=COLUMN_MAPPING,
        inplace=True
    )

    data = clean_missing_values(data)

    if "id" in data.columns:
        data.drop(
            columns=["id"],
            inplace=True
        )

    missing_columns = [
        column
        for column in training_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The prediction file is missing these columns:\n\n"
            + "\n".join(missing_columns)
        )

    data = data[
        training_columns
    ].copy()

    for column in training_columns:

        if column in feature_encoders:

            encoder = feature_encoders[column]

            numeric_values = pd.to_numeric(
                data[column],
                errors="coerce"
            )

            if (
                numeric_values.notna().all()
                and numeric_values.min() >= 0
                and numeric_values.max()
                < len(encoder.classes_)
            ):
                data[column] = numeric_values
                continue

            mode_value = encoder.classes_[0]

            values = (
                data[column]
                .fillna(mode_value)
                .astype(str)
                .str.strip()
            )

            unknown_values = sorted(
                set(values)
                - set(encoder.classes_)
            )

            if unknown_values:
                raise ValueError(
                    f"Unknown value(s) in column "
                    f"'{column}': {unknown_values}"
                )

            data[column] = encoder.transform(
                values
            )

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

    data = data.fillna(0)

    if data.isna().sum().sum() != 0:
        raise ValueError(
            "Prediction data still contains missing values."
        )

    return data


# =========================================================
# PREDICTION
# =========================================================

def predict():

    if model is None:
        messagebox.showwarning(
            "Warning",
            "Please run Logistic Regression first."
        )
        return

    filename = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[("CSV files", "*.csv")]
    )

    if not filename:
        return

    try:
        input_data = pd.read_csv(
            filename
        )

        prediction_data = prepare_prediction_data(
            input_data
        )

        y_pred = model.predict(
            prediction_data
        )

        text.delete(
            "1.0",
            END
        )

        text.insert(
            END,
            "Prediction Results\n"
        )

        text.insert(
            END,
            "========================\n\n"
        )

        for index, prediction in enumerate(
            y_pred,
            start=1
        ):

            if target_encoder is not None:
                result = target_encoder.inverse_transform(
                    [int(prediction)]
                )[0]
            else:
                result = prediction

            text.insert(
                END,
                f"Row {index}: {result}\n"
            )

    except Exception as e:
        messagebox.showerror(
            "Prediction Error",
            f"Unable to make predictions.\n\n{e}"
        )


# =========================================================
# MAIN WINDOW
# =========================================================

main = tk.Tk()

main.title(
    "PREDICTION OF CHRONIC KIDNEY DISEASE"
)

main.geometry(
    "1200x800"
)

main.configure(
    bg="#34495e"
)


# =========================================================
# TITLE
# =========================================================

font_title = (
    "Helvetica",
    20,
    "bold"
)

title = tk.Label(
    main,
    text="PREDICTION OF CHRONIC KIDNEY DISEASE",
    font=font_title,
    bg="#2c3e50",
    fg="white",
    height=2,
    width=80
)

title.place(
    x=0,
    y=5
)


# =========================================================
# TEXT AREA
# =========================================================

font1 = (
    "Arial",
    12,
    "bold"
)

text = tk.Text(
    main,
    height=20,
    width=100,
    font=font1
)

text.place(
    x=150,
    y=100
)


# =========================================================
# PATH LABEL
# =========================================================

pathlabel = tk.Label(
    main,
    bg="DarkOrange1",
    fg="white",
    font=font1,
    width=65
)

pathlabel.place(
    x=330,
    y=550
)


# =========================================================
# BUTTON STYLE
# =========================================================

button_style = {
    "font": (
        "Arial",
        12,
        "bold"
    ),
    "fg": "white",
    "width": 20,
    "height": 2,
    "bd": 3,
    "relief": "raised"
}


# =========================================================
# BUTTONS
# =========================================================

uploadButton = tk.Button(
    main,
    text="Upload Dataset",
    command=upload,
    bg="#2980b9",
    **button_style
)

uploadButton.place(
    x=50,
    y=600
)


splitButton = tk.Button(
    main,
    text="Split Dataset",
    command=splitdataset,
    bg="#27ae60",
    **button_style
)

splitButton.place(
    x=250,
    y=600
)


knnButton = tk.Button(
    main,
    text="KNN",
    command=knn,
    bg="#f39c12",
    **button_style
)

knnButton.place(
    x=450,
    y=600
)


naiveBayesButton = tk.Button(
    main,
    text="Naive Bayes",
    command=naive_bayes,
    bg="#c0392b",
    **button_style
)

naiveBayesButton.place(
    x=650,
    y=600
)


logisticButton = tk.Button(
    main,
    text="Logistic Regression",
    command=logistic_regression,
    bg="#8e44ad",
    **button_style
)

logisticButton.place(
    x=850,
    y=600
)


plotButton = tk.Button(
    main,
    text="Plot Results",
    command=plot_bar_graph,
    bg="#7f8c8d",
    **button_style
)

plotButton.place(
    x=1050,
    y=600
)


predict_button = tk.Button(
    main,
    text="Prediction",
    command=predict,
    bg="#d35400",
    **button_style
)

predict_button.place(
    x=550,
    y=680
)


# =========================================================
# START APPLICATION
# =========================================================

main.mainloop()