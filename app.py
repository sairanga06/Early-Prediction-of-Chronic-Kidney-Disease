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


# =========================
# GLOBAL VARIABLES
# =========================

filename = None
df = None

X_train = None
X_test = None
y_train = None
y_test = None

model = None

knn_acc = 0
nb_acc = 0
lr_accuracy = 0

feature_encoders = {}
target_encoder = None

training_columns = []


# =========================
# COLUMN NAME MAPPING
# =========================

COLUMN_MAPPING = {
    "blood_pressure": "bp",
    "specific_gravity": "sg",
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
    "anemia": "ane"
}


# =========================
# UPLOAD DATASET
# =========================

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
            messagebox.showerror("Error", "The selected dataset is empty.")
            return

        df.fillna(df.mode().iloc[0], inplace=True)

        pathlabel.config(text=filename)

        text.delete("1.0", END)

        text.insert(END, "Dataset loaded successfully\n\n")
        text.insert(END, "Dataset Size: " + str(len(df)) + " rows\n")
        text.insert(END, "Number of Columns: " + str(len(df.columns)) + "\n\n")

        text.insert(END, "Columns:\n")

        for column in df.columns:
            text.insert(END, f"- {column}\n")

    except Exception as e:
        messagebox.showerror(
            "Dataset Error",
            f"Unable to load dataset.\n\n{str(e)}"
        )


# =========================
# SPLIT DATASET
# =========================

def splitdataset():
    global df
    global X_train, X_test, y_train, y_test
    global feature_encoders, target_encoder
    global training_columns

    if df is None:
        messagebox.showwarning(
            "Warning",
            "Please upload a dataset first."
        )
        return

    try:
        data = df.copy()

        # Rename columns if modified dataset is selected
        data.rename(columns=COLUMN_MAPPING, inplace=True)

        # Remove ID because it is not a useful ML feature
        if "id" in data.columns:
            data.drop(columns=["id"], inplace=True)

        # Last column is the target
        target_column = data.columns[-1]

        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Store training column names
        training_columns = list(X.columns)

        feature_encoders = {}

        # Convert feature columns into numeric values
        for column in X.columns:

            if X[column].dtype == "object":

                encoder = LabelEncoder()

                X[column] = encoder.fit_transform(
                    X[column].astype(str)
                )

                feature_encoders[column] = encoder

            else:

                X[column] = pd.to_numeric(
                    X[column],
                    errors="coerce"
                )

        # Handle any remaining missing numeric values
        X.fillna(X.median(numeric_only=True), inplace=True)

        # Encode target
        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(
            y.astype(str)
        )

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=21,
            stratify=y
        )

        # Save test data for reference
        X_test.to_csv(
            "test1.csv",
            index=False
        )

        text.delete("1.0", END)

        text.insert(END, "Dataset split successfully\n\n")

        text.insert(
            END,
            "Training Size: "
            + str(len(X_train))
            + "\n"
        )

        text.insert(
            END,
            "Test Size: "
            + str(len(X_test))
            + "\n\n"
        )

        text.insert(END, "Features used:\n")

        for column in training_columns:
            text.insert(END, f"- {column}\n")

    except Exception as e:

        messagebox.showerror(
            "Split Error",
            f"Unable to split dataset.\n\n{str(e)}"
        )


# =========================
# CHECK DATASET
# =========================

def check_dataset():

    if X_train is None:
        messagebox.showwarning(
            "Warning",
            "Please split the dataset first."
        )
        return False

    return True


# =========================
# KNN
# =========================

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


# =========================
# NAIVE BAYES
# =========================

def naive_bayes():

    global nb_acc

    if not check_dataset():
        return

    try:

        nb = GaussianNB()

        nb.fit(
            X_train,
            y_train
        )

        y_pred = nb.predict(
            X_test
        )

        nb_acc = accuracy_score(
            y_test,
            y_pred
        )

        text.insert(
            END,
            f"\nNaive Bayes Accuracy: "
            f"{nb_acc * 100:.2f}%\n"
        )

    except Exception as e:

        messagebox.showerror(
            "Naive Bayes Error",
            str(e)
        )


# =========================
# LOGISTIC REGRESSION
# =========================

def logistic_regression():

    global model, lr_accuracy

    if not check_dataset():
        return

    try:

        model = LogisticRegression(
            solver="liblinear",
            max_iter=1000
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


# =========================
# PLOT RESULTS
# =========================

def plot_bar_graph():

    if X_train is None:
        messagebox.showwarning(
            "Warning",
            "Please split the dataset first."
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

    plt.figure(figsize=(9, 6))

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

    for i, accuracy in enumerate(accuracies):

        plt.text(
            i,
            accuracy + 1,
            f"{accuracy:.2f}%",
            ha="center",
            fontweight="bold"
        )

    plt.tight_layout()

    plt.show()


# =========================
# PREPARE PREDICTION DATA
# =========================

def prepare_prediction_data(input_data):

    global training_columns

    data = input_data.copy()

    # Rename modified dataset columns
    data.rename(
        columns=COLUMN_MAPPING,
        inplace=True
    )

    # Remove ID if it exists
    if "id" in data.columns:
        data.drop(
            columns=["id"],
            inplace=True
        )

    # Check missing columns
    missing_columns = [
        column
        for column in training_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            "The prediction CSV is missing these columns:\n\n"
            + "\n".join(missing_columns)
        )

    # Keep only training columns
    data = data[
        training_columns
    ]

    # Process each column
    for column in training_columns:

        if column in feature_encoders:

            encoder = feature_encoders[column]

            # Convert values to strings
            values = data[column].astype(str)

            # Check whether values already look like encoded numbers
            try:

                numeric_values = pd.to_numeric(
                    data[column]
                )

                valid_encoded = (
                    numeric_values.min() >= 0
                    and
                    numeric_values.max() < len(
                        encoder.classes_
                    )
                )

                if valid_encoded:
                    data[column] = numeric_values
                    continue

            except Exception:
                pass

            # Encode original categorical values
            known_classes = set(
                encoder.classes_
            )

            unknown_values = [
                value
                for value in values
                if value not in known_classes
            ]

            if unknown_values:

                raise ValueError(
                    f"Unknown value '{unknown_values[0]}' "
                    f"found in column '{column}'."
                )

            data[column] = encoder.transform(
                values
            )

        else:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # Fill missing numeric values
    data.fillna(
        X_train.median(),
        inplace=True
    )

    return data


# =========================
# PREDICTION
# =========================

def predict():

    global model

    if model is None:

        messagebox.showwarning(
            "Warning",
            "Please run Logistic Regression first."
        )

        return

    filename = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=[
            ("CSV files", "*.csv")
        ]
    )

    if not filename:
        return

    try:

        input_data = pd.read_csv(
            filename
        )

        input_data.fillna(
            input_data.mode().iloc[0],
            inplace=True
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

        for i, prediction in enumerate(
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
                f"Row {i}: {result}\n"
            )

    except Exception as e:

        messagebox.showerror(
            "Prediction Error",
            f"Unable to make predictions.\n\n{str(e)}"
        )


# =========================
# MAIN WINDOW
# =========================

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


# =========================
# TITLE
# =========================

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


# =========================
# TEXT AREA
# =========================

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


# =========================
# PATH LABEL
# =========================

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


# =========================
# BUTTON STYLE
# =========================

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


# =========================
# BUTTONS
# =========================

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


# =========================
# START APPLICATION
# =========================

main.mainloop()