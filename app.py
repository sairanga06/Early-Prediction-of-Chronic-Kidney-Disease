from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

global filename
global df, X_train, X_test, y_train, y_test

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    df.fillna(df.mode().iloc[0], inplace=True)
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def splitdataset(): 
    global df, X_train, X_test, y_train, y_test
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    label_encoder = LabelEncoder()
    for column in X.columns:
        if X[column].dtype == 'object':
            X[column] = label_encoder.fit_transform(X[column])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Training Size: " + str(len(X_train)) + "\n")
    text.insert(END, "Test Size: " + str(len(X_test)) + "\n")
    X_test.to_csv("test1.csv", index=False)

def knn():
    global knn_acc
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    knn_acc = accuracy_score(y_test, y_pred)
    text.insert(END, f'KNN Accuracy: {knn_acc * 100:.2f}%\n')

def naive_bayes():
    global nb_acc
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    nb_acc = accuracy_score(y_test, y_pred)
    text.insert(END, f'Naive Bayes Accuracy: {nb_acc * 100:.2f}%\n')

def logistic_regression():
    global model, lr_accuracy
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, y_pred)
    text.insert(END, f'Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%\n')

def plot_bar_graph():
    algorithms = ['KNN', 'Naive Bayes', 'Logistic Regression']
    accuracies = [knn_acc * 100, nb_acc * 100, lr_accuracy * 100]
    colors = ['#3498db', '#e67e22', '#2ecc71']
    plt.bar(algorithms, accuracies, color=colors)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of ML Algorithms')
    plt.show()

def predict():
    filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    if filename:
        input_data = pd.read_csv(filename)
        input_data.fillna(input_data.mode().iloc[0], inplace=True)
        label_encoder = LabelEncoder()
        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoder.fit_transform(input_data[column])
        y_pred = model.predict(input_data)
        text.delete('1.0', END)
        text.insert(END, "Prediction Results:\n\n")
        for i, pred in enumerate(y_pred, start=1):
            text.insert(END, f"Row {i}: {pred}\n")

main = tk.Tk()
main.title("PREDICTION OF CHRONIC KIDNEY DISEASE")
main.geometry("1200x800")
main.configure(bg='#34495e')

font_title = ('Helvetica', 20, 'bold')
title = tk.Label(main, text='PREDICTION OF CHRONIC KIDNEY DISEASE', font=font_title, bg='#2c3e50', fg='white', height=2, width=80)
title.place(x=0, y=5)

font1 = ('Arial', 12, 'bold')
text = tk.Text(main, height=20, width=100, font=font1)
text.place(x=150, y=100)
pathlabel = tk.Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=330, y=550)

button_style = {
    "font": ('Arial', 12, 'bold'),
    "fg": 'white',
    "width": 20,
    "height": 2,
    "bd": 3,
    "relief": "raised"
}

uploadButton = tk.Button(main, text="Upload Dataset", command=upload, bg="#2980b9", **button_style)
uploadButton.place(x=50, y=600)

splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, bg="#27ae60", **button_style)
splitButton.place(x=250, y=600)

knnButton = tk.Button(main, text="KNN", command=knn, bg="#f39c12", **button_style)
knnButton.place(x=450, y=600)

naiveBayesButton = tk.Button(main, text="Naive Bayes", command=naive_bayes, bg="#c0392b", **button_style)
naiveBayesButton.place(x=650, y=600)

logisticButton = tk.Button(main, text="Logistic Regression", command=logistic_regression, bg="#8e44ad", **button_style)
logisticButton.place(x=850, y=600)

plotButton = tk.Button(main, text="Plot Results", command=plot_bar_graph, bg="#7f8c8d", **button_style)
plotButton.place(x=1050, y=600)

predict_button = tk.Button(main, text="Prediction", command=predict, bg="#d35400", **button_style)
predict_button.place(x=550, y=680)

main.mainloop()
