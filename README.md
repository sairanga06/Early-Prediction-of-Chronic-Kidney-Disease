# Early Prediction of Chronic Kidney Disease

A machine learning based web application for predicting Chronic Kidney Disease (CKD) using patient clinical information.

The project originally started as a desktop application and has been converted into a Flask-based web application while keeping the machine learning prediction functionality.

---

## Project Overview

Chronic Kidney Disease is a serious health condition that can be difficult to identify at an early stage.

This project uses a machine learning model to analyze clinical parameters and provide a CKD prediction.

The web application allows users to enter patient information through a simple web interface and receive a model prediction.

> **Disclaimer:** This project is developed for educational and project demonstration purposes. The prediction should not be considered a medical diagnosis or a substitute for professional medical advice.

---

## Features

- Machine learning based CKD prediction
- Flask backend
- Web-based prediction interface
- 24 clinical input features
- Logistic Regression model
- Dataset loading and preprocessing
- Input validation
- Backend error handling
- Prediction result display
- Clear Form functionality
- Health-check endpoint
- Responsive web interface
- Medical disclaimer

---

## Technologies Used

### Programming Language

- Python

### Backend

- Flask

### Machine Learning

- Scikit-learn
- Logistic Regression
- Pandas
- NumPy

### Frontend

- HTML5
- CSS3
- JavaScript

### Development Tools

- Visual Studio Code
- Git
- GitHub

---

## Project Structure

```text
Early-Prediction-of-Chronic-Kidney-Disease-main
│
├── Dataset
│   └── CKD.csv
│
├── web_app
│   │
│   ├── app.py
│   │
│   ├── templates
│   │   └── index.html
│   │
│   └── static
│       └── style.css
│
├── screenshots
│
├── .gitignore
├── requirements.txt
├── README.md
└── ...