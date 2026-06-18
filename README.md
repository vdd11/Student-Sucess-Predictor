#Render Link
https://student-sucess-predictor-1.onrender.com/predict









# Student Success Analytics Platform

A Flask machine learning web app that predicts whether a student is likely to pass based on academic, demographic, and school-related features from the UCI Student Performance dataset.

## Project Goal

This project demonstrates an end-to-end data science workflow:

1. Find a public dataset
2. Clean and analyze the data
3. Train and evaluate machine learning models
4. Build a Flask web application
5. Prepare the project for Render deployment

## Dataset

Dataset: UCI Machine Learning Repository Student Performance Dataset  
Source: https://archive.ics.uci.edu/dataset/320/student%2Bperformance

The dataset contains student achievement data from two Portuguese secondary schools, including grades, demographic information, social factors, and school-related features.

## Model

The app trains multiple classification models and saves the best one:

- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

Target variable:

- `pass_status = 1` if final grade `G3 >= 10`
- `pass_status = 0` if final grade `G3 < 10`

## Features Used in the App

The deployed form uses a simplified set of student-facing inputs:

- Age
- Study time
- Past failures
- Absences
- First period grade
- Second period grade
- School support
- Family support
- Paid tutoring
- Internet access
- Higher education goal
- Romantic relationship status

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Flask
- HTML/CSS
- Joblib
- Render

## Project Structure

```text
student_success_predictor/
├── app.py
├── train_model.py
├── requirements.txt
├── render.yaml
├── Procfile
├── README.md
├── data/
├── models/
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
└── templates/
    ├── index.html
    ├── predict.html
    ├── dashboard.html
    └── about.html
```

## How to Run Locally

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Render Deployment

1. Push this project to GitHub.
2. Create a new Web Service on Render.
3. Connect your GitHub repository.
4. Use these settings:

Build command:

```bash
pip install -r requirements.txt && python train_model.py
```

Start command:

```bash
gunicorn app:app
```

## Resume Bullet

Built and deployed a Flask-based machine learning application that predicts student academic outcomes using Scikit-Learn, Pandas, and the UCI Student Performance dataset, combining classification modeling, model evaluation, and an interactive analytics dashboard.
"# Student-Sucess-Predictor" 
