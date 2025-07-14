# Anemia Sense – Anemia Detection System Using Machine Learning

Anemia Sense is a machine learning-based web application that helps users predict the likelihood of having anemia based on key medical parameters. The model is trained using clinical data and is deployed through a user-friendly web interface built with Flask.

# Project Description

Anemia is a common medical condition in which the body lacks enough healthy red blood cells to carry adequate oxygen to tissues. This project aims to detect anemia by using patient health data and classifying whether a person is anemic or not using various ML classifiers.

The web application allows users to input their gender and medical test values like Hemoglobin, MCH, MCHC, and MCV. Based on the input, the trained machine learning model (Gradient Boosting Classifier) predicts if the user has anemia.


# Features

- Predicts Anemia from 5 clinical parameters.
- Uses Gradient Boosting Classifier (best performing model).
- User-friendly interface developed with Flask.
- Visual feedback based on prediction results.
- Trained using real-world clinical dataset.


# Technologies Used

| Component     | Technology                     |
|---------------|--------------------------------|
| Language      | Python                         |
| Web Framework | Flask                          |
| ML Libraries  | Scikit-learn, Pandas, NumPy    |
| Visualization | Matplotlib, Seaborn            |
| Frontend      | HTML, CSS                      |
| Deployment    | (To be decided: Render, Streamlit, etc.) |


# Model Training

- Dataset: Real-world clinical dataset containing features like:
  - `Gender` (0: Male, 1: Female)
  - `Hemoglobin`
  - `MCH`
  - `MCHC`
  - `MCV`
- Models Tested:
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Naive Bayes
  - SVM
  - **Gradient Boosting (Best Accuracy)**
- Evaluation Metric: Accuracy Score, Classification Report


