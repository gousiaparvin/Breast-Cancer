# Breast-Cancer      
Breast Cancer Prediction
Overview
This project develops a machine learning model to predict breast cancer using the Breast Cancer Wisconsin dataset. It applies a Support Vector Machine (SVM) classifier for distinguishing between malignant and benign tumors.

Dataset
The dataset, available on Kaggle, consists of features computed from digitized images of FNA breast masses, focusing on predicting the diagnosis.https://www.kaggle.com/code/mragpavank/breast-cancer-wisconsin

Requirements
Python 3.10.12
scikit-learn, pandas, numpy

Install dependencies:
pip install numpy pandas scikit-learn

Features
Uses SVM with linear kernel
Employs k-fold cross-validation and GridSearchCV for robustness and tuning
Evaluates using accuracy, precision, recall, F1-score

Usage
Run python breast_cancer_prediction.py to execute the model.
