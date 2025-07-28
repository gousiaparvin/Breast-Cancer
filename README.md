     
# Breast Cancer Diagnosis Using Machine Learning (Python)
📅 Date: February 14

# 🎯 Aim of the Project:
To build a predictive classification model that distinguishes between malignant and benign breast cancer tumors using clinical diagnostic features, thereby supporting early and accurate diagnosis.

# 📊 Data Source:
Dataset: breast cancer.csv (likely based on the Wisconsin Breast Cancer Dataset)

Features: 30 numerical diagnostic measurements (e.g., radius, texture, perimeter, area, smoothness)

Target: diagnosis — M (Malignant) or B (Benign)

# 🔁 Process Workflow:
# 1. Data Loading & Exploration
Uploaded dataset using google.colab.files

Inspected structure using .head(), .info(), .describe(), and df.isna().sum()

Removed any columns with null values

# 2. EDA (Exploratory Data Analysis)
Plotted diagnosis distribution using Seaborn

Applied Label Encoding (M → 1, B → 0)

Visualized feature relationships via:

pairplot() (radius, texture, perimeter)

heatmap() of correlation matrix (top 10 features)

# 3. Feature Selection & Data Preprocessing
Selected relevant features (X = df.iloc[:, 2:31])

Target = diagnosis

Used train_test_split() for 70:30 train-test split

Scaled features using StandardScaler

# 4. Model Training
Two models were trained:

DecisionTreeClassifier (Entropy)

RandomForestClassifier (10 estimators, Entropy)

# 5. Model Evaluation
Used .score() for training accuracy

Implemented 5-fold Cross-Validation on training & test sets using cross_val_score

Performed hyperparameter tuning using GridSearchCV:

Tuned max_depth, min_samples_split, and min_samples_leaf for Decision Tree

Tuned n_estimators, max_depth, and others for Random Forest

# 6. Final Evaluation
Predicted on test set

Generated:

classification_report

Accuracy score

Sensitivity & specificity using confusion_matrix

Visualized results with confusion matrix heatmaps

# ⚙️ Key Functions Used and Purpose:
Function	Purpose
LabelEncoder()	Converts 'M' and 'B' into numeric (1/0)
StandardScaler()	Normalizes feature ranges
DecisionTreeClassifier()	Baseline classification model
RandomForestClassifier()	Ensemble model for higher accuracy
cross_val_score()	Evaluate model robustness via k-fold cross-validation
GridSearchCV()	Optimize hyperparameters
classification_report()	Precision, Recall, F1-score
confusion_matrix()	Evaluate TP, TN, FP, FN for calculating sensitivity/specificity
seaborn.heatmap()	Visual correlation and performance metrics

# ✅ Results:
Metric	Decision Tree	Random Forest
Training Accuracy	~100%	~100%
CV Accuracy (Train)	~93–95%	~95–97%
Test Accuracy	~90–95%	~95–98%
Sensitivity (Recall for 'M')	High	High
Specificity (Recall for 'B')	High	High
Best Parameters (after tuning)	Shown via GridSearchCV	Shown via GridSearchCV

🔍 Random Forest performed better overall, with high stability and better generalization on unseen data.

# 🚀 Future Enhancements:
Integrate deep learning models (e.g., MLP or CNNs) for improved feature extraction

Use SHAP or LIME for model interpretability

Deploy model via Flask API or Streamlit dashboard for clinical use

Perform feature selection techniques (e.g., PCA, RFE)

Try imbalanced classification techniques if M/B classes are skewed
















































# Breast Cancer Prediction
Overview
This project develops a machine learning model to predict breast cancer using the Breast Cancer Wisconsin dataset. It applies a Support Vector Machine (SVM) classifier for distinguishing between malignant and benign tumors.

# Dataset
The dataset, available on Kaggle, consists of features computed from digitized images of FNA breast masses, focusing on predicting the diagnosis.https://www.kaggle.com/code/mragpavank/breast-cancer-wisconsin

# Requirements
Python 3.10.12
scikit-learn, pandas, numpy

# Install dependencies:
pip install numpy pandas scikit-learn

# Features
Uses SVM with linear kernel
Employs k-fold cross-validation and GridSearchCV for robustness and tuning
Evaluates using accuracy, precision, recall, F1-score

# Usage
Run python breast_cancer_prediction.py to execute the model.
