# Phishing Email Detection System

## 📌 Project Overview
This project is a machine learning-based phishing email detection system. It classifies emails as either SAFE or PHISHING using Natural Language Processing (NLP) and supervised learning models.

---

## ⚙️ Tech Stack
- Python
- Scikit-learn
- Pandas
- NLTK
- TF-IDF Vectorization

---

## 🧠 Models Used
- Multinomial Naive Bayes (Baseline)
- Logistic Regression (Final Model)

---

## 📊 Performance
- Accuracy: ~97%
- Recall: ~96% (important for phishing detection)

---

## 🚀 How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run training
python src/03_train_model.py

### 3. Run prediction
python src/05_predict.py

---

## 🎯 Key Insight
Recall is prioritized over accuracy because missing phishing emails is more dangerous than false alerts.

---

## 📌 Output Example
Input:
"Your bank account is suspended. Click here immediately."

Output:
PHISHING 🚨

---
# Methodology

## 1. Data Collection
A dataset of safe and phishing emails was used.

## 2. Data Preprocessing
- Lowercasing
- Stopword removal
- Punctuation removal
- Handling missing values

## 3. Feature Engineering
TF-IDF vectorization with unigram and bigram features.

## 4. Model Training
Two models were trained:
- Naive Bayes (baseline)
- Logistic Regression (final model)

## 5. Evaluation
Metrics used:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## 6. Final Model Selection
Logistic Regression was selected due to higher recall and overall performance.

# Project Plan

## Week 1–2
Setup environment and dataset collection

## Week 3
Data cleaning and preprocessing

## Week 4
Feature extraction using TF-IDF

## Week 5
Model training (Naive Bayes & Logistic Regression)

## Week 6
Model evaluation and confusion matrix

## Week 7
Model improvement and prediction system

# Project Log

- Setup project environment completed
- Dataset cleaned and prepared
- TF-IDF vectorization implemented
- Naive Bayes model trained
- Logistic Regression model trained
- Evaluation metrics computed
- Prediction system built
- Improvements applied using bigrams
- Final documentation completed

# Progress Evidence

## Initial Stage
- Raw dataset with noise and duplicates

## Mid Stage
- Cleaned dataset
- TF-IDF feature extraction
- First model trained

## Final Stage
- Improved feature engineering (bigrams)
- Two trained models
- Final prediction system
- Evaluation metrics achieved ~97% accuracy