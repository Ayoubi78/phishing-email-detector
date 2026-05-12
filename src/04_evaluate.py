import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/clean_emails.csv")

df = df.dropna(subset=["clean_text", "label"])
df["clean_text"] = df["clean_text"].astype(str)

X = df["clean_text"]
y = df["label"]

# -----------------------------
# TRAIN/TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# LOAD VECTORISER
# -----------------------------
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# LOAD MODELS
# -----------------------------
with open("models/nb_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("models/lr_model.pkl", "rb") as f:
    lr_model = pickle.load(f)

# -----------------------------
# PREDICTIONS
# -----------------------------
nb_preds = nb_model.predict(X_test_vec)
lr_preds = lr_model.predict(X_test_vec)

# -----------------------------
# METRICS FUNCTION
# -----------------------------
def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))

# Evaluate both models
evaluate_model("Naive Bayes", y_test, nb_preds)
evaluate_model("Logistic Regression", y_test, lr_preds)

# -----------------------------
# CONFUSION MATRIX (BEST MODEL = Logistic Regression)
# -----------------------------
cm = confusion_matrix(y_test, lr_preds)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Save plot
plt.savefig("reports/confusion_matrix.png")

print("\nConfusion matrix saved!")

# -----------------------------
# SAVE REPORT
# -----------------------------
report_text = f"""
# Model Evaluation Report

## Key Metrics (Logistic Regression)

- Accuracy: {accuracy_score(y_test, lr_preds)}
- Precision: {precision_score(y_test, lr_preds)}
- Recall: {recall_score(y_test, lr_preds)}
- F1 Score: {f1_score(y_test, lr_preds)}

## Interpretation

- False Positives: Normal emails marked as phishing
- False Negatives: Phishing emails NOT detected (MOST DANGEROUS)

In cybersecurity, **recall is more important** because missing phishing emails is risky.
"""

with open("reports/metrics.md", "w") as f:
    f.write(report_text)

print("Metrics report saved!")