import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/clean_emails.csv")

# safety check
df = df.dropna(subset=["clean_text", "label"])
df["clean_text"] = df["clean_text"].astype(str)

X = df["clean_text"]
y = df["label"]

print("\nDataset loaded!")
print("Total samples:", len(df))

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# LOAD TF-IDF VECTORISER
# -----------------------------
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# MODEL 1: NAIVE BAYES (BASELINE)
# -----------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

nb_preds = nb_model.predict(X_test_vec)

nb_acc = accuracy_score(y_test, nb_preds)

print("\n===== NAIVE BAYES =====")
print("Accuracy:", nb_acc)
print(classification_report(y_test, nb_preds))

# Save model
with open("models/nb_model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

print("Naive Bayes model saved!")

# -----------------------------
# MODEL 2: LOGISTIC REGRESSION
# -----------------------------
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_vec, y_train)

lr_preds = lr_model.predict(X_test_vec)

lr_acc = accuracy_score(y_test, lr_preds)

print("\n===== LOGISTIC REGRESSION =====")
print("Accuracy:", lr_acc)
print(classification_report(y_test, lr_preds))

# Save model
with open("models/lr_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

print("Logistic Regression model saved!")

# -----------------------------
# FINAL COMPARISON
# -----------------------------
print("\n===== FINAL COMPARISON =====")
print("Naive Bayes Accuracy:", nb_acc)
print("Logistic Regression Accuracy:", lr_acc)