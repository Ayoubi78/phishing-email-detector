import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/clean_emails.csv")

# Use only clean text + label
X = df["clean_text"]
y = df["label"]

print("\nDataset loaded successfully!")
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

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))

# -----------------------------
# TF-IDF VECTORIZATION
# -----------------------------

vectorizer = TfidfVectorizer(
    max_features=5000,   # limit vocabulary size
    ngram_range=(1, 2)   # unigrams + bigrams
)

# Fit only on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform test data
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF conversion completed!")
print("Train shape:", X_train_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)

# -----------------------------
# SAVE VECTORISER (IMPORTANT)
# -----------------------------
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nVectorizer saved in models/tfidf_vectorizer.pkl")