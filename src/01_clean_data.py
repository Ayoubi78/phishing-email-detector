import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/cleaned_emails.csv")

# -----------------------------
# DATA UNDERSTANDING
# -----------------------------

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATASET INFO:")
print(df.info())

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nDUPLICATES:")
print(df.duplicated().sum())

print("\nCLASS DISTRIBUTION:")
print(df["label"].value_counts())

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------

df = df.drop_duplicates()

# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------

stop_words = set(stopwords.words('english'))

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# -----------------------------
# APPLY CLEANING
# -----------------------------

df["clean_text"] = df["email_text"].astype(str).apply(clean_text)

# -----------------------------
# SAVE CLEAN DATA
# -----------------------------

df.to_csv("data/clean_emails.csv", index=False)

print("\nDATA CLEANING COMPLETED!")
print("\nFIRST 5 CLEANED ROWS:")
print(df[["email_text", "clean_text"]].head())