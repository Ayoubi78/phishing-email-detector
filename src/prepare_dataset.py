import pandas as pd

# Load dataset
df = pd.read_csv("data/raw_emails.csv")

# Remove unwanted index column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Rename columns
df = df.rename(columns={
    "Email Text": "email_text",
    "Email Type": "label"
})

# Convert labels to numbers
df["label"] = df["label"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})

# Remove missing rows
df = df.dropna()

# Save cleaned dataset
df.to_csv("data/cleaned_emails.csv", index=False)

# Show results
print("\nDataset cleaned successfully!\n")

print(df.head())

print("\nLabel counts:")
print(df["label"].value_counts())