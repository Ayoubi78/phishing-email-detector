import pandas as pd

# Load dataset
df = pd.read_csv("data/raw_emails.csv")

# Show first 5 rows
print(df.head())

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

# Remove empty rows
df = df.dropna()

# Save cleaned dataset
df.to_csv("data/cleaned_emails.csv", index=False)

# Display information
print("\nDataset cleaned successfully!")
print(df.head())

# Count labels
print("\nLabel counts:")
print(df["label"].value_counts())