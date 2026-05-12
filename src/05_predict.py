import pickle

# Load vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load model (use best model - Logistic Regression)
with open("models/lr_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_email(text):
    # convert to vector
    text_vec = vectorizer.transform([text])

    # prediction
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    if prediction == 1:
        label = "PHISHING 🚨"
    else:
        label = "SAFE ✅"

    print("\nEmail Text:", text)
    print("Prediction:", label)
    print("Confidence:", max(probability))

# -----------------------------
# TEST EXAMPLES
# -----------------------------
while True:
    user_input = input("\nEnter email text (or type 'exit'): ")

    if user_input.lower() == "exit":
        break

    predict_email(user_input)