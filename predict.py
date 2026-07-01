import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# User input
text = input("Enter Tweet: ")

# Convert text
text_vector = vectorizer.transform([text])

# Predict
prediction = model.predict(text_vector)

print("\nPredicted Sentiment:", prediction[0])