import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load cleaned data
df = pd.read_csv("data/cleaned_twitter_data.csv")

# Remove any remaining missing values
df = df.dropna(subset=["Clean_Tweet"])

# Convert to string
df["Clean_Tweet"] = df["Clean_Tweet"].astype(str)

print("Dataset Shape:", df.shape)

X = df["Clean_Tweet"].fillna("")
y = df["Sentiment"]

print(df["Clean_Tweet"].isnull().sum())

# Features and Target
X = df["Clean_Tweet"]
y = df["Sentiment"]

# Convert text into numbers
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# Save Model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model Saved Successfully!")