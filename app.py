from logging import INFO

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Sentiment Dashboard",
    layout="wide"
)

# ---------------- AUTO REFRESH ----------------
st_autorefresh(
    interval=100000,
    key="dashboard_refresh"
)

# ---------------- LOAD DATA ----------------
try:
    df = pd.read_csv("data/cleaned_twitter_data.csv")
    df.to_csv("powerbi_data.csv", index=False)
except Exception as e:
    st.error(f"Dataset Load Error: {e}")
    st.stop()

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Model Load Error: {e}")
    st.stop()

# ---------------- TITLE ----------------
st.title("Real-Time Social Media Sentiment Analysis Dashboard")

# ---------------- PREDICTION ----------------
st.subheader("Try Sentiment Prediction")

tweet = st.text_area("Enter Tweet")

if st.button("Analyze Sentiment"):

    if tweet.strip():

        tweet_vector = vectorizer.transform([tweet])

        prediction = model.predict(tweet_vector)

        st.success(
            f"Predicted Sentiment: {prediction[0]}"
        )

    else:
        st.warning("Please enter a tweet.")

