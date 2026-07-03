import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Real-Time Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# ==========================
# AUTO REFRESH
# ==========================

st_autorefresh(
    interval=100000,
    key="refresh"
)

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#0f172a,#020617);
}

h1,h2,h3,h4,h5,h6,label{
color:white!important;
}

.block-container{
padding-top:2rem;
}

div[data-testid="metric-container"]{
background:#1e293b;
padding:20px;
border-radius:15px;
border:1px solid #334155;
text-align:center;
}

.stButton>button{
width:100%;
height:50px;
background:#2563eb;
color:white;
font-size:18px;
font-weight:bold;
border-radius:10px;
border:none;
}

.stButton>button:hover{
background:#1d4ed8;
}

textarea{
border-radius:10px!important;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# LOAD DATASET
# ==========================

try:

    df = pd.read_csv("data/cleaned_twitter_data.csv")

except Exception as e:

    st.error(e)

    st.stop()

# ==========================
# LOAD MODEL
# ==========================

try:

    model = joblib.load("model.pkl")

    vectorizer = joblib.load("vectorizer.pkl")

except Exception as e:

    st.error(e)

    st.stop()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("📊 Sentiment Dashboard")

st.sidebar.markdown("---")

st.sidebar.write("### Features")

st.sidebar.success("✔ Single Comment")

st.sidebar.success("✔ Multiple Comments")

st.sidebar.success("✔ Charts")

st.sidebar.success("✔ Download CSV")

st.sidebar.success("✔ Power BI")

st.sidebar.markdown("---")

st.sidebar.info("""
Developer

Isha Rane

Data Science Project
""")

# ==========================
# HEADER
# ==========================

st.markdown("""

<div style="padding:25px;
background:linear-gradient(90deg,#2563eb,#06b6d4);
border-radius:15px;
text-align:center;">

<h1 style="color:white;">
📊 Real-Time Social Media Sentiment Analysis Dashboard
</h1>

<p style="color:white;font-size:20px;">
Analyze comments using Machine Learning
</p>

</div>

""", unsafe_allow_html=True)

st.write("")

# ==========================
# KPI CARDS
# ==========================

col1,col2,col3,col4=st.columns(4)

with col1:

    st.metric(
        "Total Records",
        len(df)
    )

with col2:

    st.metric(
        "Positive",
        (df["Sentiment"]=="Positive").sum()
    )

with col3:

    st.metric(
        "Negative",
        (df["Sentiment"]=="Negative").sum()
    )

with col4:

    st.metric(
        "Neutral",
        (df["Sentiment"]=="Neutral").sum()
    )

st.divider()

# ==========================
# COMMENT INPUT
# ==========================

st.subheader("💬 Analyze Comments")

comments_input = st.text_area(

"Enter one comment per line",

height=220,

placeholder="""I love this phone.
Worst product ever.
Amazing Camera.
Battery is average."""

)

analyze_btn = st.button("🚀 Analyze Sentiment")

# =====================================================
# PART 2 : SENTIMENT PREDICTION
# =====================================================

if analyze_btn:

    # ---------------- GET COMMENTS ----------------

    comments = [
        comment.strip()
        for comment in comments_input.split("\n")
        if comment.strip()
    ]

    if len(comments) == 0:

        st.warning("⚠ Please enter at least one comment.")

    else:

        # ---------------- VECTORIZE ----------------

        vectors = vectorizer.transform(comments)

        # ---------------- PREDICT ----------------

        predictions = model.predict(vectors)

        # ---------------- CREATE DATAFRAME ----------------

        result_df = pd.DataFrame({

            "Comment": comments,

            "Predicted Sentiment": predictions

        })

        # ---------------- SAVE FOR POWER BI ----------------

        result_df.to_csv(
            "powerbi_data.csv",
            index=False
        )

        # ---------------- SHOW RESULTS ----------------

        st.divider()

        st.subheader("📋 Prediction Results")

        st.dataframe(
            result_df,
            use_container_width=True,
            height=350
        )

        # ---------------- SUMMARY ----------------

        positive = (
            result_df["Predicted Sentiment"] == "Positive"
        ).sum()

        negative = (
            result_df["Predicted Sentiment"] == "Negative"
        ).sum()

        neutral = (
            result_df["Predicted Sentiment"] == "Neutral"
        ).sum()

        st.write("")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.success(f"😊 Positive : {positive}")

        with c2:
            st.error(f"😔 Negative : {negative}")

        with c3:
            st.warning(f"😐 Neutral : {neutral}")

        # ---------------- DOWNLOAD CSV ----------------

        csv = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(

            "📥 Download Prediction Report",

            csv,

            file_name="sentiment_prediction.csv",

            mime="text/csv"

        )
        
# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
"""
<div style='text-align:center;
padding:20px;
background:#111827;
border-radius:10px;'>

<h4 style='color:white;'>
📊 Real-Time Social Media Sentiment Analysis Dashboard
</h4>

<p style='color:lightgray;'>
Developed using Python | Machine Learning | Streamlit | Power BI
</p>

</div>
""",
unsafe_allow_html=True
)
