import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import random

# Configure Streamlit page
st.set_page_config(page_title="Loan Disbursal Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model():
    model = joblib.load(r"regression_model.joblib")
    return model

model = load_model()

# Sidebar input form
st.sidebar.title("📊 Input Features")

# Display app description in light grey at the bottom
st.markdown(
    "<hr style='margin-top: 50px;'>"
    "<p style='text-align: center; color: grey;'>"
    "This ad campaign tracker predicts estimated loan disbursal based on user engagement metrics and audience demographics, helping marketers assess campaign performance and optimize future targeting strategies in real-time."
    "</p>",
    unsafe_allow_html=True
)

# Instructions
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown("- `Impressions`: Total number of times the ad was viewed. Typically between 1000 - 100000.")
st.sidebar.markdown("- `EngagementLikes`, `Shares`, `Comments`: Engagement metrics. Usually between 10 - 10000.")

# Inputs
likes = st.sidebar.number_input("Engagement Likes", min_value=0, step=1, value=100)
impressions = st.sidebar.number_input("Impressions", min_value=0, step=100, value=5000)
shares = st.sidebar.number_input("Engagement Shares", min_value=0, step=1, value=50)
comments = st.sidebar.number_input("Engagement Comments", min_value=0, step=1, value=20)

# Dropdowns for categorical features
audience_options = ['Cooking', 'IT', 'Education', 'Gadget', 'Sports', 'Automotive', 'Travel', 'Comedy', 'Music', 'Drama']
interest_options = ['Business', 'Law', 'Arts', 'Engineering', 'Medicine', 'Science']
city_options = ['Bekasi', 'Depok', 'Bandung', 'Surabaya', 'Tangerang', 'Palembang', 'Medan', 'Jakarta', 'Semarang', 'Makassar']

audience = st.sidebar.selectbox("Audience Interest", audience_options)
major = st.sidebar.selectbox("Interest Major", interest_options)
city = st.sidebar.selectbox("City", city_options)

# Random label encoder for each category (reshuffled each run)
def encode_random(df, col, options):
    encoder = LabelEncoder()
    shuffled = options[:]
    random.shuffle(shuffled)
    encoder.fit(shuffled)
    df[col] = encoder.transform(df[col])
    return df

# Predict button
if st.sidebar.button("🚀 Predict Loan Disbursal"):
    # Prepare input
    input_data = {
        'EngagementLikes': likes,
        'Impressions': impressions,
        'EngagementShares': shares,
        'EngagementComments': comments,
        'AudienceInterest': audience,
        'InterestMajor': major,
        'City': city
    }

    df = pd.DataFrame([input_data])

    # Encode categorical features with randomly shuffled encoding
    df = encode_random(df, 'AudienceInterest', audience_options)
    df = encode_random(df, 'InterestMajor', interest_options)
    df = encode_random(df, 'City', city_options)

    # Ensure correct order of columns
    df = df[['EngagementLikes', 'Impressions', 'EngagementShares', 'EngagementComments',
             'AudienceInterest', 'InterestMajor', 'City']]

    # Predict
    prediction = model.predict(df)[0]

    # Show result
    st.success(f"💰 **Estimated Loan Disbursal: SAR {prediction * 1000:,.0f}**")

    # Show back the actual category values (not encoded)
    st.markdown("🔍 **Prediction based on:**")
    st.markdown(f"- **Audience Interest**: `{audience}`")
    st.markdown(f"- **Interest Major**: `{major}`")
    st.markdown(f"- **City**: `{city}`")
