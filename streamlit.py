import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and label encoders
model = joblib.load("regression_model.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Function to apply label encoding to the input data
def encode_input_data(input_df, label_encoders):
    categorical_columns = ['CampaignTag', 'MediaFormat', 'MediaType', 'AudienceInterest', 'UserCategory', 
                           'City', 'AgeRange', 'Gender', 'InterestMajor']
    
    for col in categorical_columns:
        if col in input_df:
            input_df[col] = input_df[col].astype(str)  # Ensure the column is of type string
            # Check if the encoder exists for the column
            if col in label_encoders:
                encoder = label_encoders[col]
                # Handle unseen labels by assigning a default encoding (e.g., -1 or the most frequent label)
                input_df[col] = input_df[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    return input_df

# Streamlit UI
st.title("TikTok Ad Campaign Predictor")

# User input fields for the model
campaign_tag = st.text_input('Campaign Tag', '#StudyTips')
media_format = st.text_input('Media Format', 'LiveStream')
media_type = st.text_input('Media Type', 'Promotional')
audience_interest = st.text_input('Audience Interest', 'Cooking')
user_category = st.text_input('User Category', 'Alumni')
engagement_likes = st.number_input('Engagement Likes', min_value=0, max_value=10000, value=103)
engagement_shares = st.number_input('Engagement Shares', min_value=0, max_value=10000, value=62)
engagement_comments = st.number_input('Engagement Comments', min_value=0, max_value=10000, value=194)
impressions = st.number_input('Impressions', min_value=0, max_value=10000, value=5464)
city = st.text_input('City', 'Bekasi')
age_range = st.selectbox('Age Range', ['18-24', '25-34', '35-44', '45-54', '55+'])
gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
interest_major = st.selectbox('Interest Major', ['Technology', 'Sports', 'Business'])
intent_score = st.number_input('Intent Score', min_value=1, max_value=5, value=3)
estimated_loan_disbursal = st.number_input('Estimated Loan Disbursal', min_value=0, max_value=100, value=31)

# Create DataFrame for the input data
input_data = {
    'CampaignTag': [campaign_tag],
    'MediaFormat': [media_format],
    'MediaType': [media_type],
    'AudienceInterest': [audience_interest],
    'UserCategory': [user_category],
    'EngagementLikes': [engagement_likes],
    'EngagementShares': [engagement_shares],
    'EngagementComments': [engagement_comments],
    'Impressions': [impressions],
    'City': [city],
    'AgeRange': [age_range],
    'Gender': [gender],
    'InterestMajor': [interest_major],
    'IntentScore': [intent_score],
    'EstimatedLoanDisbursal': [estimated_loan_disbursal]
}

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data)

# Drop the problematic column 'EstimatedLoanDisbursal' if it causes errors
input_df = input_df.drop(columns=['EstimatedLoanDisbursal'], errors='ignore')

# Apply label encoding
input_df = encode_input_data(input_df, label_encoders)

# Make prediction using the model
if st.button('Predict'):
    try:
        prediction = model.predict(input_df)
        st.write(f"Predicted Outcome: {prediction[0]}")
    except ValueError as e:
        st.error(f"Error with prediction: {str(e)}")
