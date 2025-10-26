import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load saved scaler, PCA and model
with open('scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('random_forest.pkl', 'rb') as f:
    model_rf = pickle.load(f)

# Feature names expected for input (same as training)
features = [
    'price',
    'minimum_nights',
    'number_of_reviews',
    'reviews_per_month',
    'calculated_host_listings_count',
    'availability_365'
]

st.title('NYC Airbnb Room Type Prediction')

# Get user input for features
input_data = {}
for feature in features:
    val = st.number_input(f'Input {feature}', min_value=0.0, value=0.0)
    input_data[feature] = val

# Create dataframe from input
input_df = pd.DataFrame([input_data])

# Scale, then apply PCA
scaled_data = scaler.transform(input_df)
pca_data = pca.transform(scaled_data)

# Predict room_type
prediction = model_rf.predict(pca_data)[0]

# Map prediction back to label (if you have mapping)
room_type_mapping = {0: 'Entire home/apt', 1: 'Private room', 2: 'Shared room', 3: 'Other'}
predicted_label = room_type_mapping.get(prediction, "Unknown")

st.write(f'Predicted Room Type: {predicted_label}')
