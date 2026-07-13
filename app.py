import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model_rf = joblib.load('random_forest_regressor_model.pkl')

# Assuming these are the feature columns your model was trained on
# It's good practice to get this from the training data or a config file
# For simplicity, we're hardcoding based on previous notebook steps.
feature_columns = ['Year', 'Cylinder', 'Milage'] 

def predict_price(year, cylinder, milage):
    # Create an empty dataframe with the same columns as the training data
    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)

    # Populate the input_df with the values from the Streamlit UI
    input_df['Year'] = year
    input_df['Cylinder'] = cylinder
    input_df['Milage'] = milage

    # Make prediction
    prediction = model_rf.predict(input_df)[0]

    return f"Estimated Car Price: ${prediction:,.2f}"

# Streamlit UI
st.title('🚗 Car Price Prediction App')
st.write('Enter the car details to get an estimated price.')

# Input widgets
year_input = st.slider('Year', min_value=1990, max_value=2026, value=2025)
cylinder_input = st.slider('Cylinder', min_value=3, max_value=8, value=4)
milage_input = st.slider('Milage', min_value=0, max_value=300000, value=50000)

if st.button('Predict Price'):
    result = predict_price(year_input, cylinder_input, milage_input)
    st.success(result)
