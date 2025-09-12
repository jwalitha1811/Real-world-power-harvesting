import streamlit as st
import pandas as pd
import joblib
from datetime import datetime


# Load trained model and encoder

model = joblib.load('footstep_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')


# App Title

st.title('Real-World Pedal Energy Harvesting App')
st.markdown('Enter the conditions and date-time for the energy harvesting')


# User inputs

selected_date = st.date_input("Select a date")
selected_time = st.time_input("Select a time")
combined_datetime = datetime.combine(selected_date, selected_time)
st.write("You selected:", combined_datetime)

# Inputs with SAME feature names used in training
voltage = st.number_input('voltage(v)', min_value=0.0, step=0.1, format="%.2f")
current = st.number_input('current(uA)', min_value=0.0, step=0.1, format="%.2f")
weight = st.number_input('weight(kgs)', min_value=0.0, step=0.1, format="%.2f")
step_location = st.number_input('step_location', min_value=0.0, step=0.1, format="%.2f")


# Prediction

if st.button('Predict power'):
    try:
        input_df = pd.DataFrame([[
            voltage,
            current,
            weight,
            step_location
        ]], columns=['voltage', 'current', 'weight', 'step_location'])

        
        predicted_power = model.predict(input_df)

       
        try:
            predicted_power = label_encoder.inverse_transform(predicted_power)
        except Exception:
            pass

        st.success(f'Predicted Power is {predicted_power[0]:.2f} kW')

    except ValueError as ve:
        st.error(f"ValueError occurred: {ve}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
