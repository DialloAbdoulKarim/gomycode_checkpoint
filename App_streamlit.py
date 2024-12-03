import streamlit as st
import joblib
import pandas as pd

# Load the trained model
try:
    model = joblib.load('logistic_model.pkl')
    if not hasattr(model, 'predict'):
        st.error("The loaded model does not support predictions. Ensure you have loaded the correct model.")
except Exception as e:
    st.error(f"Error loading the model: {e}")

# Title of the app
st.title("Bank Account Ownership Prediction")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Function to get user input
def get_user_input():
    location_type = st.sidebar.selectbox('Location Type', ['Urban', 'Rural'])
    age_of_respondent = st.sidebar.slider('Age of Respondent', 18, 100, 30)
    gender_of_respondent = st.sidebar.selectbox('Gender of Respondent', ['Male', 'Female'])

    # Example dynamic feature creation
    job_types = ['job_type_government', 'job_type_private', 'job_type_self_employed']
    job_type_inputs = {job: st.sidebar.checkbox(job, False) for job in job_types}

    # Collect data into a dictionary
    data = {
        'location_type': 1 if location_type == 'Urban' else 0,
        'age_of_respondent': age_of_respondent,
        'gender_of_respondent': 1 if gender_of_respondent == 'Male' else 0,
    }
    data.update(job_type_inputs)

    return pd.DataFrame([data])

# Get user input
user_input = get_user_input()

# Align user_input with the training data
if model is not None and hasattr(model, 'feature_names_in_'):
    user_input = user_input.reindex(columns=model.feature_names_in_, fill_value=0)

# Display user input
st.write("User Input Features:")
st.write(user_input)

# Predict when the "Predict" button is clicked
if st.button("Predict"):
    if model is not None and hasattr(model, 'predict'):
        # Make predictions
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        # Display the prediction
        st.write("Prediction (Bank Account Ownership):", "Yes" if prediction[0] == 1 else "No")
        st.write("Prediction Probability:", prediction_proba[0][1])  # Display probability for the 'Yes' class
    else:
        st.error("Prediction cannot be made because the model is not loaded correctly.")

