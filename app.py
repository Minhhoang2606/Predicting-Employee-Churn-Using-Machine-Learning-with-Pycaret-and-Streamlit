import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load('employee_churn_model.pkl')
# Title and description
st.title('Employee Churn Prediction App')
st.write('Predict whether an employee is likely to leave the organization.')
# Single prediction input form
st.header('Single Prediction')
satisfaction_level = st.number_input('Satisfaction Level (0-1)', min_value=0.0, max_value=1.0, step=0.01)
last_evaluation = st.number_input('Last Evaluation (0-1)', min_value=0.0, max_value=1.0, step=0.01)
number_project = st.number_input('Number of Projects', min_value=1, step=1)
average_montly_hours = st.number_input('Average Monthly Hours', min_value=0, step=1)
time_spend_company = st.number_input('Years at Company', min_value=0, step=1)
department = st.selectbox('Department', ['sales', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'RandD', 'hr', 'accounting'])
salary = st.selectbox('Salary Level', ['low', 'medium', 'high'])
# Predict button
if st.button('Predict Single Employee'):
    input_data = pd.DataFrame({
        'satisfaction_level': [satisfaction_level],
        'last_evaluation': [last_evaluation],
        'number_project': [number_project],
        'average_montly_hours': [average_montly_hours],
        'time_spend_company': [time_spend_company],
        'department': [department],
        'salary': [salary]
    })
    prediction = model.predict(input_data)[0]
    result = 'Employee is likely to leave.' if prediction == 1 else 'Employee is likely to stay.'
    st.write(result)
# Batch prediction
st.header('Batch Prediction')
uploaded_file = st.file_uploader('Upload a CSV file for batch prediction', type=['csv'])
if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    predictions = model.predict(batch_data)
    batch_data['Prediction'] = predictions
    st.write(batch_data)
    st.download_button('Download Predictions', batch_data.to_csv(index=False), file_name='predictions.csv')