import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load('fin_inc.joblib')
le = joblib.load('fin_inc_le.joblib')

st.title('Financial Inclusion Prediction')
st.write('This app predicts the financial inclusion status of individuals based on various features.')

location_type = st.selectbox('Location Type', ['Urban', 'Rural'])
cellphone_access = st.selectbox('Cellphone Access', ['Yes', 'No'])
age_of_respondent = st.number_input('Age of Respondent', min_value=0, max_value=120, value=30)
gender_of_respondent = st.selectbox('Gender of Respondent', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Married/Living together', 'Widowed', 'Single/Never Married', 'Divorced/Seperated', 'Dont know'])
education_level = st.selectbox('Education Level', ['Secondary education', 'No formal education', 'Vocational/Specialised training', 'Primary education', 'Tertiary education', 'Other/Dont know/RTA'])
job_type = st.selectbox('Job Type', ['Self employed', 'Government Dependent', 'Formally employed Private', 'Informally employed', 'Formally employed Government', 'Farming and Fishing', 'Remittance Dependent', 'Other Income', 'Dont Know/Refuse to answer', 'No Income'])


input_data = {
    'location_type': location_type,
    'cellphone_access': cellphone_access,
    'age_of_respondent': age_of_respondent,
    'gender_of_respondent' : gender_of_respondent,
    'marital_status': marital_status,
    'education_level': education_level,
    'job_type': job_type
}


def fin_inc_prediction(input_data):
    input_df = pd.DataFrame([input_data])
    cols = ['location_type', 'cellphone_access', 'gender_of_respondent', 'marital_status', 'education_level', 'job_type']
    for col in cols:
        input_df[col] = le.fit_transform(input_df[col])

    prediction = model.predict(input_df)
    return prediction

if st.button('Predict'):
    prediction = fin_inc_prediction(input_data)
    if prediction == 0:
        st.error('The individual is likely not to have a bank account')
    elif prediction == 1:
        st.success('The individual is likely to have a bank account')
else:
    st.info('Click the button to predict possibilities')