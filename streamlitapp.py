from flask import Flask,request,jsonify
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from utils import Gender,Married,Education,Self_Employed,Property_Area,Transformation


rf_model_hyp = pickle.load(open('rf_model_hyp.pkl', 'rb'))
columns_list = pickle.load(open('columns_list.obj', 'rb'))


# Creating the title for the app
st.set_page_config(page_title="Loan Status",
                   layout='centered',
                   initial_sidebar_state='collapsed')

# Creating header
st.header("Loan Status Predictor")

col1, col2 = st.columns([5,5])
with col1:
    gender = st.selectbox("Gender",('Male', 'Female'))
with col2:
    married = st.selectbox("Are you Married?",('Yes', 'No'))

col1, col2 = st.columns([5,5])
with col1:
    dependents = st.number_input("Number of dependents")
with col2:
    education = st.selectbox("Qualification",('Graduate', 'Non-Graduate'))

col1, col2 = st.columns([5,5])
with col1:
    self_employed = st.selectbox("Are you self-employed?", ('Yes', 'No'))
with col2:
    applicant_income = st.number_input("Enter applicant income")

col1, col2 = st.columns([5,5])
with col1:
    coapplicant_income = st.number_input("Enter co-applicant income")
with col2:
    loan_amount = st.number_input("Enter loan amount")

col1, col2 = st.columns([5,5])
with col1:
    loan_amount_term = st.number_input("Enter loan amount term")
with col2:
    credit_history = st.selectbox("Have credit history?",('Yes', 'No'))
    if credit_history=='Yes':
        credit_history=1
    else:
        credit_history=0

property_area = st.selectbox("Select from below", ('Urban', 'Rural', 'Semiurban'))


gender = Gender(gender)
married = Married(married)
dependents = int(dependents)
education = Education(education)
self_employed = Self_Employed(self_employed)
loan_amount_term = int(loan_amount_term)
credit_history = credit_history
property_area = Property_Area(property_area)

# Creating submit button
submit = st.button("Predict")


# Final Response:
if submit:
    with st.spinner("Loading please wait..."):
        
        # Creating Data Frame here
        test_df = pd.DataFrame([[gender, married, dependents, education, self_employed, np.log(applicant_income), np.cbrt(coapplicant_income),
                                        np.log(loan_amount), loan_amount_term, credit_history, property_area]], columns=columns_list)

        prediction = rf_model_hyp.predict(test_df)

        if prediction[0]==1:
            print("Approved")
            st.write("Approved")
        else:
            print("Declined")
            st.write("Declined")
        
        
