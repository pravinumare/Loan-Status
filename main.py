from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
import pickle
from utils import Gender,Married,Education,Self_Employed,Property_Area,Transformation

app = Flask(__name__)

rf_model_hyp = pickle.load(open('rf_model_hyp.pkl', 'rb'))
columns_list = pickle.load(open('columns_list.obj', 'rb'))

@app.route('/loanprediction')
def loanprediction():

    data = request.get_json()

    gender = data['Gender']
    married = data['Married']
    dependents = data['Dependents']
    education = data['Education']
    self_employed = data['Self_Employed']
    applicant_income = data['ApplicantIncome']
    coapplicant_income = data['CoapplicantIncome']
    loan_amount = data['LoanAmount']
    loan_amount_term = data['Loan_Amount_Term']
    credit_history = data['Credit_History']
    property_area = data['Property_Area']

    gender = Gender(gender)
    married = Married(married)
    education = Education(education)
    self_employed = Self_Employed(self_employed)
    applicant_income, coapplicant_income, loan_amount = Transformation(applicant_income, coapplicant_income, loan_amount)
    property_area = Property_Area(property_area)

    test_df = pd.DataFrame([[gender, married, dependents, education, self_employed, applicant_income, coapplicant_income,
    loan_amount, loan_amount_term, credit_history, property_area]], columns=columns_list)

    prediction = rf_model_hyp.predict(test_df)
    

    return jsonify({"Loan_Status" : int(prediction[0])})

    

if __name__ == '__main__':
    app.run()