import numpy as np

def Gender(input):

    if input == 'Male':
        return 0
    else:
        return 1
    
def Married(input):

    if input == 'No':
        return 0
    else:
        return 1
    
def Education(input):

    if input == "Graduate":
        return 0
    else:
        return 1
    
def Self_Employed(input):

    if input == "No":
        return 0
    else:
        return 1
    
def Property_Area(input):

    if input == "Urban":
        return 0
    elif input == "Rural":
        return 1
    else:
        return 2
    
def Transformation(applicant_income, coapplicant_income, loan_amount):

    applicant_income = np.log(applicant_income)
    coapplicant_income = np.cbrt(coapplicant_income)
    loan_amount = np.log(loan_amount)

    return applicant_income, coapplicant_income, loan_amount
