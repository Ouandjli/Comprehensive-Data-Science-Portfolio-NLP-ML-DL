import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats
import statsmodels.api as sm
import pandas as pd
from scipy.stats import f

#Analyse

FileName ='LoanApproval.csv'
df=pd.read_csv(FileName)
print(df.head())



# convertire les valeurs
df['Gender'] = df['Gender'].replace({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].replace({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].replace({'3+': 3})
df['Education'] = df['Education'].replace({'Graduate': 1, 'Not Graduate': 0})
df['Self_Employed'] = df['Self_Employed'].replace({'Yes': 1, 'No': 0})
df['Property_Area'] = df['Property_Area'].replace({'Rural': 0, 'Semiurban': 1, 'Urban':2})
df['Loan_Status'] = df['Loan_Status'].replace({'Y': 1, 'N': 0})
df=df.drop(columns=['Loan_ID'])# pour supprimer une column

print(df.head())

#donnèes manquantes
df_filled=df.fillna(df.median())
df_filled['Dependents']=df_filled['Dependents'].astype(int)


Y = df_filled['Loan_Status']
X = df_filled[['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount',
       'Loan_Amount_Term','Credit_History','Property_Area']]
X = sm.add_constant(X)
# Fit logistic regression model
model = sm.Logit(Y,X)
result = model.fit()
print(result.summary())

Y = df_filled['Loan_Status']
X = df_filled[['Married','Credit_History']]
X = sm.add_constant(X)
# Fit logistic regression model
model = sm.Logit(Y,X)
result = model.fit()
print(result.summary())

#sensitive
beta=result













