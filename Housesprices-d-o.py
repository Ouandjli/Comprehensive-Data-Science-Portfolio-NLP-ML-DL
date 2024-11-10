from __future__ import print_function
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#display detect outliers

FileName ='HousePricesD-O.csv'
df = pd.read_csv(FileName)
print(df.head())

#FIT model 1(Rooms)
#with outliers
np.random.seed(110)
T = 1000
X = np.random.normal(0,1,T)
sigma = np.std(X) # usual standard deviation
sigma_r = 1.4826 * np.median(np.abs(X)) # MAD robust estimator
print("Without outlier: sigma = %1.2f; sigma_r = %1.2f"%(sigma,sigma_r))

# example with outlier
X[300] = 10000.
sigma = np.std(X)
sigma_r = 1.4826 * np.median(np.abs(X))
print("WITH outlier: sigma = %1.2f; sigma_r = %1.2f"%(sigma,sigma_r))
print("***************************************************************************************************************")

#detect outliers
median = np.median(df['Rooms'].values)
sigma_r = 1.4826 * np.median(np.abs(df['Rooms'].values - median))
print("median=",median,"sigma_r=",sigma_r)
X1 = np.abs(df['Rooms'].values - median)
OutliersIndices, = np.where(X1 > 4 * sigma_r)
print("OutliersIndices:",OutliersIndices)
print("Removing outliers...")
df = df.drop(OutliersIndices)
X = df['Rooms']
X = sm.add_constant(X)
Y = df['Price(USD)']
# Fit the model
model = sm.OLS(Y, X).fit()
# Print the summary of the model
print(model.summary())


#detect outliers model 2
median = np.median(df['Parking'].values)
sigma_r = 1.4826 * np.median(np.abs(df['Parking'].values - median))
print("median=",median,"sigma_r=",sigma_r)
X2 = np.abs(df['Parking'].values - median)
OutliersIndices, = np.where(X2 > 4 * sigma_r)
print("OutliersIndices:",OutliersIndices)
print("Removing outliers...")
X = df['Parking']
X = sm.add_constant(X)
Y = df['Price(USD)']
# Fit the model
model = sm.OLS(Y, X).fit()
# Print the summary of the model
print(model.summary())

#fit model 3
median = np.median(df['Elevator'].values)
sigma_r = 1.4826 * np.median(np.abs(df['Elevator'].values - median))
print("median=",median,"sigma_r=",sigma_r)
X3 = np.abs(df['Elevator'].values - median)
OutliersIndices, = np.where(X3 > 4 * sigma_r)
print("OutliersIndices:",OutliersIndices)
print("Removing outliers...")
X = df['Elevator']
X = sm.add_constant(X)
Y = df['Price(USD)']
# Fit the model
model = sm.OLS(Y, X).fit()
# Print the summary of the model
print(model.summary())

#calculate percentile
Y=df['Price(USD)']
# Calcul du percentile pour un seuil de 0,1% & 0,99%
seuil_001 = 0.1
seuil_099 = 99.0
percentile_001 = np.percentile(Y, seuil_001)
percentile_099 = np.percentile(Y, seuil_099)
print("Percentile pour un seuil de 0,1% :", percentile_001)
print("Percentile pour un seuil de 99,0% :", percentile_099)

#verification des valeurs manquantes(missing value checks)
df.isnull().sum()
df.shape
print(df)


# Boxplot visualisation
sns.set(style='whitegrid')
sns.boxplot(x='Rooms', y='Price(USD)', data=df)
plt.xlabel('Rooms')
plt.ylabel('Price (USD)')
plt.title('Boxplot of House Prices by Number of Rooms')
plt.show()












