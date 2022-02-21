# -*- coding: utf-8 -*-
"""
INSTITUTO TECNOLOGICO Y DE ESTUDIOS SUPERIORES DE OCCIDENTE
@Program: Master in Data Science
@Subject: Predictive Modeling
@Author: Adrian Ramos Perez
@Homework title: PCA vs PLS for Regression

@Database from UCI Machine Learning:
    https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
"""
#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%% Data extraction
col_names = ['Frequency','Angle','Chord_length','Free-stream_velocity','Suction','Sound_Pressure_Level']
data = pd.read_table('../Data/airfoil_self_noise.dat', names=col_names, header=None)
data_desc = data.describe()

#%% 1. Determination of missin data:
miss_vals = pd.DataFrame(data.isnull().sum(), columns=['Missing_values'])
# According to evaluation there's no missing values.

#%% 2. Dataset split into train and test subsets:
from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(data.iloc[:,0:5], data.Sound_Pressure_Level, test_size=0.2, random_state=50)

#%% 3. Train a linear model to estimate the feature "Sound Pressure Level" using as inputs to the model all other variables.
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)
y_predicted = lin_reg.predict(x_test)

# Get the values of RMSE and R^2 to evaluate the performance of the model in both training and testing.
from sklearn.metrics import (mean_squared_error, r2_score)

rmse = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

fig = plt.figure(figsize=(10,8))
plt.scatter(y_test, y_predicted)
# Plot reference line:
ref = np.linspace(min(y_test),max(y_test))
plt.plot(ref,ref,'k--')
# into a square:
plt.axis('square')
plt.xlabel('real y')
plt.ylabel('predicted y')
plt.title('Linear regression (original), RMSE=%0.4f, R^2=%0.4f'%(rmse,r2))
plt.grid()
fig.savefig('../Figures/Hw2_3_original_LinReg.png')
#%% 4. Considering the data set used in point 3, perform elimination of some variables using variance criterion or correlation criterion.
# Variance criterion of less variance are 'Suction' and 'Free-stream_velocity':
var_criterion = data.var()
print("Variance of features:\n\n", var_criterion,end="")
data_var_crit = data.drop(columns=['Chord_length','Suction'])

# With the variables resulting from the elimination process, train a new linear model and calculate the metrics 
X_train_vc, x_test_vc,  = X_train.drop(columns=['Chord_length','Suction']), x_test.drop(columns=['Chord_length','Suction'])

lin_reg_vc = LinearRegression()
lin_reg_vc.fit(X_train_vc, Y_train)
y_predicted_vc = lin_reg_vc.predict(x_test_vc)

rmse_vc = mean_squared_error(y_test, y_predicted_vc)
r2_vc = r2_score(y_test, y_predicted_vc)

fig = plt.figure(figsize=(10,8))
plt.scatter(y_test,y_predicted_vc)
plt.plot(ref,ref,'k--')
plt.axis('square')
plt.xlabel('Real y'),plt.ylabel('Predicted y')
plt.title('Linear regression (Variance Criterion), RMSE=%0.4f, R^2=%0.4f'%(rmse_vc,r2_vc))
plt.grid()
fig.savefig('../Figures/Hw2_4_Variance_Criterion_LinReg.png')

# Using sklearn's feature selection:
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=1)
sel.fit_transform(data)

#%% 5. Variable reduction by Principal Component Analysis (PCA). Training new linear model
from sklearn.decomposition import PCA

pca = PCA()
# Train and transform with all the data because at first we don't know which are the principal components
pca.fit(data.iloc[:,0:5])
data_pca = pca.transform(data.iloc[:,0:5])
data_pca = pd.DataFrame(data_pca, columns=['x1*','x2*','x3*','x4*','x5*'])
# Real y remains the same:
data_pca['y'] = data['Sound_Pressure_Level']


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca['x1*'], data_pca['x2*'], data_pca['x3*'])
ax.set_xlabel('X1*')
ax.set_ylabel('X2*')
ax.set_zlabel('X3*')
plt.show()
fig.savefig('../Figures/Hw2_5_PCA_3Dplot.png')

fig = sns.pairplot(data_pca,x_vars=['x1*','x2*','x3*','x4*','x5*'],
             y_vars=['y'])
fig.savefig('../Figures/Hw2_5_PCA_Components.png')

# Linear Regression with PCA data:
    
X_train, X_test, y_train, y_test = train_test_split(data_pca[['x1*','x2*','x3*','x4*','x5*']], data_pca.y,
                                                    test_size=0.2, random_state=42)
lin_reg_pca = LinearRegression()
# linreg.fit(np.array(X_train['x1*']).reshape(80,1), y_train) # case one-dimensional 
# linreg.fit(np.array(X_train[['x1*','x2*']]), y_train) # case two-dimensional 
lin_reg_pca.fit(np.array(X_train[['x1*','x2*','x3*']]), y_train) # case three-dimensional

ref = np.linspace(min(y_test),max(y_test))

# y_predict = linreg.predict(np.array(X_test['x1*']).reshape(20,1)) # case one-dimensional 
# y_predict = linreg.predict(np.array(X_test[['x1*','x2*']])) # case two-dimensional


y_predicted_pca = lin_reg_pca.predict(np.array(X_test[['x1*','x2*','x3*']])) # case three-dimensional
rmse_pca = mean_squared_error(y_test, y_predicted_pca)
r2_pca = r2_score(y_test, y_predicted_pca)

fig = plt.figure(figsize=(10,8))
plt.scatter(y_test,y_predicted_pca)
plt.plot(ref,ref,'k--')
plt.axis('square')
plt.xlabel('y real'),plt.ylabel('y predict')
plt.title('PCA regression, RMSE=%0.4f, R^2=%0.4f'%(rmse_pca,r2_pca ))
plt.grid()
fig.savefig('../Figures/Hw2_5_PCA_LinReg.png')

#%% 6. USING PARTIAL LEAST SQUARES (PLS) TECHNIQUE:
from sklearn.cross_decomposition import PLSRegression
    
pls = PLSRegression(n_components=5)
pls.fit(X_train, Y_train)
data_pls = pls.transform(data.iloc[:,0:5])
data_pls = pd.DataFrame(data_pls,columns=['x1*','x2*','x3*','x4*','x5*'])
data_pls['y'] = data['Sound_Pressure_Level']

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pls['x1*'], data_pls['x2*'], data_pls['x3*'])
ax.set_xlabel('X1*')
ax.set_ylabel('X2*')
ax.set_zlabel('X3*')
plt.show()
fig.savefig('../Figures/Hw2_6_PLS_3Dplot.png')

fig = sns.pairplot(data_pls,x_vars=['x1*','x2*','x3*','x4*','x5*'],
             y_vars=['y'])
fig.savefig('../Figures/Hw2_6_PLS_Components.png')
  
# PLS applied to Linear Regression model:
    
pls = PLSRegression(n_components=5)
pls.fit(X_train, y_train)
y_predicted_pls = pls.predict(X_test)

'''
T: x_scores_
U: y_scores_
W: x_weights_
C: y_weights_
P: x_loadings_
Q: y_loadings_
'''


rmse_pls = mean_squared_error(y_test, y_predicted_pls)
r2_pls = r2_score(y_test, y_predicted_pls)

fig = plt.figure(figsize=(10,8))
plt.scatter(y_test,y_predicted_pls)
plt.plot(ref,ref,'k--')
plt.axis('square')
plt.xlabel('y real'),plt.ylabel('y predict')
plt.title('PLS regression, RMSE=%0.4f, R^2=%0.4f'%(rmse_pls,r2_pls))
plt.grid()
fig.savefig('../Figures/Hw2_6_PLS_LinReg.png')
    
#%% 7. Table with metrics of each model to make a compariosn:
results = {'Linear Regression':[rmse,r2],'Variable Elimination':[rmse_vc,r2_vc],'PCA':[rmse_pca,r2_pca],'PLS':[rmse_pls,r2_pls]}
results_df = pd.DataFrame(results, index =['RMSE','R2'])
    

