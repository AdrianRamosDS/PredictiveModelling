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
data.describe()

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
fig.savefig('../Figures/Hw2_3_original_LR.png')
#%% 4. Considering the data set used in point 3, perform elimination of some variables using variance criterion or correlation criterion.
# Variance criterion:
print("Variance of x variables:\n\n", data.var())


# With the variables resulting from the elimination process, train a new linear model and calculate the metrics 


#%% 5. Variable reduction by Principal Component Analysis (PCA). Training new linear model
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(data.iloc[:,0:5])
data_pca = pca.transform(data.iloc[:,0:5])
data_pca = pd.DataFrame(data_pca, columns=['Frequency*','Angle*','Chord_length*','Free-stream_velocity*','Suction*'])
# Real y remains the same:
data_pca['Sound_Pressure_Level'] = data['Sound_Pressure_Level']


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_pca['Frequency*'], data_pca['Angle*'], data_pca['Chord_length*'])
ax.set_xlabel('Frequency*')
ax.set_ylabel('Angle*')
ax.set_zlabel('Chord_length*')
plt.show()
# fig.savefig('../figures/P2_fig/F3.png')

fig = sns.pairplot(data_pca,x_vars=['Frequency*','Angle*','Chord_length*'],
             y_vars=['Sound_Pressure_Level'])
fig.savefig('../Figures/Hw2_5_PCA_LR.png')


#%% 6.