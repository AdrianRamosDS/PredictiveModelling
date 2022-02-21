# -*- coding: utf-8 -*-
"""
@author: Adrian Ramos
"""
#%% APIs and packages imports
import pandas as pd
from pandas_profiling import ProfileReport 
import matplotlib.pyplot as plt
import seaborn as sns


#%% Dataframe extraction
df = pd.read_csv("soybean-large.data", header = None)
names = ["Class","Date", "Plant-stand","Precip","Temp","Hail","Crop-hist","Area-damaged",
         "Severity","Seed-tmt","Germination", "Plant-growth","Leaves","Leafspots-halo",
         "Leafspots-marg","Leafspots-size","Leaf-shread","leaf-malf","leaf-mild","Stem","Lodging",
         "Stem-cankers","Canker-lesion","Fruiting-bodies","External-decay","Mycelium",
         "Int-discolor","Sclerotia","Fruit-pods","Fruit-spots","Seeds","Mold-growth",
         "Seed-discolor","Seed-size","Shriveling","Roots"]
df.columns = names


#%% Data Quality Report

def dqr():



print(df.info())
print(df.describe(include='all'))
# pp_report = ProfileReport(df)
#pp_report.to_file(output_file="HW1_AdrianRamos.html")

print(df.mean())
print(df.mode())
#%% Preprocessing
# Dataframe with Nan substitution by 0:
df_nan_0 = df.replace("?",0) 
df_nan_mode = df
df['Class'] = df['Class'].astype(str)
    
for col in list(df.columns[1:]):
    df_nan_0[col] = df_nan_0[col].astype(str).astype(int)
    
    df_nan_mode = df.replace("?",df[col].mode()[0])
    df_nan_mode[col] = df_nan_mode[col].astype(str).astype(int)
   #df_nan_mode[col] = df_nan_mode[col].fillna(df[col].mode()[0], inplace=False)
    
    plt.figure()
    plt.xlabel(col),plt.ylabel('val')
    df[col].hist()
    plt.savefig(col)
    plt.show()
#%% Distributions and Histograms
for col in list(df_nan_mode.columns[1:]):
    plt.figure() 
    
    plt.subplot(2,1,1)
    plt.xlabel(col),plt.ylabel('Specimen No.')
    plt.title('NaN filled with 0s:')
    plt.grid(True)
    plt.hist(df_nan_0[col])
    
    plt.subplot(2,1,2)
    plt.xlabel(col),plt.ylabel('Specimen No.')
    plt.title('NaN filled with mode:')
    plt.grid(True)
    plt.hist(df_nan_mode[col])
    
    plt.tight_layout()
    # plt.scatter(df[col],len(df)-1)
# sns.pairplot(df)

#%% 2. Skewness checking
# skewness = pd.DataFrame()
# skewness['NaN=0']= df_nan_0.skew()
# skewness['NaN=mode'] = df_nan_mode.skew()

# skew = []
# for col in list(df_nan_mode.columns[1:]):
#     skew.append(df_nan_0[col].max()/df_nan_0[col].min()+0.01)

#%% 3. Outlier detection
# for col in list(df_nan_mode.columns[1:]):
#     sns.boxplot(y=df_nan_0[col])
# sns.boxplot(y=data['Refractive_index_scale'])


#%% Function to determine the outliers
def find_boundaries(df_var,distance=1.5):
    IQR = df_var.quantile(0.75)-df_var.quantile(0.25)
    lower = df_var.quantile(0.25)-IQR*distance
    upper = df_var.quantile(0.75)+IQR*distance
    return lower,upper

# lmin,lmax = find_boundaries(data['Refractive_index'])
# outliers = np.where(data['Refractive_index'] > lmax, True,np.where(data['Refractive_index'] < lmin, True, False))
# outliers_df = data.loc[outliers, 'Refractive_index']
